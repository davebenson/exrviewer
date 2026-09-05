//! GPU-accelerated implementation of [`crate::Composition::compose`].
//!
//! Each layer's pixels are uploaded to the GPU once, when the file is
//! loaded. From then on, recompositing is just a handful of render passes
//! that draw one full-screen triangle each, so it stays fast regardless of
//! how expensive the equivalent CPU loop would be.
//!
//! This module only needs a `wgpu::Device`/`Queue`: it doesn't care whether
//! those came from a GUI (`egui_wgpu::RenderState`, which lets the result be
//! displayed directly as a GPU texture via `display_view()`) or a headless
//! setup (a CLI tool, which reads the result back to the CPU via
//! `read_display_rgba()` to save it to a file).
//!
//! ## Filters run as their own passes
//!
//! Filters (see the `filters` module) can need a whole image, not just a
//! pixel's own color - `blur` is the obvious example. So each filter is its
//! own render pass, reading one texture and writing another, rather than
//! being folded into the accumulate/resolve math. A layer's filter chain
//! runs (ping-ponging between two scratch textures) before that layer is
//! blended in; the composite's filter chain runs after blending, on the
//! clamped result, before the final format conversion for display.
//!
//! ## How blending maps to the CPU formula
//!
//! The CPU compose loop does, per layer, per pixel:
//! ```text
//! alpha = layer_alpha * level
//! out.rgb += in.rgb * alpha
//! out.a = min(out.a + alpha, 1.0)
//! ```
//! The accumulate pass's fragment shader outputs `in.rgb * alpha, alpha`
//! (i.e. premultiplied), and the color target's blend state is
//! `(One, One, Add)` for both color and alpha, which is exactly the
//! `out += ...` above. The final `min(_, 1.0)` (and the RGB clamp the CPU
//! path applies before running composite filters) happens in the resolve
//! (or clamp) pass, since a floating-point render target doesn't clamp on
//! write the way a `Unorm` target does.

use std::collections::HashMap;

const ACCUMULATE_SHADER: &str = include_str!("shaders/accumulate.wgsl");
const RESOLVE_SHADER: &str = include_str!("shaders/resolve.wgsl");
const CLAMP_SHADER: &str = include_str!("shaders/clamp.wgsl");
const BLIT_SHADER: &str = include_str!("shaders/blit.wgsl");

/// The format used for layer textures, scratch (filter ping-pong) textures,
/// and the HDR accumulation target. `Rgba16Float` (unlike `Rgba32Float`) is
/// guaranteed renderable and blendable on all wgpu backends without opt-in
/// features.
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The format of the final, displayable composite. This matches what
/// `Composition::compose` produces on the CPU (linear bytes, no sRGB
/// encoding), so the two paths look the same.
const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn params_bytes(params: [f32; 4]) -> [u8; 16] {
    let mut bytes = [0_u8; 16];
    for (chunk, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(params) {
        *chunk = value.to_le_bytes();
    }
    bytes
}

/// One layer's pixels, resident on the GPU, plus a pair of scratch textures
/// (same size, `ACCUM_FORMAT`) used to run its filter chain, if it has one.
struct GpuLayer {
    raw_view: wgpu::TextureView,
    level_buffer: wgpu::Buffer,
    scratch_a: wgpu::TextureView,
    scratch_b: wgpu::TextureView,
}

/// A composition's GPU-side render targets, rebuilt whenever the canvas size
/// changes (i.e. whenever a new file is loaded).
struct Targets {
    size: [usize; 2],
    accum_view: wgpu::TextureView,
    /// Wraps `accum_view` with `texture_only_layout`; used by whichever of
    /// `resolve_pipeline`/`clamp_pipeline` runs.
    accum_bind_group: wgpu::BindGroup,
    /// Scratch pair for the composite's own filter chain (canvas-sized).
    composite_scratch_a: wgpu::TextureView,
    composite_scratch_b: wgpu::TextureView,
    display_texture: wgpu::Texture,
    display_view: wgpu::TextureView,
}

/// Owns the GPU pipelines used to composite layers, plus the per-composition
/// resources (layer textures, render targets) for whichever composition was
/// last uploaded.
pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    sampler_bind_group: wgpu::BindGroup,
    /// group1 shape `(texture, uniform)`: the accumulate pass (layer
    /// texture + level) and every filter pass (source texture + params).
    texture_uniform_layout: wgpu::BindGroupLayout,
    /// group1 shape `(texture)`: resolve, clamp, and blit, none of which
    /// take parameters.
    texture_only_layout: wgpu::BindGroupLayout,
    accumulate_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,
    clamp_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    /// One pipeline per [`crate::filters::FilterKind`] (built from its
    /// `shader()`), keyed by `label()`. Every `Filter` instance of a given
    /// kind shares its pipeline; see `filter_pipeline`.
    filter_pipelines: HashMap<&'static str, wgpu::RenderPipeline>,
    layers: Vec<GpuLayer>,
    targets: Option<Targets>,
}

impl GpuCompositor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (sampler_bind_group_layout, sampler_bind_group) = create_sampler(device);
        let (texture_uniform_layout, texture_only_layout) = create_shared_layouts(device);

        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let accumulate_pipeline = create_pipeline(
            device,
            "accumulate",
            ACCUMULATE_SHADER,
            &[&sampler_bind_group_layout, &texture_uniform_layout],
            ACCUM_FORMAT,
            Some(additive_blend),
        );
        let resolve_pipeline = create_pipeline(
            device,
            "resolve",
            RESOLVE_SHADER,
            &[&sampler_bind_group_layout, &texture_only_layout],
            DISPLAY_FORMAT,
            None,
        );
        let clamp_pipeline = create_pipeline(
            device,
            "clamp",
            CLAMP_SHADER,
            &[&sampler_bind_group_layout, &texture_only_layout],
            ACCUM_FORMAT,
            None,
        );
        let blit_pipeline = create_pipeline(
            device,
            "blit",
            BLIT_SHADER,
            &[&sampler_bind_group_layout, &texture_only_layout],
            DISPLAY_FORMAT,
            None,
        );
        let filter_pipelines = crate::filters::ALL_KINDS
            .iter()
            .map(|kind| {
                let pipeline = create_pipeline(
                    device,
                    kind.label(),
                    kind.shader(),
                    &[&sampler_bind_group_layout, &texture_uniform_layout],
                    ACCUM_FORMAT,
                    None,
                );
                (kind.label(), pipeline)
            })
            .collect();

        Self {
            device: device.clone(),
            queue: queue.clone(),
            sampler_bind_group,
            texture_uniform_layout,
            texture_only_layout,
            accumulate_pipeline,
            resolve_pipeline,
            clamp_pipeline,
            blit_pipeline,
            filter_pipelines,
            layers: Vec::new(),
            targets: None,
        }
    }

    /// Uploads every layer of `composition` to the GPU, replacing whatever
    /// was uploaded before. Call this once per loaded file, not per frame.
    pub fn load(&mut self, composition: &crate::Composition) {
        self.layers = composition
            .layers
            .iter()
            .map(|layer| self.upload_layer(layer))
            .collect();
        self.targets = Some(self.create_targets(composition.size));
    }

    /// The render target holding the current composite, if a file has been
    /// loaded. A GUI can register this directly as a texture to display;
    /// stays valid across `compose()` calls (only replaced by `load()`).
    pub fn display_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|targets| &targets.display_view)
    }

    fn upload_layer(&self, layer: &crate::CompositionLayer) -> GpuLayer {
        let [width, height] = layer.size;

        let mut half_pixels = Vec::with_capacity(layer.pixels.len() * 4 * 2);
        for &[r, g, b, a] in layer.pixels.iter() {
            for component in [r, g, b, a] {
                half_pixels.extend_from_slice(&half::f16::from_f32(component).to_le_bytes());
            }
        }

        #[expect(clippy::cast_possible_truncation)]
        let size = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("exrviewer-layer-texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ACCUM_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &half_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                #[expect(clippy::cast_possible_truncation)]
                bytes_per_row: Some(width as u32 * 4 * 2),
                rows_per_image: None,
            },
            size,
        );

        let raw_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let level_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exrviewer-layer-level"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.write_level(&level_buffer, layer.level);

        let (scratch_a, scratch_b) = self.create_scratch_pair(width, height);

        GpuLayer {
            raw_view,
            level_buffer,
            scratch_a,
            scratch_b,
        }
    }

    fn write_level(&self, buffer: &wgpu::Buffer, level: f32) {
        let mut bytes = [0_u8; 16];
        bytes[..4].copy_from_slice(&level.to_le_bytes());
        self.queue.write_buffer(buffer, 0, &bytes);
    }

    /// Creates a pair of `ACCUM_FORMAT` textures of the given size, for use
    /// as a filter chain's ping-pong buffers.
    fn create_scratch_pair(
        &self,
        width: usize,
        height: usize,
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        #[expect(clippy::cast_possible_truncation)]
        let extent = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        };

        let make = || {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("exrviewer-filter-scratch"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ACCUM_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };

        (make(), make())
    }

    fn create_targets(&self, size: [usize; 2]) -> Targets {
        #[expect(clippy::cast_possible_truncation)]
        let extent = wgpu::Extent3d {
            width: size[0] as u32,
            height: size[1] as u32,
            depth_or_array_layers: 1,
        };

        let accum_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("exrviewer-accum-texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ACCUM_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let accum_view = accum_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let accum_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exrviewer-accum-bind-group"),
            layout: &self.texture_only_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&accum_view),
            }],
        });

        let (composite_scratch_a, composite_scratch_b) = self.create_scratch_pair(size[0], size[1]);

        let display_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("exrviewer-display-texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DISPLAY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let display_view = display_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Targets {
            size,
            accum_view,
            accum_bind_group,
            composite_scratch_a,
            composite_scratch_b,
            display_texture,
            display_view,
        }
    }

    fn filter_pipeline(&self, filter: &dyn crate::Filter) -> &wgpu::RenderPipeline {
        self.filter_pipelines
            .get(filter.label())
            .expect("filter instance's label doesn't match any registered FilterKind")
    }

    /// Runs one full-screen pass: binds `bind_group` (already wrapping
    /// whatever `pipeline`'s shader needs) at group 1, and renders into
    /// `dest`.
    fn run_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
        dest: &wgpu::TextureView,
        label: &str,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.sampler_bind_group, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Runs one filter stage: samples `source`, applies `pipeline` with the
    /// given `params`, and writes into `dest`.
    fn run_filter_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::TextureView,
        params: [f32; 4],
        dest: &wgpu::TextureView,
    ) {
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exrviewer-filter-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &params_bytes(params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exrviewer-filter-bind-group"),
            layout: &self.texture_uniform_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        self.run_pass(
            encoder,
            pipeline,
            &bind_group,
            dest,
            "exrviewer-filter-pass",
        );
    }

    /// Runs `filters` in order, each as one or more GPU passes (see
    /// `Filter::stage_count`), ping-ponging between `first_dest` and
    /// `second_dest`. Returns whichever view holds the final result:
    /// `source` itself if `filters` is empty. `first_dest`/`second_dest`
    /// must both differ from `source` (they don't need to differ from each
    /// other in any particular order, just from `source`, so the composite
    /// chain can pass its own post-clamp buffer as `source` and reuse it as
    /// `second_dest`).
    fn run_filter_chain<'v>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        filters: &[crate::FilterEntry],
        size: (u32, u32),
        source: &'v wgpu::TextureView,
        first_dest: &'v wgpu::TextureView,
        second_dest: &'v wgpu::TextureView,
    ) -> &'v wgpu::TextureView {
        let mut current = source;
        let mut use_first = true;

        for entry in filters {
            let filter = entry.filter.as_ref();
            let pipeline = self.filter_pipeline(filter);
            for stage in 0..filter.stage_count() {
                let dest = if use_first { first_dest } else { second_dest };
                let params = filter.stage_params(stage, size);
                self.run_filter_stage(encoder, pipeline, current, params, dest);
                current = dest;
                use_first = !use_first;
            }
        }

        current
    }

    /// Re-runs the composite for the currently loaded layers, using each
    /// layer's current `level` and filters, plus the composite's own
    /// filters. The result can be displayed via `display_view()` or read
    /// back via `read_display_rgba()`. Does nothing if nothing has been
    /// loaded yet.
    pub fn compose(&mut self, composition: &crate::Composition) {
        let Some(targets) = self.targets.as_ref() else {
            return;
        };
        if targets.size != composition.size {
            // The canvas size only changes when a new file is loaded, which
            // always goes through `load` first; this should not happen, but
            // guard against a stale/mismatched call rather than panicking.
            return;
        }

        #[expect(clippy::cast_possible_truncation)]
        let size = (targets.size[0] as u32, targets.size[1] as u32);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("exrviewer-compose-encoder"),
            });

        let layer_sources = self.filter_layers(&mut encoder, composition, size);
        self.accumulate(&mut encoder, targets, &layer_sources);
        self.resolve_composite(&mut encoder, targets, composition, size);

        self.queue.submit(Some(encoder.finish()));
    }

    /// Filters each layer (its own passes) before `accumulate` blends the
    /// result in - can't interleave the two, since a render pass
    /// exclusively borrows the encoder. Returns each layer's final source
    /// view (its raw texture, if it has no filters).
    fn filter_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        composition: &crate::Composition,
        size: (u32, u32),
    ) -> Vec<&wgpu::TextureView> {
        self.layers
            .iter()
            .zip(composition.layers.iter())
            .map(|(gpu_layer, layer)| {
                self.write_level(&gpu_layer.level_buffer, layer.level);
                if layer.filters.is_empty() {
                    &gpu_layer.raw_view
                } else {
                    self.run_filter_chain(
                        encoder,
                        &layer.filters,
                        size,
                        &gpu_layer.raw_view,
                        &gpu_layer.scratch_a,
                        &gpu_layer.scratch_b,
                    )
                }
            })
            .collect()
    }

    /// Blends each layer's (already filtered) source into `targets.accum_view`.
    fn accumulate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &Targets,
        layer_sources: &[&wgpu::TextureView],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("exrviewer-accumulate-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.accum_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.accumulate_pipeline);
        pass.set_bind_group(0, &self.sampler_bind_group, &[]);
        for (gpu_layer, source) in self.layers.iter().zip(layer_sources.iter()) {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("exrviewer-layer-bind-group"),
                layout: &self.texture_uniform_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_layer.level_buffer.as_entire_binding(),
                    },
                ],
            });
            pass.set_bind_group(1, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Clamps the accumulated HDR result and writes the final display
    /// texture, running the composite's own filter chain in between if it
    /// has one.
    fn resolve_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &Targets,
        composition: &crate::Composition,
        size: (u32, u32),
    ) {
        if composition.filters.is_empty() {
            self.run_pass(
                encoder,
                &self.resolve_pipeline,
                &targets.accum_bind_group,
                &targets.display_view,
                "exrviewer-resolve-pass",
            );
            return;
        }

        self.run_pass(
            encoder,
            &self.clamp_pipeline,
            &targets.accum_bind_group,
            &targets.composite_scratch_a,
            "exrviewer-clamp-pass",
        );
        let final_view = self.run_filter_chain(
            encoder,
            &composition.filters,
            size,
            &targets.composite_scratch_a,
            &targets.composite_scratch_b,
            &targets.composite_scratch_a,
        );
        let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exrviewer-blit-bind-group"),
            layout: &self.texture_only_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(final_view),
            }],
        });
        self.run_pass(
            encoder,
            &self.blit_pipeline,
            &blit_bind_group,
            &targets.display_view,
            "exrviewer-blit-pass",
        );
    }

    /// Reads the current composite back to the CPU as interleaved RGBA8
    /// bytes, row-major. Blocks until the GPU work is done. Meant for
    /// headless callers (the GUI displays `display_view()` directly instead,
    /// since it never needs the pixels on the CPU).
    pub fn read_display_rgba(&self) -> Option<Vec<u8>> {
        let targets = self.targets.as_ref()?;
        #[expect(clippy::cast_possible_truncation)]
        let width = targets.size[0] as u32;
        #[expect(clippy::cast_possible_truncation)]
        let height = targets.size[1] as u32;

        let unpadded_bytes_per_row = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exrviewer-readback-buffer"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("exrviewer-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets.display_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            drop(tx.send(result));
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        rx.recv().ok()?.ok()?;

        let mapped = slice.get_mapped_range().ok()?;
        let mut rgba = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
        for row in 0..height as usize {
            let start = row * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            rgba.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        buffer.unmap();

        Some(rgba)
    }
}

/// Creates the sampler shared by every pass. Layers are composited at their
/// native resolution with no resampling, so nearest filtering is not just
/// enough but correct; it also sidesteps needing "filterable float" backend
/// support.
fn create_sampler(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("exrviewer-compose-sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("exrviewer-compose-sampler-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("exrviewer-compose-sampler-bind-group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Sampler(&sampler),
        }],
    });

    (layout, bind_group)
}

/// Creates the two group-1 layout shapes every pass uses: `(texture,
/// uniform)` for the accumulate pass and every filter, `(texture)` alone for
/// resolve/clamp/blit.
fn create_shared_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let texture_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exrviewer-texture-uniform-layout"),
            entries: &[texture_entry(0), uniform_entry(1)],
        });
    let texture_only_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("exrviewer-texture-only-layout"),
        entries: &[texture_entry(0)],
    });

    (texture_uniform_layout, texture_only_layout)
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader_src: &str,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_group_layouts: Vec<_> = bind_group_layouts.iter().copied().map(Some).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bind_group_layouts,
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
