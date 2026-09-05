//! GPU-accelerated implementation of [`crate::Composition::compose`].
//!
//! Each layer's pixels are uploaded to the GPU once, when the file is
//! loaded. From then on, recompositing is just a couple of render passes
//! that draw one full-screen quad per layer, so it stays fast regardless of
//! how expensive the equivalent CPU loop would be.
//!
//! This module only needs a `wgpu::Device`/`Queue`: it doesn't care whether
//! those came from a GUI (`egui_wgpu::RenderState`, which lets the result be
//! displayed directly as a GPU texture via `display_view()`) or a headless
//! setup (a CLI tool, which reads the result back to the CPU via
//! `read_display_rgba()` to save it to a file).
//!
//! ## How it maps to the CPU formula
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
//! path applies before quantizing to bytes) happens in the resolve pass,
//! since a floating-point render target doesn't clamp on write the way a
//! `Unorm` target does.

const ACCUMULATE_SHADER: &str = include_str!("shaders/accumulate.wgsl");
const RESOLVE_SHADER: &str = include_str!("shaders/resolve.wgsl");

/// The format used for the accumulation target. `Rgba16Float` (unlike
/// `Rgba32Float`) is guaranteed renderable and blendable on all wgpu
/// backends without opt-in features.
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The format of the final, displayable composite. This matches what
/// `Composition::compose` produces on the CPU (linear bytes, no sRGB
/// encoding), so the two paths look the same.
const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One layer's pixels, resident on the GPU.
struct GpuLayer {
    bind_group: wgpu::BindGroup,
    level_buffer: wgpu::Buffer,
}

/// A composition's GPU-side render targets, rebuilt whenever the canvas size
/// changes (i.e. whenever a new file is loaded).
struct Targets {
    size: [usize; 2],
    accum_view: wgpu::TextureView,
    display_texture: wgpu::Texture,
    display_view: wgpu::TextureView,
    accum_bind_group: wgpu::BindGroup,
}

/// Owns the GPU pipelines used to composite layers, plus the per-composition
/// resources (layer textures, render targets) for whichever composition was
/// last uploaded.
pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    sampler_bind_group: wgpu::BindGroup,
    layer_bind_group_layout: wgpu::BindGroupLayout,
    accum_bind_group_layout: wgpu::BindGroupLayout,
    accumulate_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,
    layers: Vec<GpuLayer>,
    targets: Option<Targets>,
}

impl GpuCompositor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Layers are composited at their native resolution with no
        // resampling, so nearest filtering is not just enough but correct.
        // It also sidesteps needing "filterable float" backend support.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("exrviewer-compose-sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let sampler_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("exrviewer-compose-sampler-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                }],
            });

        let sampler_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exrviewer-compose-sampler-bind-group"),
            layout: &sampler_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            }],
        });

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

        let layer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("exrviewer-compose-layer-layout"),
                entries: &[
                    texture_entry(0),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let accum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("exrviewer-compose-accum-layout"),
                entries: &[texture_entry(0)],
            });

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
            &[&sampler_bind_group_layout, &layer_bind_group_layout],
            ACCUM_FORMAT,
            Some(additive_blend),
        );

        let resolve_pipeline = create_pipeline(
            device,
            "resolve",
            RESOLVE_SHADER,
            &[&sampler_bind_group_layout, &accum_bind_group_layout],
            DISPLAY_FORMAT,
            None,
        );

        Self {
            device: device.clone(),
            queue: queue.clone(),
            sampler_bind_group,
            layer_bind_group_layout,
            accum_bind_group_layout,
            accumulate_pipeline,
            resolve_pipeline,
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

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let level_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exrviewer-layer-level"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.write_level(&level_buffer, layer.level);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exrviewer-layer-bind-group"),
            layout: &self.layer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: level_buffer.as_entire_binding(),
                },
            ],
        });

        GpuLayer {
            bind_group,
            level_buffer,
        }
    }

    fn write_level(&self, buffer: &wgpu::Buffer, level: f32) {
        let mut bytes = [0_u8; 16];
        bytes[..4].copy_from_slice(&level.to_le_bytes());
        self.queue.write_buffer(buffer, 0, &bytes);
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
            layout: &self.accum_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&accum_view),
            }],
        });

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
            display_texture,
            display_view,
            accum_bind_group,
        }
    }

    /// Re-runs the composite for the currently loaded layers, using each
    /// layer's current `level`. The result can be displayed via
    /// `display_view()` or read back via `read_display_rgba()`. Does
    /// nothing if nothing has been loaded yet.
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

        for (gpu_layer, layer) in self.layers.iter().zip(composition.layers.iter()) {
            self.write_level(&gpu_layer.level_buffer, layer.level);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("exrviewer-compose-encoder"),
            });

        {
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
            for gpu_layer in &self.layers {
                pass.set_bind_group(1, &gpu_layer.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("exrviewer-resolve-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.display_view,
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
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &self.sampler_bind_group, &[]);
            pass.set_bind_group(1, &targets.accum_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
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
