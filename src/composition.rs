use std::path::Path;
use std::sync::Arc;

use exr::prelude::*;

use crate::filters::{self, Filter};

/// A single, decoded layer of an EXR file, holding its RGBA pixels as `f32`
/// samples in `0.0..=1.0`-ish linear range (whatever the file itself stored).
#[derive(Clone)]
pub struct CompositionLayer {
    pub name: String,
    pub size: [usize; 2],
    /// Row-major RGBA pixels, one `[r, g, b, a]` per pixel.
    ///
    /// Shared via `Arc` (rather than `Vec`) so that cloning a `Composition`
    /// to hand off to a background compositing thread is cheap.
    pub pixels: Arc<[[f32; 4]]>,
    pub level: f32,
    /// Applied to this layer's RGB, in order, before it's blended into the
    /// composite.
    pub filters: Vec<Filter>,
}

impl CompositionLayer {
    /// Extracts one pseudo-layer per distinct channel-name prefix found in a
    /// raw EXR layer.
    ///
    /// Some tools (e.g. Blender's "`OpenEXR` `MultiLayer`" format) don't write
    /// separate EXR layers/parts at all: they write a single layer whose
    /// channels are named `"<layer>.<channel>"`, e.g. `"ViewLayer.Combined_ryan.R"`.
    /// We group channels by everything before the last `.` to recover the
    /// layers such files intend. Files that use plain, unprefixed channel
    /// names (`"R"`, `"G"`, ...) are grouped together under the raw layer's
    /// own name instead.
    fn extract(index: usize, layer: &Layer<AnyChannels<FlatSamples>>) -> Vec<Self> {
        let fallback_name = layer
            .attributes
            .layer_name
            .as_ref()
            .map_or_else(|| format!("layer {index}"), ToString::to_string);

        let mut order: Vec<String> = Vec::new();
        let mut groups: std::collections::HashMap<String, Vec<[f32; 4]>> =
            std::collections::HashMap::new();
        let pixel_count = layer.size.area();

        for channel in &layer.channel_data.list {
            let full_name = channel.name.to_string();
            let (group_name, component_name) = match full_name.rsplit_once('.') {
                Some((prefix, component)) => (prefix.to_owned(), component),
                None => (fallback_name.clone(), full_name.as_str()),
            };

            let component = match component_name {
                "R" => 0,
                "G" => 1,
                "B" => 2,
                "A" => 3,
                _ => continue,
            };

            let pixels = groups.entry(group_name.clone()).or_insert_with(|| {
                order.push(group_name.clone());
                // Channels without alpha are treated as fully opaque.
                vec![[0.0, 0.0, 0.0, 1.0]; pixel_count]
            });

            for (pixel, value) in pixels.iter_mut().zip(channel.sample_data.values_as_f32()) {
                pixel[component] = value;
            }
        }

        order
            .into_iter()
            .map(|name| {
                #[expect(clippy::unwrap_used)]
                let pixels = groups.remove(&name).unwrap();

                Self {
                    name,
                    size: [layer.size.0, layer.size.1],
                    pixels: Arc::from(pixels),
                    level: 1.0,
                    filters: Vec::new(),
                }
            })
            .collect()
    }
}

/// Layers that some tools always emit alongside the layers a user actually
/// wants to composite (e.g. Blender's aggregate "Combined" beauty pass and
/// its noisy pre-denoise render), and which we never want to show or
/// composite.
const EXCLUDED_LAYER_NAMES: [&str; 3] = [
    "Composite.Combined",
    "ViewLayer.Combined",
    "ViewLayer.Noisy Image",
];

/// A stack of decoded EXR layers, ready to be flattened into a single image.
#[derive(Clone)]
pub struct Composition {
    pub size: [usize; 2],
    /// Layers in bottom-to-top compositing order, as they appear in the file.
    pub layers: Vec<CompositionLayer>,
    /// Applied to the final composite's RGB, in order, after all layers are
    /// blended together.
    pub filters: Vec<Filter>,
}

impl Composition {
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or does not contain a
    /// valid, non-deep EXR image.
    pub fn load_exr(path: impl AsRef<Path>) -> exr::error::Result<Self> {
        let image = read()
            .no_deep_data()
            .largest_resolution_level()
            .all_channels()
            .all_layers()
            .all_attributes()
            .from_file(path)?;

        let layers: Vec<CompositionLayer> = image
            .layer_data
            .iter()
            .enumerate()
            .flat_map(|(index, layer)| CompositionLayer::extract(index, layer))
            .filter(|layer| !EXCLUDED_LAYER_NAMES.contains(&layer.name.as_str()))
            .collect();

        let size = layers.first().map_or([0, 0], |layer| layer.size);

        Ok(Self {
            size,
            layers,
            filters: Vec::new(),
        })
    }

    /// Flattens all layers, bottom to top, using standard "over" alpha
    /// blending, and returns the result as interleaved RGBA8 pixels,
    /// row-major.
    pub fn compose(&self) -> Vec<u8> {
        let [width, height] = self.size;
        let area = width * height;
        let mut result = vec![[0.0_f32; 4]; area];

        for layer in &self.layers {
            // Layers positioned or sized differently from the canvas are not
            // yet supported.
            if layer.size != self.size {
                continue;
            }

            // Filters need the whole layer's RGB up front: several (e.g.
            // blur) look at neighboring pixels, not just their own.
            let rgb: Vec<[f32; 3]> = layer.pixels.iter().map(|&[r, g, b, _]| [r, g, b]).collect();
            let filtered = filters::apply_all_cpu(&layer.filters, width, height, &rgb);

            for ((out, &[.., a]), &[r, g, b]) in result
                .iter_mut()
                .zip(layer.pixels.iter())
                .zip(filtered.iter())
            {
                let alpha = a * layer.level;
                for (out_channel, in_channel) in out.iter_mut().take(3).zip([r, g, b]) {
                    *out_channel = in_channel.mul_add(alpha, *out_channel);
                }
                out[3] = (alpha + out[3]).min(1.0);
            }
        }

        // Clamp before applying composite filters, matching the GPU path
        // (which can't let HDR values feed a blur before this point either).
        let clamped: Vec<[f32; 3]> = result
            .iter()
            .map(|&[r, g, b, _]| [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)])
            .collect();
        let filtered = filters::apply_all_cpu(&self.filters, width, height, &clamped);

        result
            .iter()
            .zip(filtered.iter())
            .flat_map(|(&[_, _, _, a], &[r, g, b])| {
                [r, g, b, a].map(|value| {
                    // `round()` of a value clamped to `0.0..=1.0` always fits in a `u8`.
                    #[expect(clippy::cast_possible_truncation)]
                    let byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    byte
                })
            })
            .collect()
    }
}
