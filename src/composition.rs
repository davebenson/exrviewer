use std::path::Path;
use std::sync::Arc;

use exr::prelude::*;

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

        Ok(Self { size, layers })
    }

    /// Flattens all layers, bottom to top, using standard "over" alpha
    /// blending, and returns the result as interleaved RGBA8 pixels,
    /// row-major.
    pub fn compose(&self) -> Vec<u8> {
        let area = self.size[0] * self.size[1];
        let mut result = vec![[0.0_f32; 4]; area];

        for layer in &self.layers {
            // Layers positioned or sized differently from the canvas are not
            // yet supported.
            if layer.size != self.size {
                continue;
            }

            for (out, &[r, g, b, a]) in result.iter_mut().zip(layer.pixels.iter()) {
                let alpha = a * layer.level;
                for (out_channel, in_channel) in out.iter_mut().take(3).zip([r, g, b]) {
                    *out_channel = in_channel.mul_add(alpha, *out_channel);
                }
                out[3] = (alpha + out[3]).min(1.0);
            }
        }

        result
            .into_iter()
            .flat_map(|pixel| {
                pixel.map(|value| {
                    // `round()` of a value clamped to `0.0..=1.0` always fits in a `u8`.
                    #[expect(clippy::cast_possible_truncation)]
                    let byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    byte
                })
            })
            .collect()
    }
}
