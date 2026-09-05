//! A separable box blur: `radius` is the kernel radius in pixels (clamped
//! to `MAX_RADIUS`, both to bound cost and because the GPU shader's loop
//! shouldn't run unboundedly). This is the one filter that genuinely needs
//! more than a single pixel's own color, which is why it needs two passes
//! (horizontal, then vertical) on the GPU rather than one.
//!
//! Both the CPU and GPU implementations sample with the same "clamp to
//! edge" boundary behavior (repeating the edge pixel for taps that fall
//! outside the image, always dividing by the full tap count) so they
//! produce the same result.

use super::{Filter, FilterKind};

const SHADER: &str = include_str!("blur.wgsl");
const LABEL: &str = "Blur";
const DEFAULT_RADIUS: f32 = 4.0;
const MAX_RADIUS: f32 = 32.0;

pub(crate) struct Blur;

impl FilterKind for Blur {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn shader(&self) -> &'static str {
        SHADER
    }

    fn create(&self) -> Box<dyn Filter> {
        Box::new(BlurFilter {
            radius: DEFAULT_RADIUS,
        })
    }
}

struct BlurFilter {
    radius: f32,
}

impl Filter for BlurFilter {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn stage_count(&self) -> usize {
        2
    }

    /// Stage 0 blurs horizontally, stage 1 vertically.
    fn stage_params(&self, stage: usize, size: (u32, u32)) -> [f32; 4] {
        let radius = self.radius.clamp(0.0, MAX_RADIUS).round();
        let (width, height) = size;
        let (step_x, step_y) = if stage == 0 {
            (1.0 / width.max(1) as f32, 0.0)
        } else {
            (0.0, 1.0 / height.max(1) as f32)
        };
        [radius, step_x, step_y, 0.0]
    }

    fn apply_cpu(&self, width: usize, height: usize, rgb: &[[f32; 3]]) -> Vec<[f32; 3]> {
        #[expect(clippy::cast_possible_truncation)]
        let radius = self.radius.clamp(0.0, MAX_RADIUS).round() as usize;
        if radius == 0 || width == 0 || height == 0 {
            return rgb.to_vec();
        }

        let horizontal = box_blur_pass(rgb, width, height, radius, true);
        box_blur_pass(&horizontal, width, height, radius, false)
    }

    fn make_ui(&mut self, ui: &mut egui::Ui) -> bool {
        ui.add(egui::Slider::new(&mut self.radius, 0.0..=MAX_RADIUS).text("Radius"))
            .changed()
    }
}

fn box_blur_pass(
    src: &[[f32; 3]],
    width: usize,
    height: usize,
    radius: usize,
    horizontal: bool,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0_f32; 3]; src.len()];
    #[expect(clippy::cast_precision_loss)]
    let taps = (2 * radius + 1) as f32;
    let radius = radius.cast_signed();
    let (width_i, height_i) = (width.cast_signed(), height.cast_signed());

    for y in 0..height {
        for x in 0..width {
            let (x_i, y_i) = (x.cast_signed(), y.cast_signed());
            let mut sum = [0.0_f32; 3];
            for offset in -radius..=radius {
                let (sx, sy) = if horizontal {
                    ((x_i + offset).clamp(0, width_i - 1), y_i)
                } else {
                    (x_i, (y_i + offset).clamp(0, height_i - 1))
                };
                let [r, g, b] = src[sy.cast_unsigned() * width + sx.cast_unsigned()];
                sum[0] += r;
                sum[1] += g;
                sum[2] += b;
            }
            out[y * width + x] = [sum[0] / taps, sum[1] / taps, sum[2] / taps];
        }
    }

    out
}
