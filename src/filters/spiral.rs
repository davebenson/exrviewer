//! A separable box spiral: `radius` is the kernel radius in pixels (clamped
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

const SHADER: &str = include_str!("spiral.wgsl");
const LABEL: &str = "Spiral";
const DEFAULT_ANGLE_SCALE: f32 = 1.0;
const MAX_ANGLE_SCALE: f32 = 32.0;
const DEFAULT_TWIST: f32 = 1.0;
const MAX_TWIST: f32 = 4.0;

pub(crate) struct Spiral;

impl FilterKind for Spiral {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn shader(&self) -> &'static str {
        SHADER
    }

    fn create(&self) -> Box<dyn Filter> {
        Box::new(SpiralFilter {
            center_x: 0.5,
            center_y: 0.5,
            twist: DEFAULT_TWIST,
            angle_scale: DEFAULT_ANGLE_SCALE,
        })
    }
}

struct SpiralFilter {
    center_x: f32,
    center_y: f32,
    twist: f32,
    angle_scale: f32,
}

impl Filter for SpiralFilter {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn stage_count(&self) -> usize {
        1
    }

    fn stage_params(&self, _stage: usize, _size: (u32, u32)) -> [f32; 4] {
        [self.center_x, self.center_y, self.twist, self.angle_scale]
    }

    fn apply_cpu(&self, width: usize, height: usize, rgb: &[[f32; 3]]) -> Vec<[f32; 3]> {
        spiral_pass(
            &rgb,
            width,
            height,
            self.center_x,
            self.center_y,
            self.twist,
            self.angle_scale,
        )
    }

    fn make_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let widgets = [
            egui::Slider::new(&mut self.center_x, 0.0..=1.0).text("X"),
            egui::Slider::new(&mut self.center_y, 0.0..=1.0).text("Y"),
            egui::Slider::new(&mut self.twist, 0.0..=MAX_TWIST).text("Twist"),
            egui::Slider::new(&mut self.angle_scale, 0.0..=MAX_ANGLE_SCALE).text("Angle Scale"),
        ];
        let mut changed = false;
        ui.vertical(|ui| {
            for widget in widgets {
                if ui.add(widget).changed() {
                    changed = true;
                }
            }
        });
        changed
    }
}

fn spiral_pass(
    src: &[[f32; 3]],
    width: usize,
    height: usize,
    center_x: f32,
    center_y: f32,
    twist: f32,
    angle_scale: f32,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0_f32; 3]; src.len()];
    let (width_i, height_i) = (width.cast_signed(), height.cast_signed());

    let w2 = (width as f32) * center_x;
    let h2 = (height as f32) * center_y;
    let maxr = (width.max(height) as f32) * 0.5;
    for y in 0..height {
        for x in 0..width {
            let xx = (x as f32) - w2;
            let yy = (y as f32) - h2;
            let r = xx.hypot(yy);
            let orig_theta = yy.atan2(xx);
            let theta = orig_theta + (r / maxr).powf(twist) * angle_scale;
            #[allow(clippy::cast_possible_truncation)]
            let newx = (theta.cos() * r + w2) as isize;
            #[allow(clippy::cast_possible_truncation)]
            let newy = (theta.sin() * r + h2) as isize;
            out[y * width + x] = if newx < 0 || newy < 0 || newx >= width_i || newy >= height_i {
                [0.0, 0.0, 0.0]
            } else {
                src[(newx as usize) + (newy as usize) * width]
            };
        }
    }

    out
}
