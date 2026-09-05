use super::{Filter, FilterKind};

const SHADER: &str = include_str!("fisheye.wgsl");
const LABEL: &str = "Fisheye";
const DEFAULT_POWER: f32 = 1.0;
const MAX_POWER: f32 = 4.0;
const MIN_POWER: f32 = 0.1;

pub(crate) struct Fisheye;

impl FilterKind for Fisheye {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn shader(&self) -> &'static str {
        SHADER
    }

    fn create(&self) -> Box<dyn Filter> {
        Box::new(FisheyeFilter {
            center_x: 0.5,
            center_y: 0.5,
            power: DEFAULT_POWER,
        })
    }
}

struct FisheyeFilter {
    center_x: f32,
    center_y: f32,
    power: f32,
}

impl Filter for FisheyeFilter {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn stage_count(&self) -> usize {
        1
    }

    fn stage_params(&self, _stage: usize, _size: (u32, u32)) -> [f32; 4] {
        [self.center_x, self.center_y, self.power, 0.0]
    }

    fn apply_cpu(&self, width: usize, height: usize, rgb: &[[f32; 3]]) -> Vec<[f32; 3]> {
        fisheye_pass(rgb, width, height, self.center_x, self.center_y, self.power)
    }

    fn make_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let widgets = [
            egui::Slider::new(&mut self.center_x, 0.0..=1.0).text("X"),
            egui::Slider::new(&mut self.center_y, 0.0..=1.0).text("Y"),
            egui::Slider::new(&mut self.power, MIN_POWER..=MAX_POWER).text("Power"),
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

fn fisheye_pass(
    src: &[[f32; 3]],
    width: usize,
    height: usize,
    center_x: f32,
    center_y: f32,
    power: f32,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0_f32; 3]; src.len()];
    let (width_i, height_i) = (width.cast_signed(), height.cast_signed());

    let cx = center_x * (width as f32);
    let cy = center_y * (height as f32);
    let w2 = (width as f32) * center_x;
    let h2 = (height as f32) * center_y;
    let maxr = (width.max(height) as f32) * 0.5;
    for y in 0..height {
        for x in 0..width {
            let xx = (x as f32) - cx;
            let yy = (y as f32) - cy;
            let r = xx.hypot(yy) / maxr;
            let scale = if r < 1e-5 { 0.0 } else { r.powf(power - 1.0) };
            #[allow(clippy::cast_possible_truncation)]
            let newx = (xx * scale + w2) as isize;
            #[allow(clippy::cast_possible_truncation)]
            let newy = (xx * scale + h2) as isize;
            out[y * width + x] = if newx < 0 || newy < 0 || newx >= width_i || newy >= height_i {
                [0.0, 0.0, 0.0]
            } else {
                src[(newx as usize) + (newy as usize) * width]
            };
        }
    }

    out
}
