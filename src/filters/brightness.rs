//! Multiplies RGB by a constant factor. Trivially per-pixel, but shaped like
//! every other filter (a single GPU pass, a CPU whole-image function) so it
//! composes uniformly with filters that aren't, e.g. `blur`.

use super::{Filter, FilterKind};

const SHADER: &str = include_str!("brightness.wgsl");
const LABEL: &str = "Brightness";
const DEFAULT_AMOUNT: f32 = 1.2;

pub(crate) struct Brightness;

impl FilterKind for Brightness {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn shader(&self) -> &'static str {
        SHADER
    }

    fn create(&self) -> Box<dyn Filter> {
        Box::new(BrightnessFilter {
            amount: DEFAULT_AMOUNT,
        })
    }
}

struct BrightnessFilter {
    amount: f32,
}

impl Filter for BrightnessFilter {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn stage_count(&self) -> usize {
        1
    }

    fn stage_params(&self, _stage: usize, _size: (u32, u32)) -> [f32; 4] {
        [self.amount, 0.0, 0.0, 0.0]
    }

    fn apply_cpu(&self, _width: usize, _height: usize, rgb: &[[f32; 3]]) -> Vec<[f32; 3]> {
        rgb.iter()
            .map(|&[r, g, b]| [r * self.amount, g * self.amount, b * self.amount])
            .collect()
    }

    fn make_ui(&mut self, ui: &mut egui::Ui) -> bool {
        ui.add(egui::Slider::new(&mut self.amount, 0.0..=3.0).text("Amount"))
            .changed()
    }
}
