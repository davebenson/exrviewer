//! Mixes RGB with its inverse. Trivially per-pixel; see `brightness` for why
//! it's still shaped like a full filter pass.

use super::{Filter, FilterKind};

const SHADER: &str = include_str!("invert.wgsl");
const LABEL: &str = "Invert";
const DEFAULT_AMOUNT: f32 = 1.0;

pub(crate) struct Invert;

impl FilterKind for Invert {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn shader(&self) -> &'static str {
        SHADER
    }

    fn create(&self) -> Box<dyn Filter> {
        Box::new(InvertFilter {
            amount: DEFAULT_AMOUNT,
        })
    }
}

/// `amount` is the mix factor: 0 = no change, 1 = fully inverted.
struct InvertFilter {
    amount: f32,
}

impl Filter for InvertFilter {
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
            .map(|&[r, g, b]| {
                [r, g, b].map(|c| c.mul_add(1.0 - self.amount, (1.0 - c) * self.amount))
            })
            .collect()
    }

    fn make_ui(&mut self, ui: &mut egui::Ui) -> bool {
        ui.add(egui::Slider::new(&mut self.amount, 0.0..=1.0).text("Amount"))
            .changed()
    }
}
