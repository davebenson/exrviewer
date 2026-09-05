//! Color-adjustment filters that can be stacked onto a layer or onto the
//! final composite.
//!
//! [`FilterKind`] is a lightweight, stateless descriptor for a *type* of
//! filter: its label, its shader, and how to create a new instance.
//! [`Filter`] is an actual instance, holding whatever parameters that kind
//! needs - which can be more than just one number, e.g. `blur`'s radius
//! versus a hypothetical levels filter's black/white points. Each kind
//! lives in its own file (`brightness.rs`, `invert.rs`, `blur.rs`), which
//! owns both the `FilterKind` and `Filter` impls for that kind, its WGSL
//! shader source, and its CPU whole-image implementation.
//!
//! Filters run on the *whole* image, not pixel-by-pixel: `blur` needs its
//! neighbors' colors, so even the per-pixel filters (`brightness`,
//! `invert`) are shaped as whole-image passes for uniformity.
//! `gpu_compose` is the only place that actually touches `wgpu` types; this
//! module hands it shader source and plain `[f32; 4]` per-stage parameters.

mod blur;
mod brightness;
mod bw;
mod fisheye;
mod invert;
mod spiral;

/// A stateless descriptor for a *type* of filter: enough to list it in an
/// "add filter" menu (`label`), build its GPU pipeline once (`shader`), and
/// create new instances of it (`create`).
pub trait FilterKind {
    fn label(&self) -> &'static str;

    /// This kind's WGSL shader source. Every instance of a kind shares one
    /// GPU pipeline built from this; see `gpu_compose`.
    fn shader(&self) -> &'static str;

    /// Creates a new instance with default parameters.
    fn create(&self) -> Box<dyn Filter>;
}

/// One filter instance: a specific kind, holding whatever parameters that
/// kind needs. `make_ui` builds the controls for editing them.
pub trait Filter {
    /// Must match the [`FilterKind::label`] of whatever kind created this
    /// instance; used both for UI display and to look up this instance's
    /// GPU pipeline (see `gpu_compose`).
    fn label(&self) -> &'static str;

    /// How many GPU passes this filter needs (`blur` is separable: one
    /// horizontal pass, one vertical). All stages share one pipeline;
    /// `stage_params` is what tells them apart.
    fn stage_count(&self) -> usize;

    /// The `(x, y, z, w)` uniform passed to this filter's shader for the
    /// given stage. `size` is the working image's `(width, height)` in
    /// texels, needed by filters (like `blur`) whose per-tap UV step
    /// depends on resolution.
    fn stage_params(&self, stage: usize, size: (u32, u32)) -> [f32; 4];

    /// Applies this filter to a whole image's RGB, on the CPU. Used by the
    /// CLI's fallback path when no GPU is available; the GPU path is in
    /// `gpu_compose`.
    fn apply_cpu(&self, width: usize, height: usize, rgb: &[[f32; 3]]) -> Vec<[f32; 3]>;

    /// Builds this filter's parameter-editing UI (e.g. a slider). Returns
    /// whether anything changed, so the caller knows to recomposite.
    fn make_ui(&mut self, ui: &mut egui::Ui) -> bool;
}

/// Every known filter kind, e.g. for populating an "add filter" menu.
pub const ALL_KINDS: &[&dyn FilterKind] = &[
    &brightness::Brightness,
    &invert::Invert,
    &blur::Blur,
    &bw::BW,
    &fisheye::Fisheye,
    &spiral::Spiral,
];

/// A filter instance plus UI-only "expanded" state.
///
/// `expanded` controls whether its editing controls (built by
/// [`Filter::make_ui`]) are shown; kept separate from `Filter` itself so the
/// trait doesn't need to carry UI-only state.
pub struct FilterEntry {
    pub filter: Box<dyn Filter>,
    pub expanded: bool,
}

impl FilterEntry {
    pub fn new(filter: Box<dyn Filter>) -> Self {
        Self {
            filter,
            expanded: false,
        }
    }
}

/// Applies `filters` in order to a whole image's RGB, on the CPU.
pub fn apply_all_cpu(
    filters: &[FilterEntry],
    width: usize,
    height: usize,
    rgb: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    let mut current: Option<Vec<[f32; 3]>> = None;
    for entry in filters {
        let source = current.as_deref().unwrap_or(rgb);
        current = Some(entry.filter.apply_cpu(width, height, source));
    }
    current.unwrap_or_else(|| rgb.to_vec())
}
