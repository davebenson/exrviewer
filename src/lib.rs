#![warn(clippy::all, rust_2018_idioms)]

mod app;
mod composition;
mod filters;
mod gpu_compose;

pub use app::LayerCompositorApp;
pub use composition::{Composition, CompositionLayer};
pub use filters::{ALL_KINDS, Filter, FilterEntry, FilterKind};
pub use gpu_compose::GpuCompositor;
