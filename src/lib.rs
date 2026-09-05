#![warn(clippy::all, rust_2018_idioms)]

mod app;
mod composition;
mod gpu_compose;

pub use app::LayerCompositorApp;
pub use composition::{Composition, CompositionLayer};
pub use gpu_compose::GpuCompositor;
