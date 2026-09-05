#![warn(clippy::all, rust_2018_idioms)]

mod app;
mod composition;

pub use app::LayerCompositorApp;
pub use composition::{Composition, CompositionLayer};
