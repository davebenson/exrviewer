/// We derive Deserialize/Serialize so we can persist app state on shutdown.
use egui_file_dialog::FileDialog;
use std::path::Path;

use crate::gpu_compose::GpuCompositor;
use crate::{Composition, FilterEntry};

/// How tall one "line" of the filter-list cell is: either a run of
/// collapsed filter chips (+ the "+" menu, if it's the last run), or one
/// expanded filter's own controls.
const FILTER_ROW_LINE_HEIGHT: f32 = 30.0;

enum FilterHorizontalRun {
    Unexpanded {
        first_filter: usize,
        num_filters: usize,
        show_plus: bool,
    },
    Expanded {
        index: usize,
    },
}

/// Groups `filters` into UI rows: consecutive collapsed filters share one
/// row (rendered as a horizontal strip of chips), each expanded filter gets
/// its own row, and the trailing "+" menu attaches to the last collapsed
/// run (or gets a row of its own if the list is empty or ends expanded).
fn layout_filters(filters: &[FilterEntry]) -> Vec<FilterHorizontalRun> {
    let mut runs = Vec::<FilterHorizontalRun>::new();
    for (index, entry) in filters.iter().enumerate() {
        if entry.expanded {
            runs.push(FilterHorizontalRun::Expanded { index });
        } else {
            match runs.last_mut() {
                Some(FilterHorizontalRun::Unexpanded { num_filters, .. }) => *num_filters += 1,
                _ => runs.push(FilterHorizontalRun::Unexpanded {
                    first_filter: index,
                    num_filters: 1,
                    show_plus: false,
                }),
            }
        }
    }

    match runs.last_mut() {
        Some(FilterHorizontalRun::Unexpanded { show_plus, .. }) => *show_plus = true,
        _ => runs.push(FilterHorizontalRun::Unexpanded {
            first_filter: 0,
            num_filters: 0,
            show_plus: true,
        }),
    }

    runs
}

/// The height needed to show `filters`' whole UI: one
/// [`FILTER_ROW_LINE_HEIGHT`] per row `layout_filters` would produce.
fn filter_row_height(filters: &[FilterEntry]) -> f32 {
    #[expect(clippy::cast_precision_loss)]
    let lines = layout_filters(filters).len() as f32;
    lines * FILTER_ROW_LINE_HEIGHT
}

/// Shows each filter as a button (click to expand/collapse its parameter
/// controls, built by `Filter::make_ui`) plus an "x" to remove it, and, at
/// the end, a "+" menu button to append a new filter of a chosen kind.
fn filters_cell_ui(ui: &mut egui::Ui, filters: &mut Vec<FilterEntry>, dirty: &mut bool) {
    let mut remove = None;

    ui.vertical(|ui| {
        for run in layout_filters(filters) {
            match run {
                FilterHorizontalRun::Unexpanded {
                    first_filter,
                    num_filters,
                    show_plus,
                } => {
                    ui.horizontal(|ui| {
                        for entry in filters.iter_mut().skip(first_filter).take(num_filters) {
                            if ui.button(entry.filter.label()).clicked() {
                                entry.expanded = true;
                            }
                        }
                        if show_plus {
                            ui.menu_button("+", |ui| {
                                for kind in crate::ALL_KINDS {
                                    if ui.button(kind.label()).clicked() {
                                        filters.push(FilterEntry::new(kind.create()));
                                        *dirty = true;
                                        ui.close();
                                    }
                                }
                            });
                        }
                    });
                }
                FilterHorizontalRun::Expanded { index } => {
                    ui.horizontal(|ui| {
                        let entry = &mut filters[index];
                        if ui.button(entry.filter.label()).clicked() {
                            entry.expanded = false;
                        }
                        if entry.filter.make_ui(ui) {
                            *dirty = true;
                        }
                        if ui.small_button("x").clicked() {
                            remove = Some(index);
                        }
                    });
                }
            }
        }
    });

    if let Some(index) = remove {
        filters.remove(index);
        *dirty = true;
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct LayerCompositorApp {
    // Example stuff:
    label: String,

    #[serde(skip)] // This how you opt-out of serialization of a field
    value: f32,

    #[serde(skip)]
    file_dialog: FileDialog,

    // Not yet displayed in the UI.
    #[serde(skip)]
    composition: Option<Composition>,

    #[serde(skip)]
    image_zoom: f32,

    #[serde(skip)]
    image_pan: egui::Vec2,

    /// Recomposited only when `composition_dirty` is set, so panning/zooming
    /// doesn't require recompositing every frame.
    #[serde(skip)]
    composition_dirty: bool,

    /// GPU compositor, when a `wgpu` render backend is available (it always
    /// is natively; see `GpuCompositor` for why this does the actual
    /// compositing work instead of `Composition::compose`).
    #[serde(skip)]
    gpu: Option<GpuCompositor>,

    /// Needed alongside `gpu` to register its output texture with egui for
    /// display; see `register_display_texture`.
    #[serde(skip)]
    render_state: Option<egui_wgpu::RenderState>,

    #[serde(skip)]
    display_texture_id: Option<egui::TextureId>,
}

impl Default for LayerCompositorApp {
    fn default() -> Self {
        Self {
            // Example stuff:
            label: "Hello World!".to_owned(),
            value: 2.7,
            file_dialog: FileDialog::new(),
            composition: None,
            image_zoom: 1.0,
            image_pan: egui::Vec2::ZERO,
            composition_dirty: false,
            gpu: None,
            render_state: None,
            display_texture_id: None,
        }
    }
}

impl LayerCompositorApp {
    fn load_exr(&mut self, path: &Path) {
        match Composition::load_exr(path) {
            Ok(composition) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.load(&composition);
                }
                self.composition = Some(composition);
                self.composition_dirty = true;
                self.register_display_texture();
            }
            Err(error) => log::error!("failed to load {}: {error}", path.display()),
        }
    }

    /// Registers the GPU compositor's current output texture with egui's
    /// renderer, so it can be drawn with `ui.painter().image(...)`. Needs
    /// re-doing (freeing the old id first) whenever `gpu.load()` recreates
    /// the underlying texture; recomposing into the same texture doesn't
    /// require it.
    fn register_display_texture(&mut self) {
        let (Some(render_state), Some(gpu)) = (&self.render_state, &self.gpu) else {
            return;
        };
        let Some(view) = gpu.display_view() else {
            return;
        };

        let mut renderer = render_state.renderer.write();
        if let Some(old_id) = self.display_texture_id.take() {
            renderer.free_texture(&old_id);
        }
        let id = renderer.register_native_texture(
            &render_state.device,
            view,
            egui_wgpu::wgpu::FilterMode::Nearest,
        );
        drop(renderer);
        self.display_texture_id = Some(id);
    }

    /// Shows the layer name / level-slider / filter-list table. The last row
    /// is the final composite's own filter list (see `LayerTableDelegate`).
    fn layer_table_ui(&mut self, ui: &mut egui::Ui) {
        let Some(comp) = &mut self.composition else {
            return;
        };

        let mut dirty = false;

        egui_extras::TableBuilder::new(ui)
            .id_salt("layer_table")
            .column(egui_extras::Column::initial(180.0).resizable(true))
            .column(egui_extras::Column::initial(120.0).resizable(true))
            .column(egui_extras::Column::remainder().resizable(true))
            .max_scroll_height(200.0)
            .header(20.0, |mut header| {
                header.col(|ui| {
                    ui.strong("Layer");
                });
                header.col(|ui| {
                    ui.strong("Level");
                });
                header.col(|ui| {
                    ui.strong("Filters");
                });
            })
            .body(|mut body| {
                for layer in &mut comp.layers {
                    let height = filter_row_height(&layer.filters);
                    body.row(height, |mut row| {
                        row.col(|ui| {
                            ui.label(&layer.name);
                        });
                        row.col(|ui| {
                            if ui
                                .add(egui::Slider::new(&mut layer.level, 0.0..=2.0))
                                .changed()
                            {
                                dirty = true;
                            }
                        });
                        row.col(|ui| {
                            filters_cell_ui(ui, &mut layer.filters, &mut dirty);
                        });
                    });
                }

                let height = filter_row_height(&comp.filters);
                body.row(height, |mut row| {
                    row.col(|ui| {
                        ui.strong("Composite");
                    });
                    row.col(|_ui| {});
                    row.col(|ui| {
                        filters_cell_ui(ui, &mut comp.filters, &mut dirty);
                    });
                });
            });

        if dirty {
            self.composition_dirty = true;
        }
    }

    /// Shows the composited image, panable by dragging and zoomable with the
    /// scroll wheel. Recompositing runs on the GPU (see `gpu_compose`), so
    /// it's cheap enough to just do synchronously whenever dirty.
    fn image_viewer_ui(&mut self, ui: &mut egui::Ui) {
        let Some(comp) = &self.composition else {
            return;
        };

        if self.composition_dirty {
            if let Some(gpu) = &mut self.gpu {
                gpu.compose(comp);
            }
            self.composition_dirty = false;
        }

        let comp_size = egui::vec2(comp.size[0] as f32, comp.size[1] as f32);
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

        // The image rect before this frame's pan/zoom input is applied, used
        // as the reference for zooming around the pointer.
        let previous_image_rect = egui::Rect::from_center_size(
            rect.center() + self.image_pan,
            comp_size * self.image_zoom,
        );

        if response.dragged() {
            self.image_pan += response.drag_delta();
        }

        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 && response.hovered() {
            let new_zoom = (self.image_zoom * (scroll * 0.002).exp()).clamp(0.05, 40.0);

            if let Some(pointer) = response.hover_pos() {
                let pointer_fraction =
                    (pointer - previous_image_rect.min) / previous_image_rect.size();
                let new_image_size = comp_size * new_zoom;
                let new_min = pointer - pointer_fraction * new_image_size;
                self.image_pan = (new_min + new_image_size / 2.0) - rect.center();
            }

            self.image_zoom = new_zoom;
        }

        let image_rect = egui::Rect::from_center_size(
            rect.center() + self.image_pan,
            comp_size * self.image_zoom,
        );

        if let Some(texture_id) = self.display_texture_id {
            // Clip to `rect` so panning/zooming never paints over the layer
            // list above.
            ui.painter_at(rect).image(
                texture_id,
                image_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
    }
}

impl LayerCompositorApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        let mut app: Self = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };

        // `gpu`/`render_state` are always skipped by serde, so this has to
        // happen after loading/defaulting the rest of the state above, not
        // inside it.
        app.render_state = cc.wgpu_render_state.clone();
        app.gpu = app
            .render_state
            .as_ref()
            .map(|rs| GpuCompositor::new(&rs.device, &rs.queue));

        app
    }
}

impl eframe::App for LayerCompositorApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::Panel::top("top_panel").show(ui, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::MenuBar::new().ui(ui, |ui| {
                let is_web = cfg!(target_arch = "wasm32");
                ui.menu_button("File", |ui| {
                    if ui.button("Open EXR").clicked() {
                        self.file_dialog.pick_file();
                    }

                    // NOTE: no File->Quit on web pages!
                    if !is_web && ui.button("Quit").clicked() {
                        ui.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.add_space(16.0);
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("exr viewer");

            self.layer_table_ui(ui);

            //ui.add(egui::Slider::new(&mut self.value, 0.0..=10.0).text("value"));
            //if ui.button("Increment").clicked() {
            //self.value += 1.0;
            //}

            self.file_dialog.update(ui);

            if let Some(path) = self.file_dialog.take_picked() {
                self.load_exr(&path);
            }

            self.image_viewer_ui(ui);

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                powered_by_egui_and_eframe(ui);
                egui::warn_if_debug_build(ui);
            });
        });
    }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}
