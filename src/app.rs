/// We derive Deserialize/Serialize so we can persist app state on shutdown.
use egui_file_dialog::FileDialog;
use std::path::Path;

use crate::gpu_compose::GpuCompositor;
use crate::{Composition, CompositionLayer};

/// Renders the layer name / level-slider table shown above the image preview.
struct LayerTableDelegate<'a> {
    layers: &'a mut [CompositionLayer],
    dirty: &'a mut bool,
}

impl egui_table::TableDelegate for LayerTableDelegate<'_> {
    fn header_cell_ui(&mut self, ui: &mut egui::Ui, cell: &egui_table::HeaderCellInfo) {
        let title = if cell.col_range.start == 0 {
            "Layer"
        } else {
            "Level"
        };
        ui.strong(title);
    }

    fn cell_ui(&mut self, ui: &mut egui::Ui, cell: &egui_table::CellInfo) {
        #[expect(clippy::cast_possible_truncation)]
        let Some(layer) = self.layers.get_mut(cell.row_nr as usize) else {
            return;
        };

        if cell.col_nr == 0 {
            ui.label(&layer.name);
        } else {
            let slider = ui.add(egui::Slider::new(&mut layer.level, 0.0..=2.0));
            if slider.changed() {
                *self.dirty = true;
            }
        }
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

    /// Shows the layer name / level-slider table, and returns its height.
    fn layer_table_ui(&mut self, ui: &mut egui::Ui) {
        let Some(comp) = &mut self.composition else {
            return;
        };

        let num_rows = comp.layers.len() as u64;
        let header_height = 20.0;
        let row_height = 20.0;
        #[expect(clippy::cast_precision_loss)]
        let table_height =
            (header_height + num_rows as f32 * row_height).clamp(header_height + row_height, 200.0);

        let mut delegate = LayerTableDelegate {
            layers: &mut comp.layers,
            dirty: &mut self.composition_dirty,
        };

        ui.allocate_ui(egui::vec2(ui.available_width(), table_height), |ui| {
            egui_table::Table::new()
                .id_salt("layer_table")
                .num_rows(num_rows)
                .columns(vec![
                    egui_table::Column::new(180.0).resizable(true),
                    egui_table::Column::new(120.0).resizable(true),
                ])
                .headers(vec![egui_table::HeaderRow::new(header_height)])
                .show(ui, &mut delegate);
        });
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
                    // NOTE: no File->Quit on web pages!
                    if !is_web && ui.button("Quit").clicked() {
                        ui.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button("Open EXR").clicked() {
                        self.file_dialog.pick_file();
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

            ui.separator();

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
