//! Composites the layers of an EXR file into a single flattened PNG.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::PathBuf;
use std::process::ExitCode;

use exrviewer::{Composition, GpuCompositor};

/// Sets up a headless (windowless) `wgpu` device, for the same
/// `GpuCompositor` the GUI uses to composite on the GPU. Returns `None` if
/// no suitable adapter is available (e.g. a machine/CI runner with no GPU),
/// in which case the caller should fall back to `Composition::compose`.
fn create_gpu_compositor() -> Option<GpuCompositor> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(GpuCompositor::new(&device, &queue))
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(input), Some(output), None) = (args.next(), args.next(), args.next()) else {
        eprintln!("usage: exr_compose <input.exr> <output.png>");
        return ExitCode::FAILURE;
    };

    let input = PathBuf::from(input);
    let output = PathBuf::from(output);

    let composition = match Composition::load_exr(&input) {
        Ok(composition) => composition,
        Err(error) => {
            eprintln!("failed to read {}: {error}", input.display());
            return ExitCode::FAILURE;
        }
    };

    let gpu_rgba = create_gpu_compositor().and_then(|mut gpu| {
        gpu.load(&composition);
        gpu.compose(&composition);
        gpu.read_display_rgba()
    });
    let pixels = gpu_rgba.unwrap_or_else(|| {
        eprintln!("no usable GPU found; compositing on the CPU instead");
        composition.compose()
    });

    // EXR/PNG dimensions never approach `u32::MAX`.
    #[expect(clippy::cast_possible_truncation)]
    let [width, height] = composition.size.map(|dimension| dimension as u32);

    let Some(image) = image::RgbaImage::from_raw(width, height, pixels) else {
        eprintln!("composed pixel buffer does not match the image dimensions");
        return ExitCode::FAILURE;
    };

    if let Err(error) = image.save(&output) {
        eprintln!("failed to write {}: {error}", output.display());
        return ExitCode::FAILURE;
    }

    println!("wrote {}", output.display());
    ExitCode::SUCCESS
}
