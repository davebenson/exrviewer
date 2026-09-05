// A separable box blur pass. `filter_params` is `(radius, step_x, step_y, _)`:
// `step` is the UV offset per tap, along whichever axis this pass blurs
// (horizontal or vertical - see blur.rs's two stages). The sampler's
// "clamp to edge" address mode handles taps that fall outside the image.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    var out: VsOut;
    out.uv = uv;
    out.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var samp: sampler;
@group(1) @binding(0) var source_tex: texture_2d<f32>;
@group(1) @binding(1) var<uniform> filter_params: vec4<f32>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let radius = i32(filter_params.x);
    let step = filter_params.yz;

    var sum = vec4<f32>(0.0);
    for (var i: i32 = -radius; i <= radius; i = i + 1) {
        sum = sum + textureSample(source_tex, samp, in.uv + step * f32(i));
    }

    let taps = f32(2 * radius + 1);
    return sum / taps;
}
