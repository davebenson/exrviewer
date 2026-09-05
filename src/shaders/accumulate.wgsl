// Draws one full-screen triangle per layer, additively blending
// `layer_rgb * (layer_alpha * layer_level)` into the accumulation target.
// See gpu_compose.rs for how the blend state completes the formula.

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
@group(1) @binding(0) var layer_tex: texture_2d<f32>;
@group(1) @binding(1) var<uniform> layer_level: f32;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(layer_tex, samp, in.uv);
    let alpha = texel.a * layer_level;
    return vec4<f32>(texel.rgb * alpha, alpha);
}
