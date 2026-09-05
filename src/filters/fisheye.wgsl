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
    let centerx = filter_params.x;
    let centery = filter_params.y;
    let power = filter_params.z;
    let xx = in.uv[0] - centerx;
    let yy = in.uv[1] - centery;

    let orig_r = sqrt(xx * xx + yy * yy);
    let scale = pow(orig_r, power - 1);
    let new_uv = vec2<f32>(xx * scale + centerx, yy * scale + centery);
    return textureSample(source_tex, samp, new_uv);
}
