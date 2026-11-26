//! Hardware-accelerated OpenGL (`glow`) 3D viewport rendered inside `egui` via `PaintCallback`.
//! Uploads geometry, displacements, normals, and contours once per scene revision and animates/orbits
//! purely with uniforms into an MSAA + 24-bit depth FBO resolved onto `egui`'s viewport rect.

use crate::geometry::V3;
use crate::render::{BANDS, Camera, GpuScene, GpuVertex, background};
use eframe::egui;
use eframe::egui_glow::{self, ShaderVersion};
use eframe::glow::{self, HasContext as _};
use std::sync::{Arc, Mutex};

const TRI_VS_BODY: &str = r#"
attribute_in vec3 a_pos;
attribute_in vec3 a_disp;
attribute_in vec3 a_norm;
attribute_in vec3 a_color;
attribute_in vec2 a_uv;
attribute_in float a_t;
attribute_in float a_kind;

uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_fwd;
uniform vec3 u_target;
uniform vec2 u_view_size;
uniform float u_zoom;
uniform float u_depth_span;
uniform float u_perspective;
uniform float u_deform_k;
uniform vec3 u_light;

varying_out vec3 v_color;
varying_out vec2 v_uv;
varying_out float v_t;
varying_out float v_kind;
varying_out float v_shade;
varying_out float v_nrm_fwd;

void main() {
    vec3 p = a_pos + u_deform_k * a_disp;
    vec3 d = p - u_target;
    float px = dot(d, u_right) * u_zoom;
    float py = dot(d, u_up) * u_zoom;
    float pz = dot(d, u_fwd);
    vec3 view_dir = u_fwd;
    if (u_perspective > 0.5) {
        float dist = max(0.5 * u_view_size.y / (max(u_zoom, 1e-6) * 0.324919696), 1e-3);
        vec3 eye = u_target - u_fwd * dist;
        view_dir = normalize(p - eye);
        float w_clip = max(dist + pz, 1e-4 * dist);
        gl_Position = vec4(
            px * dist / max(0.5 * u_view_size.x, 1.0),
            py * dist / max(0.5 * u_view_size.y, 1.0),
            (pz / u_depth_span) * w_clip,
            w_clip
        );
    } else {
        gl_Position = vec4(
            px / max(0.5 * u_view_size.x, 1.0),
            py / max(0.5 * u_view_size.y, 1.0),
            pz / u_depth_span,
            1.0
        );
    }
    v_color = a_color;
    v_uv = a_uv;
    v_t = a_t;
    v_kind = a_kind;
    v_shade = 0.38 + 0.62 * abs(dot(a_norm, u_light));
    v_nrm_fwd = dot(a_norm, view_dir);
}
"#;

const TRI_FS_BODY: &str = r#"
uniform float u_show_mesh;
uniform float u_enclosure_alpha;
uniform vec3 u_bands[9];

varying_in vec3 v_color;
varying_in vec2 v_uv;
varying_in float v_t;
varying_in float v_kind;
varying_in float v_shade;
varying_in float v_nrm_fwd;

out vec4 o_color;

void main() {
    if ((v_kind < 0.5 || v_kind > 2.5) && v_nrm_fwd > 0.05) {
        discard;
    }
    if (u_enclosure_alpha > 0.99 && v_kind > 0.5 && v_kind < 1.5 && v_nrm_fwd < -0.05) {
        discard;
    }
    vec3 col = v_color;
    if (v_t >= -0.2) {
        int idx = clamp(int(floor(clamp(v_t, 0.0, 1.0) * 9.0)), 0, 8);
        col = u_bands[idx];
    }
    col *= v_shade;
    if (u_show_mesh > 0.5) {
        vec2 fw = max(fwidth(v_uv), vec2(1e-5));
        float edge_px = 1.0 / max(fw.x, fw.y);
        float show = smoothstep(5.0, 7.5, edge_px);
        vec2 d = min(v_uv, 1.0 - v_uv) / fw;
        float dist = min(d.x, d.y);
        float line = 1.0 - smoothstep(0.0, 1.15, dist);
        col *= mix(1.0, 0.55, line * show);
    }
    float alpha = (v_kind > 2.5) ? 0.22 : ((v_kind > 0.5 && v_kind < 1.5) ? u_enclosure_alpha : 1.0);
    o_color = vec4(col, alpha);
}
"#;

const LINE_VS_BODY: &str = r#"
attribute_in vec3 a_pos;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_fwd;
uniform vec3 u_target;
uniform vec2 u_view_size;
uniform float u_zoom;
uniform float u_depth_span;
uniform float u_perspective;

void main() {
    vec3 d = a_pos - u_target;
    float px = dot(d, u_right) * u_zoom;
    float py = dot(d, u_up) * u_zoom;
    float pz = dot(d, u_fwd);
    if (u_perspective > 0.5) {
        float dist = max(0.5 * u_view_size.y / (max(u_zoom, 1e-6) * 0.324919696), 1e-3);
        float w_clip = max(dist + pz, 1e-4 * dist);
        gl_Position = vec4(
            px * dist / max(0.5 * u_view_size.x, 1.0),
            py * dist / max(0.5 * u_view_size.y, 1.0),
            (pz / u_depth_span) * w_clip,
            w_clip
        );
    } else {
        gl_Position = vec4(
            px / max(0.5 * u_view_size.x, 1.0),
            py / max(0.5 * u_view_size.y, 1.0),
            pz / u_depth_span,
            1.0
        );
    }
}
"#;

const LINE_FS_BODY: &str = r#"
uniform vec4 u_line_color;
out vec4 o_color;
void main() {
    o_color = u_line_color;
}
"#;

const BG_VS_BODY: &str = r#"
attribute_in vec2 a_pos;
varying_out float v_y;
void main() {
    v_y = a_pos.y * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"#;

const BG_FS_BODY: &str = r#"
uniform vec3 u_top;
uniform vec3 u_bottom;
varying_in float v_y;
out vec4 o_color;
void main() {
    o_color = vec4(mix(u_bottom, u_top, clamp(v_y, 0.0, 1.0)), 1.0);
}
"#;

const BLIT_VS_BODY: &str = r#"
attribute_in vec2 a_pos;
varying_out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"#;

const BLIT_FS_BODY: &str = r#"
uniform sampler2D u_tex;
varying_in vec2 v_uv;
out vec4 o_color;
void main() {
    o_color = texture2D(u_tex, v_uv);
}
"#;

fn prep_shader(ver: ShaderVersion, body: &str) -> String {
    let mut s = String::from(ver.version_declaration());
    if ver == ShaderVersion::Es300 {
        s.push_str("precision highp float;\n");
    }
    let body = body
        .replace("attribute_in ", "in ")
        .replace("varying_out ", "out ")
        .replace("varying_in ", "in ")
        .replace("texture2D(", "texture(");
    s.push_str(&body);
    s
}

unsafe fn compile_program(gl: &glow::Context, ver: ShaderVersion, vs: &str, fs: &str) -> Result<glow::Program, String> {
    let vs_src = prep_shader(ver, vs);
    let fs_src = prep_shader(ver, fs);
    let v = unsafe { gl.create_shader(glow::VERTEX_SHADER)? };
    unsafe {
        gl.shader_source(v, &vs_src);
        gl.compile_shader(v);
        if !gl.get_shader_compile_status(v) {
            let log = gl.get_shader_info_log(v);
            gl.delete_shader(v);
            return Err(format!("vertex shader compile error: {log}"));
        }
    }
    let f = unsafe { gl.create_shader(glow::FRAGMENT_SHADER)? };
    unsafe {
        gl.shader_source(f, &fs_src);
        gl.compile_shader(f);
        if !gl.get_shader_compile_status(f) {
            let log = gl.get_shader_info_log(f);
            gl.delete_shader(v);
            gl.delete_shader(f);
            return Err(format!("fragment shader compile error: {log}"));
        }
    }
    let prog = unsafe { gl.create_program()? };
    unsafe {
        gl.attach_shader(prog, v);
        gl.attach_shader(prog, f);
        gl.link_program(prog);
        gl.detach_shader(prog, v);
        gl.detach_shader(prog, f);
        gl.delete_shader(v);
        gl.delete_shader(f);
        if !gl.get_program_link_status(prog) {
            let log = gl.get_program_info_log(prog);
            gl.delete_program(prog);
            return Err(format!("program link error: {log}"));
        }
    }
    Ok(prog)
}

#[derive(Clone, PartialEq)]
pub struct UploadKey {
    pub rev: u64,
    pub sel: u64,
    pub mesh: usize,
    pub dark: bool,
}

#[derive(Clone)]
pub struct DrawParams {
    pub cam: Camera,
    pub w_pts: f32,
    pub h_pts: f32,
    pub anim: f64,
    pub show_mesh: bool,
    pub dark: bool,
    pub translucent_enclosure: bool,
    pub scene: Arc<GpuScene>,
    pub upload_key: UploadKey,
}

struct FboTargets {
    w: i32,
    h: i32,
    fbo_msaa: Option<glow::Framebuffer>,
    rbo_color_msaa: Option<glow::Renderbuffer>,
    rbo_depth_msaa: Option<glow::Renderbuffer>,
    fbo_resolve: glow::Framebuffer,
    tex_resolve: glow::Texture,
    rbo_depth_single: Option<glow::Renderbuffer>,
}

struct Inner {
    gl: Arc<glow::Context>,
    ver: ShaderVersion,
    max_samples: i32,
    prog_mesh: glow::Program,
    prog_line: glow::Program,
    prog_bg: glow::Program,
    prog_blit: glow::Program,
    vao_mesh: glow::VertexArray,
    vbo_mesh: glow::Buffer,
    ebo_solid: glow::Buffer,
    ebo_enc: glow::Buffer,
    ebo_ghost: glow::Buffer,
    solid_len: i32,
    enc_len: i32,
    ghost_len: i32,
    vao_line: glow::VertexArray,
    vbo_line: glow::Buffer,
    line_len: i32,
    vao_quad: glow::VertexArray,
    vbo_quad: glow::Buffer,
    targets: Option<FboTargets>,
    uploaded: Option<UploadKey>,
    lo: V3,
    hi: V3,
    destroyed: bool,
}

/// Persistent GPU buffers, shaders, and offscreen MSAA/depth render targets for the 3D viewport.
pub struct GlViewport {
    inner: Arc<Mutex<Inner>>,
}

impl GlViewport {
    pub fn try_new(gl: &Arc<glow::Context>) -> Option<Self> {
        let ver = ShaderVersion::get(gl);
        if !ver.is_new_shader_interface() {
            return None;
        }
        unsafe { Self::init(gl.clone(), ver).ok() }
    }

    unsafe fn init(gl: Arc<glow::Context>, ver: ShaderVersion) -> Result<Self, String> {
        let prog_mesh = unsafe { compile_program(&gl, ver, TRI_VS_BODY, TRI_FS_BODY)? };
        let prog_line = unsafe { compile_program(&gl, ver, LINE_VS_BODY, LINE_FS_BODY)? };
        let prog_bg = unsafe { compile_program(&gl, ver, BG_VS_BODY, BG_FS_BODY)? };
        let prog_blit = unsafe { compile_program(&gl, ver, BLIT_VS_BODY, BLIT_FS_BODY)? };

        let vao_mesh = unsafe { gl.create_vertex_array()? };
        let vbo_mesh = unsafe { gl.create_buffer()? };
        let ebo_solid = unsafe { gl.create_buffer()? };
        let ebo_enc = unsafe { gl.create_buffer()? };
        let ebo_ghost = unsafe { gl.create_buffer()? };
        let stride = std::mem::size_of::<GpuVertex>() as i32;
        unsafe {
            gl.bind_vertex_array(Some(vao_mesh));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_mesh));
            for (name, comps, offset) in [
                ("a_pos", 3, 0),
                ("a_disp", 3, 12),
                ("a_norm", 3, 24),
                ("a_color", 3, 36),
                ("a_uv", 2, 48),
                ("a_t", 1, 56),
                ("a_kind", 1, 60),
            ] {
                if let Some(loc) = gl.get_attrib_location(prog_mesh, name) {
                    gl.enable_vertex_attrib_array(loc);
                    gl.vertex_attrib_pointer_f32(loc, comps, glow::FLOAT, false, stride, offset);
                }
            }
            gl.bind_vertex_array(None);
        }

        let vao_line = unsafe { gl.create_vertex_array()? };
        let vbo_line = unsafe { gl.create_buffer()? };
        unsafe {
            gl.bind_vertex_array(Some(vao_line));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_line));
            if let Some(loc) = gl.get_attrib_location(prog_line, "a_pos") {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, 3, glow::FLOAT, false, 12, 0);
            }
            gl.bind_vertex_array(None);
        }

        let vao_quad = unsafe { gl.create_vertex_array()? };
        let vbo_quad = unsafe { gl.create_buffer()? };
        let quad: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        unsafe {
            gl.bind_vertex_array(Some(vao_quad));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_quad));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&quad), glow::STATIC_DRAW);
            for prog in [prog_bg, prog_blit] {
                if let Some(loc) = gl.get_attrib_location(prog, "a_pos") {
                    gl.enable_vertex_attrib_array(loc);
                    gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 8, 0);
                }
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }

        let max_samples = unsafe { gl.get_parameter_i32(glow::MAX_SAMPLES) }.clamp(0, 4);
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                gl,
                ver,
                max_samples,
                prog_mesh,
                prog_line,
                prog_bg,
                prog_blit,
                vao_mesh,
                vbo_mesh,
                ebo_solid,
                ebo_enc,
                ebo_ghost,
                solid_len: 0,
                enc_len: 0,
                ghost_len: 0,
                vao_line,
                vbo_line,
                line_len: 0,
                vao_quad,
                vbo_quad,
                targets: None,
                uploaded: None,
                lo: [0.0; 3],
                hi: [0.0; 3],
                destroyed: false,
            })),
        })
    }

    pub fn paint_callback(&self, rect: egui::Rect, params: DrawParams) -> egui::PaintCallback {
        let state = self.inner.clone();
        egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, _painter| {
                let mut inner = state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if inner.destroyed {
                    return;
                }
                inner.render(info, &params);
            })),
        }
    }

    pub fn destroy(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.destroy();
    }
}

impl Drop for GlViewport {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.destroy();
        }
    }
}

impl Inner {
    unsafe fn drop_targets(&mut self) {
        if let Some(t) = self.targets.take() {
            unsafe {
                if let Some(fb) = t.fbo_msaa { self.gl.delete_framebuffer(fb); }
                if let Some(rb) = t.rbo_color_msaa { self.gl.delete_renderbuffer(rb); }
                if let Some(rb) = t.rbo_depth_msaa { self.gl.delete_renderbuffer(rb); }
                self.gl.delete_framebuffer(t.fbo_resolve);
                self.gl.delete_texture(t.tex_resolve);
                if let Some(rb) = t.rbo_depth_single { self.gl.delete_renderbuffer(rb); }
            }
        }
    }

    unsafe fn ensure_targets(&mut self, w: i32, h: i32) -> bool {
        if let Some(t) = &self.targets {
            if t.w == w && t.h == h {
                return true;
            }
        }
        unsafe { self.drop_targets(); }
        let gl = &self.gl;
        unsafe {
            let Ok(fbo_resolve) = gl.create_framebuffer() else { return false };
            let Ok(tex_resolve) = gl.create_texture() else {
                gl.delete_framebuffer(fbo_resolve);
                return false;
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(tex_resolve));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, w, h, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
            gl.bind_texture(glow::TEXTURE_2D, None);

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo_resolve));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex_resolve), 0);

            if self.max_samples >= 2 {
                if let (Ok(fbo_msaa), Ok(rbo_c), Ok(rbo_d)) = (gl.create_framebuffer(), gl.create_renderbuffer(), gl.create_renderbuffer()) {
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rbo_c));
                    gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, self.max_samples, glow::RGBA8, w, h);
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rbo_d));
                    gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, self.max_samples, glow::DEPTH_COMPONENT24, w, h);
                    gl.bind_renderbuffer(glow::RENDERBUFFER, None);

                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo_msaa));
                    gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(rbo_c));
                    gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(rbo_d));
                    if gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE {
                        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo_resolve));
                        if gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE {
                            self.targets = Some(FboTargets {
                                w,
                                h,
                                fbo_msaa: Some(fbo_msaa),
                                rbo_color_msaa: Some(rbo_c),
                                rbo_depth_msaa: Some(rbo_d),
                                fbo_resolve,
                                tex_resolve,
                                rbo_depth_single: None,
                            });
                            return true;
                        }
                    }
                    gl.delete_framebuffer(fbo_msaa);
                    gl.delete_renderbuffer(rbo_c);
                    gl.delete_renderbuffer(rbo_d);
                }
            }

            let Ok(rbo_d) = gl.create_renderbuffer() else {
                gl.delete_framebuffer(fbo_resolve);
                gl.delete_texture(tex_resolve);
                return false;
            };
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rbo_d));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, w, h);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo_resolve));
            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(rbo_d));
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                gl.delete_renderbuffer(rbo_d);
                gl.delete_framebuffer(fbo_resolve);
                gl.delete_texture(tex_resolve);
                return false;
            }
            self.targets = Some(FboTargets {
                w,
                h,
                fbo_msaa: None,
                rbo_color_msaa: None,
                rbo_depth_msaa: None,
                fbo_resolve,
                tex_resolve,
                rbo_depth_single: Some(rbo_d),
            });
            true
        }
    }

    fn upload_scene(&mut self, key: &UploadKey, scene: &GpuScene) {
        if self.uploaded.as_ref() == Some(key) {
            return;
        }
        let gl = &self.gl;
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo_mesh));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&scene.vertices), glow::STATIC_DRAW);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo_solid));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, bytemuck::cast_slice(&scene.solid_indices), glow::STATIC_DRAW);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo_enc));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, bytemuck::cast_slice(&scene.enclosure_indices), glow::STATIC_DRAW);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo_ghost));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, bytemuck::cast_slice(&scene.ghost_indices), glow::STATIC_DRAW);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo_line));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&scene.undeformed_lines), glow::STATIC_DRAW);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
        }
        self.solid_len = scene.solid_indices.len() as i32;
        self.enc_len = scene.enclosure_indices.len() as i32;
        self.ghost_len = scene.ghost_indices.len() as i32;
        self.line_len = scene.undeformed_lines.len() as i32;
        self.lo = scene.lo;
        self.hi = scene.hi;
        self.uploaded = Some(key.clone());
    }

    fn render(&mut self, info: egui::PaintCallbackInfo, p: &DrawParams) {
        let vp = info.viewport_in_pixels();
        if vp.width_px < 4 || vp.height_px < 4 {
            return;
        }
        let gl = self.gl.clone();
        self.upload_scene(&p.upload_key, &p.scene);
        let prev_fbo = unsafe { gl.get_parameter_i32(glow::DRAW_FRAMEBUFFER_BINDING) };
        let prev_fb_obj = std::num::NonZeroU32::new(prev_fbo as u32).map(glow::NativeFramebuffer);
        if !unsafe { self.ensure_targets(vp.width_px, vp.height_px) } {
            unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, prev_fb_obj); }
            return;
        }
        let t = self.targets.as_ref().unwrap();
        let draw_fbo = t.fbo_msaa.unwrap_or(t.fbo_resolve);

        let (r, u, f) = p.cam.basis();
        let light = crate::geometry::normalize(crate::geometry::sub(
            crate::geometry::scale(f, -1.0),
            crate::geometry::add(crate::geometry::scale(r, 0.35), crate::geometry::scale(u, -0.45)),
        ));
        let center = [
            0.5 * (self.lo[0] + self.hi[0]),
            0.5 * (self.lo[1] + self.hi[1]),
            0.5 * (self.lo[2] + self.hi[2]),
        ];
        let diag = crate::geometry::norm(crate::geometry::sub(self.hi, self.lo)).max(1.0);
        let dist_c = crate::geometry::dot(crate::geometry::sub(center, p.cam.target), f).abs();
        let depth_span = (dist_c + diag * 2.5).max(10.0) as f32;
        let (bg_top, bg_bot) = background(p.dark);

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(draw_fbo));
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.viewport(0, 0, vp.width_px, vp.height_px);
            gl.depth_mask(true);
            gl.clear_depth(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);

            // Gradient background
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
            gl.use_program(Some(self.prog_bg));
            if let Some(loc) = gl.get_uniform_location(self.prog_bg, "u_top") {
                gl.uniform_3_f32(Some(&loc), bg_top[0], bg_top[1], bg_top[2]);
            }
            if let Some(loc) = gl.get_uniform_location(self.prog_bg, "u_bottom") {
                gl.uniform_3_f32(Some(&loc), bg_bot[0], bg_bot[1], bg_bot[2]);
            }
            gl.bind_vertex_array(Some(self.vao_quad));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // Main 3D mesh pass
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LESS);
            gl.depth_mask(true);
            gl.use_program(Some(self.prog_mesh));

            let u3 = |name: &str, v: [f64; 3]| {
                if let Some(loc) = gl.get_uniform_location(self.prog_mesh, name) {
                    gl.uniform_3_f32(Some(&loc), v[0] as f32, v[1] as f32, v[2] as f32);
                }
            };
            let u1 = |name: &str, v: f32| {
                if let Some(loc) = gl.get_uniform_location(self.prog_mesh, name) {
                    gl.uniform_1_f32(Some(&loc), v);
                }
            };
            u3("u_right", r);
            u3("u_up", u);
            u3("u_fwd", f);
            u3("u_target", p.cam.target);
            u3("u_light", light);
            if let Some(loc) = gl.get_uniform_location(self.prog_mesh, "u_view_size") {
                gl.uniform_2_f32(Some(&loc), p.w_pts, p.h_pts);
            }
            u1("u_zoom", p.cam.zoom as f32);
            u1("u_depth_span", depth_span);
            u1("u_perspective", if p.cam.perspective { 1.0 } else { 0.0 });
            u1("u_deform_k", p.anim as f32);
            u1("u_show_mesh", if p.show_mesh { 1.0 } else { 0.0 });
            if let Some(loc) = gl
                .get_uniform_location(self.prog_mesh, "u_bands")
                .or_else(|| gl.get_uniform_location(self.prog_mesh, "u_bands[0]"))
            {
                let flat: Vec<f32> = BANDS.iter().flat_map(|b| [b[0] as f32 / 255.0, b[1] as f32 / 255.0, b[2] as f32 / 255.0]).collect();
                gl.uniform_3_f32_slice(Some(&loc), &flat);
            }
            gl.bind_vertex_array(Some(self.vao_mesh));

            if self.solid_len > 0 {
                u1("u_enclosure_alpha", 1.0);
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo_solid));
                gl.draw_elements(glow::TRIANGLES, self.solid_len, glow::UNSIGNED_INT, 0);
            }

            if self.line_len > 0 {
                gl.enable(glow::BLEND);
                gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
                gl.depth_func(glow::LEQUAL);
                gl.use_program(Some(self.prog_line));
                let lu3 = |name: &str, v: [f64; 3]| {
                    if let Some(loc) = gl.get_uniform_location(self.prog_line, name) {
                        gl.uniform_3_f32(Some(&loc), v[0] as f32, v[1] as f32, v[2] as f32);
                    }
                };
                lu3("u_right", r);
                lu3("u_up", u);
                lu3("u_fwd", f);
                lu3("u_target", p.cam.target);
                if let Some(loc) = gl.get_uniform_location(self.prog_line, "u_view_size") {
                    gl.uniform_2_f32(Some(&loc), p.w_pts, p.h_pts);
                }
                for (name, val) in [("u_zoom", p.cam.zoom as f32), ("u_depth_span", depth_span), ("u_perspective", if p.cam.perspective { 1.0 } else { 0.0 })] {
                    if let Some(loc) = gl.get_uniform_location(self.prog_line, name) {
                        gl.uniform_1_f32(Some(&loc), val);
                    }
                }
                if let Some(loc) = gl.get_uniform_location(self.prog_line, "u_line_color") {
                    gl.uniform_4_f32(Some(&loc), 0.47, 0.47, 0.51, 0.55);
                }
                gl.bind_vertex_array(Some(self.vao_line));
                gl.draw_arrays(glow::LINES, 0, self.line_len);
                gl.depth_func(glow::LESS);
                gl.disable(glow::BLEND);
                gl.use_program(Some(self.prog_mesh));
                gl.bind_vertex_array(Some(self.vao_mesh));
            }

            if self.enc_len > 0 {
                if p.translucent_enclosure {
                    gl.enable(glow::BLEND);
                    gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
                    gl.depth_mask(false);
                    u1("u_enclosure_alpha", 0.22);
                    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo_enc));
                    gl.draw_elements(glow::TRIANGLES, self.enc_len, glow::UNSIGNED_INT, 0);
                    gl.depth_mask(true);
                    gl.disable(glow::BLEND);
                } else {
                    u1("u_enclosure_alpha", 1.0);
                    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo_enc));
                    gl.draw_elements(glow::TRIANGLES, self.enc_len, glow::UNSIGNED_INT, 0);
                }
            }
            if self.ghost_len > 0 {
                gl.enable(glow::BLEND);
                gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
                gl.depth_mask(false);
                u1("u_enclosure_alpha", 0.22);
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo_ghost));
                gl.draw_elements(glow::TRIANGLES, self.ghost_len, glow::UNSIGNED_INT, 0);
                gl.depth_mask(true);
                gl.disable(glow::BLEND);
            }

            if t.fbo_msaa.is_some() {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(draw_fbo));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(t.fbo_resolve));
                gl.blit_framebuffer(0, 0, vp.width_px, vp.height_px, 0, 0, vp.width_px, vp.height_px, glow::COLOR_BUFFER_BIT, glow::NEAREST);
            }

            // Blit resolved color texture back to egui's active framebuffer inside its scissor clip rect
            gl.bind_framebuffer(glow::FRAMEBUFFER, prev_fb_obj);
            let clip = info.clip_rect_in_pixels();
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(clip.left_px, clip.from_bottom_px, clip.width_px, clip.height_px);
            gl.viewport(vp.left_px, vp.from_bottom_px, vp.width_px, vp.height_px);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.use_program(Some(self.prog_blit));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(t.tex_resolve));
            if let Some(loc) = gl.get_uniform_location(self.prog_blit, "u_tex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.bind_vertex_array(Some(self.vao_quad));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.use_program(None);
            let _ = self.ver;
        }
    }

    fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;
        unsafe {
            self.drop_targets();
            self.gl.delete_vertex_array(self.vao_mesh);
            self.gl.delete_buffer(self.vbo_mesh);
            self.gl.delete_buffer(self.ebo_solid);
            self.gl.delete_buffer(self.ebo_enc);
            self.gl.delete_buffer(self.ebo_ghost);
            self.gl.delete_vertex_array(self.vao_line);
            self.gl.delete_buffer(self.vbo_line);
            self.gl.delete_vertex_array(self.vao_quad);
            self.gl.delete_buffer(self.vbo_quad);
            self.gl.delete_program(self.prog_mesh);
            self.gl.delete_program(self.prog_line);
            self.gl.delete_program(self.prog_bg);
            self.gl.delete_program(self.prog_blit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_expand_cleanly_for_gl140_and_es300() {
        for ver in [ShaderVersion::Gl140, ShaderVersion::Es300] {
            for body in [TRI_VS_BODY, TRI_FS_BODY, LINE_VS_BODY, LINE_FS_BODY, BG_VS_BODY, BG_FS_BODY, BLIT_VS_BODY, BLIT_FS_BODY] {
                let s = prep_shader(ver, body);
                assert!(s.starts_with(ver.version_declaration()));
                assert!(!s.contains("attribute_in "));
                assert!(!s.contains("varying_out "));
                assert!(!s.contains("varying_in "));
                assert!(!s.contains("texture2D("));
            }
        }
    }
}
