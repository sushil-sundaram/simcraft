//! Optional GPU acceleration: double-precision Jacobi-preconditioned conjugate gradients running as
//! GPU compute shaders (via wgpu), preferring the discrete GPU.
//!
//! The GPU is only touched when a solve actually uses it (never at app start-up), and the device
//! is released when the solver is dropped so a hybrid laptop's dGPU can power down again.

use crate::sparse::NodeSparse;
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
use std::sync::Mutex;

/// Where linear solves run.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ComputeMode {
    /// CPU direct solver when it is cheap, GPU for large iterative solves (falls back to CPU).
    #[default]
    Auto,
    /// Always CPU.
    Cpu,
    /// GPU for every linear solve (falls back to CPU if no suitable GPU).
    Gpu,
}

thread_local! {
    static MODE: Cell<ComputeMode> = const { Cell::new(ComputeMode::Auto) };
    static LAST_DEVICE: RefCell<Option<String>> = const { RefCell::new(None) };
    static FALLBACK: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Record why a requested GPU solve fell back to the CPU (reported as a result warning).
pub fn note_fallback(msg: String) {
    FALLBACK.with(|f| *f.borrow_mut() = Some(msg));
}
pub fn take_fallback() -> Option<String> {
    FALLBACK.with(|f| f.borrow_mut().take())
}

pub fn set_mode(m: ComputeMode) {
    MODE.with(|c| c.set(m));
}
pub fn mode() -> ComputeMode {
    MODE.with(|c| c.get())
}
pub(crate) fn set_last_device(name: String) {
    LAST_DEVICE.with(|d| *d.borrow_mut() = Some(name));
}
/// Name of the GPU used by the last GPU solve on this thread (None if no GPU solve has run).
pub fn last_device() -> Option<String> {
    LAST_DEVICE.with(|d| d.borrow().clone())
}

const WG: u32 = 256;
const BATCH: usize = 25;

const SHADER: &str = r#"
// ELLPACK storage, column-major (entry k of row i at k*n + i) so neighbouring threads read
// neighbouring addresses; padding entries point at the row itself with value 0.
struct Params { n: u32, nwg: u32, width: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> ecol: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f64>;
@group(0) @binding(3) var<storage, read> dinv: array<f64>;
@group(0) @binding(4) var<storage, read_write> x: array<f64>;
@group(0) @binding(5) var<storage, read_write> r: array<f64>;
@group(0) @binding(6) var<storage, read_write> z: array<f64>;
@group(0) @binding(7) var<storage, read_write> p: array<f64>;
@group(0) @binding(8) var<storage, read_write> ap: array<f64>;
@group(0) @binding(9) var<storage, read_write> partial: array<f64>;
// sc: 0 = rz, 1 = pAp, 2 = rz_new, 3 = rr, 4 = beta
@group(0) @binding(10) var<storage, read_write> sc: array<f64>;
@group(0) @binding(11) var<uniform> prm: Params;

var<workgroup> sh: array<f64, 256>;
var<workgroup> sh2: array<f64, 256>;

@compute @workgroup_size(256)
fn spmv(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < prm.n) {
        var s = f64(0.0);
        let n = prm.n;
        for (var k = 0u; k < prm.width; k = k + 1u) {
            let idx = k * n + i;
            s = s + vals[idx] * p[ecol[idx]];
        }
        ap[i] = s;
    }
}

@compute @workgroup_size(256)
fn dot_pap(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = f64(0.0);
    if (g.x < prm.n) { v = p[g.x] * ap[g.x]; }
    sh[l.x] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (l.x < s) { sh[l.x] = sh[l.x] + sh[l.x + s]; }
        workgroupBarrier();
    }
    if (l.x == 0u) { partial[w.x] = sh[0]; }
}

@compute @workgroup_size(256)
fn reduce_pap(@builtin(local_invocation_id) l: vec3<u32>) {
    var v = f64(0.0);
    for (var i = l.x; i < prm.nwg; i = i + 256u) { v = v + partial[i]; }
    sh[l.x] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (l.x < s) { sh[l.x] = sh[l.x] + sh[l.x + s]; }
        workgroupBarrier();
    }
    if (l.x == 0u) { sc[1] = sh[0]; }
}

@compute @workgroup_size(256)
fn update_xr(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < prm.n) {
        var alpha = f64(0.0);
        if (sc[1] > f64(0.0)) { alpha = sc[0] / sc[1]; }
        x[i] = x[i] + alpha * p[i];
        r[i] = r[i] - alpha * ap[i];
        z[i] = dinv[i] * r[i];
    }
}

@compute @workgroup_size(256)
fn dot_rz_rr(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var a = f64(0.0);
    var b = f64(0.0);
    if (g.x < prm.n) { a = r[g.x] * z[g.x]; b = r[g.x] * r[g.x]; }
    sh[l.x] = a;
    sh2[l.x] = b;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (l.x < s) { sh[l.x] = sh[l.x] + sh[l.x + s]; sh2[l.x] = sh2[l.x] + sh2[l.x + s]; }
        workgroupBarrier();
    }
    if (l.x == 0u) { partial[2u * w.x] = sh[0]; partial[2u * w.x + 1u] = sh2[0]; }
}

@compute @workgroup_size(256)
fn reduce_rz(@builtin(local_invocation_id) l: vec3<u32>) {
    var a = f64(0.0);
    var b = f64(0.0);
    for (var i = l.x; i < prm.nwg; i = i + 256u) { a = a + partial[2u * i]; b = b + partial[2u * i + 1u]; }
    sh[l.x] = a;
    sh2[l.x] = b;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (l.x < s) { sh[l.x] = sh[l.x] + sh[l.x + s]; sh2[l.x] = sh2[l.x] + sh2[l.x + s]; }
        workgroupBarrier();
    }
    if (l.x == 0u) {
        sc[2] = sh[0];
        sc[3] = sh2[0];
        var beta = f64(0.0);
        if (sc[0] != f64(0.0)) { beta = sh[0] / sc[0]; }
        sc[4] = beta;
        sc[0] = sh[0];
    }
}

@compute @workgroup_size(256)
fn update_p(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < prm.n) { p[i] = z[i] + sc[4] * p[i]; }
}
"#;

struct Kernels {
    spmv: wgpu::ComputePipeline,
    dot_pap: wgpu::ComputePipeline,
    reduce_pap: wgpu::ComputePipeline,
    update_xr: wgpu::ComputePipeline,
    dot_rz_rr: wgpu::ComputePipeline,
    reduce_rz: wgpu::ComputePipeline,
    update_p: wgpu::ComputePipeline,
}

/// A sparse SPD matrix resident on the GPU, ready for repeated PCG solves.
pub struct GpuSystem {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Kernels,
    bind: wgpu::BindGroup,
    bufs: Vec<wgpu::Buffer>,
    staging_sc: wgpu::Buffer,
    staging_x: wgpu::Buffer,
    n: usize,
    nwg: u32,
    dinv: Vec<f64>,
    pub device_name: String,
    fault: std::sync::Arc<Mutex<Option<String>>>,
}

/// Pick the best GPU adapter across `items`: DiscreteGpu before IntegratedGpu, skipping CPU/Virtual
/// devices and optionally requiring `SHADER_F64`.
pub(crate) fn select_adapter_index<T>(
    items: &[T],
    require_f64: bool,
    dev_type: impl Fn(&T) -> wgpu::DeviceType,
    has_f64: impl Fn(&T) -> bool,
) -> Option<usize> {
    let rank = |item: &T| match dev_type(item) {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        _ => 9,
    };
    (0..items.len())
        .filter(|&i| (!require_f64 || has_f64(&items[i])) && rank(&items[i]) < 9)
        .min_by_key(|&i| rank(&items[i]))
}

pub(crate) fn pick_adapter(require_f64: bool) -> Result<wgpu::Adapter> {
    let backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let mut adapters = adapters;
    let idx = select_adapter_index(&adapters, require_f64, |a| a.get_info().device_type, |a| a.features().contains(wgpu::Features::SHADER_F64));
    match idx {
        Some(i) => Ok(adapters.swap_remove(i)),
        None => Err(if require_f64 {
            anyhow!("no Vulkan/Metal/DX12 GPU with 64-bit float shader support found")
        } else {
            anyhow!("no Vulkan/Metal/DX12 GPU adapter found")
        }),
    }
}

pub(crate) fn read_f64(device: &wgpu::Device, buf: &wgpu::Buffer, len: usize) -> Result<Vec<f64>> {
    let slice = buf.slice(..(len * 8) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
    rx.recv().map_err(|_| anyhow!("GPU map callback lost"))?.map_err(|e| anyhow!("GPU buffer map failed: {e:?}"))?;
    let out: Vec<f64> = {
        let data = slice.get_mapped_range().map_err(|e| anyhow!("GPU buffer view failed: {e:?}"))?;
        bytemuck::cast_slice::<u8, f64>(&data[..len * 8]).to_vec()
    };
    buf.unmap();
    Ok(out)
}

impl GpuSystem {
    /// Upload the matrix (after Dirichlet conditions have been applied) to the GPU.
    pub fn new(a: &NodeSparse) -> Result<GpuSystem> {
        let adapter = pick_adapter(true)?;
        let info = adapter.get_info();
        let al = adapter.limits();
        let n = a.n();
        let bs = a.bs;
        // ELLPACK (column-major) from the node-block layout
        let width = (0..a.ptr.len() - 1).map(|nd| (a.ptr[nd + 1] - a.ptr[nd]) * bs).max().unwrap_or(1);
        if (width as u64) * (n as u64) >= u32::MAX as u64 {
            bail!("matrix too large for 32-bit GPU indexing");
        }
        let need = (width * n * 8) as u64;
        if need > al.max_storage_buffer_binding_size as u64 || need > al.max_buffer_size {
            bail!("matrix ({} MB) exceeds the GPU's buffer size limit", need / 1_000_000);
        }
        let mut ecol: Vec<u32> = vec![0; width * n];
        let mut evals: Vec<f64> = vec![0.0; width * n];
        for node in 0..a.ptr.len() - 1 {
            let nb = &a.adj[a.ptr[node]..a.ptr[node + 1]];
            let deg = nb.len();
            for c in 0..bs {
                let row = node * bs + c;
                let rs = a.ptr[node] * bs * bs + c * deg * bs;
                let mut k = 0;
                for (q, &m) in nb.iter().enumerate() {
                    for d in 0..bs {
                        ecol[k * n + row] = m * bs as u32 + d as u32;
                        evals[k * n + row] = a.vals[rs + q * bs + d];
                        k += 1;
                    }
                }
                for k in k..width {
                    ecol[k * n + row] = row as u32;
                }
            }
        }
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 11;
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("simcraft-compute"),
            required_features: wgpu::Features::SHADER_F64,
            required_limits: limits,
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        }))
        .map_err(|e| anyhow!("could not open GPU device: {e}"))?;

        // Never let a GPU error abort the app: record it and fall back to the CPU instead.
        let fault: std::sync::Arc<Mutex<Option<String>>> = Default::default();
        let f2 = fault.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            *f2.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(e.to_string());
        }));
        let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pcg"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..11).map(|b| storage(b, b < 4)).collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 11,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let kernels = Kernels {
            spmv: mk("spmv"),
            dot_pap: mk("dot_pap"),
            reduce_pap: mk("reduce_pap"),
            update_xr: mk("update_xr"),
            dot_rz_rr: mk("dot_rz_rr"),
            reduce_rz: mk("reduce_rz"),
            update_p: mk("update_p"),
        };

        let nwg = (n as u32).div_ceil(WG);
        let dinv: Vec<f64> = a.diag().iter().map(|d| if d.abs() > 0.0 { 1.0 / d } else { 1.0 }).collect();
        let usage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        // unmapped buffers + queued uploads: an out-of-memory allocation is then reported through the
        // error scope instead of panicking inside a mapping
        let init = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            let size = (bytes.len() as u64).div_ceil(4).max(1) * 4;
            let b = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            if !bytes.is_empty() {
                queue.write_buffer(&b, 0, bytes);
            }
            b
        };
        let zeros = vec![0.0f64; n];
        let bufs = vec![
            init("ell_col", bytemuck::cast_slice(&ecol), wgpu::BufferUsages::STORAGE),
            init("unused", bytemuck::cast_slice(&[0u32; 4]), wgpu::BufferUsages::STORAGE),
            init("ell_val", bytemuck::cast_slice(&evals), wgpu::BufferUsages::STORAGE),
            init("dinv", bytemuck::cast_slice(&dinv), wgpu::BufferUsages::STORAGE),
            init("x", bytemuck::cast_slice(&zeros), usage_rw),
            init("r", bytemuck::cast_slice(&zeros), usage_rw),
            init("z", bytemuck::cast_slice(&zeros), usage_rw),
            init("p", bytemuck::cast_slice(&zeros), usage_rw),
            init("ap", bytemuck::cast_slice(&zeros), usage_rw),
            init("partial", bytemuck::cast_slice(&vec![0.0f64; 2 * nwg as usize + 2]), usage_rw),
            init("sc", bytemuck::cast_slice(&[0.0f64; 8]), usage_rw),
            init("params", bytemuck::cast_slice(&[n as u32, nwg, width as u32, 0]), wgpu::BufferUsages::UNIFORM),
        ];
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect::<Vec<_>>(),
        });
        let staging = |size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let staging_sc = staging(64);
        let staging_x = staging((n * 8) as u64);
        let device_name = info.name.clone();
        set_last_device(device_name.clone());
        if let Some(e) = pollster::block_on(val_scope.pop()) {
            bail!("GPU setup failed: {e}");
        }
        if let Some(e) = pollster::block_on(oom_scope.pop()) {
            bail!("not enough free GPU memory for this model ({} MB needed; another program such as a local AI model may be using the GPU): {e}", (width * n * 12 + n * 56) / 1_000_000);
        }
        Ok(GpuSystem { device, queue, kernels, bind, bufs, staging_sc, staging_x, n, nwg, dinv, device_name, fault })
    }

    /// Solve A x = b. `x` holds the initial guess (and the prescribed values on Dirichlet rows).
    pub fn solve(&self, a: &NodeSparse, b: &[f64], x: &mut [f64], tol: f64, max_iter: usize, progress: &(dyn Fn(f64) -> Result<()> + Sync)) -> Result<usize> {
        let n = self.n;
        // initial residual on the CPU (one mat-vec)
        let mut r = vec![0.0; n];
        a.matvec(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
        let r0 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r0 == 0.0 {
            return Ok(0);
        }
        let z: Vec<f64> = r.iter().zip(&self.dinv).map(|(a, d)| a * d).collect();
        let rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let q = &self.queue;
        q.write_buffer(&self.bufs[4], 0, bytemuck::cast_slice(x));
        q.write_buffer(&self.bufs[5], 0, bytemuck::cast_slice(&r));
        q.write_buffer(&self.bufs[6], 0, bytemuck::cast_slice(&z));
        q.write_buffer(&self.bufs[7], 0, bytemuck::cast_slice(&z));
        q.write_buffer(&self.bufs[10], 0, bytemuck::cast_slice(&[rz, 0.0, rz, r0 * r0, 0.0, 0.0, 0.0, 0.0]));

        let (log_r0, log_tol) = (1.0f64.ln(), tol.ln());
        let mut it = 0;
        let mut rel = 1.0;
        while it < max_iter {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_bind_group(0, &self.bind, &[]);
                for _ in 0..BATCH {
                    for (k, groups) in [
                        (&self.kernels.spmv, self.nwg),
                        (&self.kernels.dot_pap, self.nwg),
                        (&self.kernels.reduce_pap, 1),
                        (&self.kernels.update_xr, self.nwg),
                        (&self.kernels.dot_rz_rr, self.nwg),
                        (&self.kernels.reduce_rz, 1),
                        (&self.kernels.update_p, self.nwg),
                    ] {
                        pass.set_pipeline(k);
                        pass.dispatch_workgroups(groups, 1, 1);
                    }
                }
            }
            enc.copy_buffer_to_buffer(&self.bufs[10], 0, &self.staging_sc, 0, 64);
            q.submit(Some(enc.finish()));
            let sc = read_f64(&self.device, &self.staging_sc, 8)?;
            if let Some(e) = self.fault.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
                bail!("GPU error during solve: {e}");
            }
            it += BATCH;
            rel = sc[3].max(0.0).sqrt() / r0;
            if !rel.is_finite() {
                bail!("GPU solver broke down (matrix not positive definite?): the model is probably under-constrained");
            }
            progress(((log_r0 - rel.max(1e-300).ln()) / (log_r0 - log_tol)).clamp(0.0, 1.0))?;
            if rel < tol {
                break;
            }
        }
        if rel >= tol * 1e3 {
            bail!("GPU solver did not converge (residual {rel:.2e}); the model may be under-constrained or ill-conditioned");
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.bufs[4], 0, &self.staging_x, 0, (n * 8) as u64);
        q.submit(Some(enc.finish()));
        let out = read_f64(&self.device, &self.staging_x, n)?;
        x.copy_from_slice(&out);
        Ok(it)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_split_prefers_f32_integrated_when_f64_not_required() {
        let cands = [
            (wgpu::DeviceType::Cpu, true),
            (wgpu::DeviceType::IntegratedGpu, false),
            (wgpu::DeviceType::VirtualGpu, false),
        ];
        assert_eq!(select_adapter_index(&cands, true, |x| x.0, |x| x.1), None);
        assert_eq!(select_adapter_index(&cands, false, |x| x.0, |x| x.1), Some(1));

        let dual = [
            (wgpu::DeviceType::IntegratedGpu, true),
            (wgpu::DeviceType::DiscreteGpu, false),
        ];
        assert_eq!(select_adapter_index(&dual, true, |x| x.0, |x| x.1), Some(0));
        assert_eq!(select_adapter_index(&dual, false, |x| x.0, |x| x.1), Some(1));
    }
}
