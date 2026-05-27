use cudarc::driver::CudaContext;
use luminal::prelude::*;
use tracing::{Level, enabled};

use crate::{cuda_bandwidth_gbps, cuda_compute_f32_tflops};
use crate::runtime::CudaRuntime;

/// Test that measures bandwidth utilization for a large element-wise add kernel.
/// This demonstrates that KernelAdd can achieve reasonable bandwidth with large tensors.
#[test]
pub fn kernel_add_bandwidth_test() {
    // 64M elements = 256MB per tensor, 768MB total memory traffic (2 reads + 1 write)
    let size = 64 * 1024 * 1024;

    let mut cx = Graph::default();
    let a = cx.tensor(size).persist();
    let b = cx.tensor(size).persist();
    let output = (a + b).output();

    // Generate test data
    let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 * 0.001).collect();
    let data_b: Vec<f32> = (0..size)
        .map(|i| ((i + 500) % 1000) as f32 * 0.001)
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    rt = cx.search(rt, 5);

    // Warm up
    rt.execute(&cx.dyn_map);

    // Run and measure
    rt.execute(&cx.dyn_map);

    // Print stats
    println!("\n=== Large KernelAdd Bandwidth Test ===");
    println!(
        "Tensor size: {} elements ({} MB per tensor)",
        size,
        size * 4 / 1024 / 1024
    );
    println!(
        "Total memory traffic: {} MB (2 reads + 1 write)",
        size * 4 * 3 / 1024 / 1024
    );
    if enabled!(Level::INFO) {
        rt.print_execution_stats();
    }

    // Verify correctness (spot check)
    let result = rt.get_f32(output);
    for i in [0, size / 2, size - 1] {
        let expected = data_a[i] + data_b[i];
        let got = result[i];
        assert!(
            (got - expected).abs() < 1e-5,
            "Mismatch at {}: expected {}, got {}",
            i,
            expected,
            got
        );
    }

    // Check bandwidth is reasonable (at least 50% of peak for large kernels)
    if let Some(peak_bw) = cuda_bandwidth_gbps(&ctx) {
        for stat in &rt.last_kernel_stats {
            let total_bytes = stat.bytes_loaded + stat.bytes_stored;
            if stat.name == "Add" && total_bytes > 0 {
                let utilization = stat.bandwidth_gbps / peak_bw as f64 * 100.0;
                println!(
                    "\nAdd kernel achieved {:.1} GB/s ({:.1}% of {:.0} GB/s peak)",
                    stat.bandwidth_gbps, utilization, peak_bw
                );
                println!(
                    "  Loaded: {} bytes, Stored: {} bytes",
                    stat.bytes_loaded, stat.bytes_stored
                );
                // Large adds should achieve decent bandwidth
                assert!(
                    utilization > 50.0,
                    "Bandwidth utilization too low: {:.1}%",
                    utilization
                );
            }
        }
    }
}

/// Benchmark KernelBatchMatMul across shapes representative of attention and FFN layers.
///
/// Run with:
///   cargo test -p luminal_cuda_lite batch_matmul_perf -- --nocapture
#[test]
pub fn batch_matmul_perf() {
    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let peak_tf = cuda_compute_f32_tflops(&ctx);

    // (label, batch, M, K, N)
    // attention QK^T: [batch*heads, seq, head_dim] x [batch*heads, head_dim, seq]
    // FFN up-proj: [batch, seq, d_model] x [d_model, d_ff]
    let shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("attn  32h seq=128  d=64  ", 32, 128,  64,  128),
        ("attn  32h seq=512  d=64  ", 32, 512,  64,  512),
        ("attn  32h seq=2048 d=64  ", 32, 2048, 64,  2048),
        ("attn  32h seq=512  d=128 ", 32, 512,  128, 512),
        ("ffn   bs=1 seq=512 k=4096", 1,  512,  4096, 16384),
    ];

    println!("\n{:<32} {:>8} {:>10} {:>10}", "shape", "TFLOPS", "% peak", "ms");
    println!("{}", "-".repeat(64));

    for &(label, batch, m, k, n) in shapes {
        // Build a batched matmul: A [batch, m, k] x B [batch, k, n] -> [batch, m, n]
        // Achieved via expand + matmul so the egglog rewrite picks KernelBatchMatMul
        // (non-contiguous strides from the expand trigger the custom kernel path)
        let mut cx = Graph::default();
        let a = cx.tensor((batch, m, k)).persist();
        let b = cx.tensor((batch, k, n)).persist();
        let _out = a.matmul(b).output();

        cx.build_search_space::<CudaRuntime>();
        let mut rt = CudaRuntime::initialize(stream.clone());

        let a_data: Vec<f32> = (0..batch * m * k).map(|i| (i % 100) as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..batch * k * n).map(|i| (i % 100) as f32 * 0.01).collect();
        rt.set_data(a, a_data);
        rt.set_data(b, b_data);

        let mut rt = cx.search(rt, 5);

        // Warm-up
        for _ in 0..3 {
            rt.execute(&cx.dyn_map);
        }

        // Timed runs
        const ITERS: usize = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            rt.execute(&cx.dyn_map);
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

        let flops = 2 * batch * m * k * n;
        let tflops = flops as f64 / (elapsed_ms * 1e-3) / 1e12;
        let pct = peak_tf
            .map(|p| format!("{:>9.1}%", tflops / p as f64 * 100.0))
            .unwrap_or_else(|| "        N/A".to_string());

        println!("{:<32} {:>8.3} {:>10} {:>10.3}", label, tflops, pct, elapsed_ms);

        // Print per-kernel breakdown if any BatchMatMul kernels fired
        for stat in &rt.last_kernel_stats {
            if stat.name.contains("BatchMat") || stat.name.contains("CuBlas") {
                println!("  └─ {:<20} {:.3} TFLOPS  {:.1} GB/s",
                    stat.name, stat.tflops, stat.bandwidth_gbps);
            }
        }
    }
}
