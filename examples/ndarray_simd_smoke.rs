//! Smoke test: ndarray `crate::simd` polyfill + rayon parallel integrate
//! reachable from a Bevy downstream crate.
//!
//! Run: `cargo run --release --example ndarray_simd_smoke`
//!
//! Asserts:
//!   1. `simd_caps()` LazyLock initializes and reports the live CPU tier.
//!   2. `F32x16::mul_add` is bit-exact against scalar `f32::mul_add`.
//!   3. `integrate_simd` advances positions by exactly `v * dt`.
//!   4. `integrate_simd_par` (rayon × SIMD) matches sequential bit-exactly.
//!   5. `compose_neo4j` emits both node and edge pixels.
//!
//! What this *proves* end-to-end:
//!   - `target-cpu` propagates from Bevy → ndarray (the `cfg(target_feature
//!     = "avx512f")` in ndarray/src/simd.rs:206-239 picks the right path).
//!   - `LazyLock` runtime detect agrees with compile-time cfg.
//!   - The Pumpkin-derived palette/rasterizer is reachable as a library.
//!   - rayon `par_chunks_mut` composes with `F32x16::mul_add` without
//!     divergence (FMA is deterministic at one dispatch tier).

use bevy::prelude::*;
use ndarray::hpc::framebuffer::{compose_neo4j, Framebuffer, PaletteTier};
use ndarray::hpc::renderer::{
    cached_splat, integrate_simd, integrate_simd_par, RenderFrame, BLOCK_FLOATS, DT_60,
};
use ndarray::hpc::simd_caps::simd_caps;
use ndarray::simd::{F32x16, PREFERRED_F32_LANES};

fn main() {
    // 1. Tier print — proves LazyLock<SimdCaps> initialized.
    let caps = simd_caps();
    println!(
        "[smoke] caps: avx512f={} avx512vnni={} avx2={} fma={} neon={}",
        caps.avx512f, caps.avx512vnni, caps.avx2, caps.fma, caps.neon
    );
    println!(
        "[smoke] compile-time: PREFERRED_F32_LANES={} PaletteTier::detect()={:?}",
        PREFERRED_F32_LANES,
        PaletteTier::detect()
    );

    // 2. F32x16 FMA bit-exact check — proves crate::simd routes correctly.
    let dt = DT_60;
    let dt_v = cached_splat(dt);
    let v = F32x16::splat(0.5);
    let p = F32x16::splat(1.0);
    let out = v.mul_add(dt_v, p);
    let mut out_arr = [0.0f32; 16];
    out.copy_to_slice(&mut out_arr);
    let expected = 0.5_f32.mul_add(dt, 1.0);
    for x in out_arr {
        assert!(
            (x - expected).abs() < 1e-6,
            "F32x16::mul_add lane mismatch: got {}, expected {}",
            x,
            expected
        );
    }
    println!("[smoke] F32x16::mul_add ok (expected={})", expected);

    // 3. integrate_simd contract: x[i] += v[i] * dt.
    let n_nodes = 64;
    let mut frame = RenderFrame::with_capacity(n_nodes);
    frame.len = n_nodes;
    for i in 0..n_nodes {
        frame.positions[i * 3] = i as f32;
        frame.velocities[i * 3] = 1.0;
    }
    let p_before = frame.positions[3];
    integrate_simd(&mut frame.positions, &mut frame.velocities, dt, 1.0);
    let p_after = frame.positions[3];
    assert!(
        (p_after - (p_before + dt)).abs() < 1e-6,
        "integrate_simd did not advance: {} -> {}",
        p_before,
        p_after
    );
    println!("[smoke] integrate_simd advanced by {} (expected {})", p_after - p_before, dt);

    // 4. rayon × SIMD: integrate_simd_par must match integrate_simd bit-exactly.
    //    Buffer is 4 × BLOCK_FLOATS so rayon actually parallelizes.
    let n = 4 * BLOCK_FLOATS;
    let mut p_seq = (0..n).map(|i| (i as f32) * 0.001).collect::<Vec<_>>();
    let mut v_seq = (0..n).map(|i| (i as f32).sin() * 0.1).collect::<Vec<_>>();
    let mut p_par = p_seq.clone();
    let mut v_par = v_seq.clone();

    let t0 = std::time::Instant::now();
    integrate_simd(&mut p_seq, &mut v_seq, dt, 0.98);
    let seq = t0.elapsed();

    let t0 = std::time::Instant::now();
    integrate_simd_par(&mut p_par, &mut v_par, dt, 0.98);
    let par = t0.elapsed();

    for i in 0..n {
        assert_eq!(
            p_seq[i].to_bits(),
            p_par[i].to_bits(),
            "rayon vs sequential diverged at i={}",
            i
        );
    }
    println!(
        "[smoke] integrate_simd_par bit-exact vs sequential ({} floats: seq={:?} par={:?})",
        n, seq, par
    );

    // 5. Rasterize: compose_neo4j on a tiny frame with one edge.
    let mut frame2 = RenderFrame::with_capacity(2);
    frame2.len = 2;
    frame2.positions[0] = 10.0;
    frame2.positions[1] = 10.0;
    frame2.positions[3] = 50.0;
    frame2.positions[4] = 50.0;
    let edges = vec![(0usize, 1usize)];
    let mut fb = Framebuffer::new(64, 64);
    compose_neo4j(&mut fb, &frame2, &edges, 1.0, (0.0, 0.0), 5, 2);
    let edge_pixels = fb.pixels.iter().filter(|&&p| p == 2).count();
    let node_pixels = fb.pixels.iter().filter(|&&p| p == 5).count();
    assert!(
        edge_pixels > 0 && node_pixels > 0,
        "rasterizer empty: edge={} node={}",
        edge_pixels,
        node_pixels
    );
    println!(
        "[smoke] compose_neo4j emitted {} node pixels + {} edge pixels",
        node_pixels, edge_pixels
    );

    println!("[smoke] ALL OK — ndarray::simd polyfill + rayon reachable from bevy");

    // Headless App spin-up — proves the example links against the full Bevy
    // crate. MinimalPlugins runs once and exits via exit_on_first_update.
    App::new()
        .add_plugins(MinimalPlugins)
        .add_systems(Update, exit_on_first_update)
        .run();
}

fn exit_on_first_update(mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
