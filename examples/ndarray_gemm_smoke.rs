//! Smoke test: ndarray's reverse-engineered, **zero-external-dependency**
//! BLAS Level-3 GEMM reachable from a Bevy downstream crate.
//!
//! Run: `cargo run --release --example ndarray_gemm_smoke --features ndarray-examples`
//!
//! ## What this proves
//!
//! ndarray provides the MKL/OpenBLAS CBLAS GEMM *surface* — `sgemm`,
//! `dgemm`, INT8 `gemm_s8s8s32`, BF16 `gemm_bf16bf16f32` — as **pure-Rust
//! SIMD-dispatched kernels** (AVX-512 / AMX / VNNI → AVX2 → scalar). No
//! `libmkl_rt`, no `libopenblas`, no C toolchain: the `native` backend is
//! the default and needs **no external library** (see
//! `ndarray/src/backend/mod.rs` — "No external C library needed").
//!
//! That is why this example is CI-safe with the *unmodified* ndarray dep
//! (`features = ["rayon"]`, no `intel-mkl`): the GEMM it exercises links
//! nothing outside the Rust dependency graph.
//!
//! ## Asserts
//!   1. `simd_caps()` reports the live CPU tier the kernels dispatch to.
//!   2. f32 `gemm_f32` matches the canonical 2x3 . 3x2 product
//!      `[58, 64, 139, 154]` (the exact case from ndarray's own backend
//!      test) AND a scalar reference.
//!   3. The `cblas_sgemm` drop-in alias is bit-identical to `gemm_f32`.
//!   4. INT8 `gemm_i8` (u8 x i8 -> i32) matches a scalar reference
//!      (AMX `TDPBUSD` / VNNI `VPDPBUSD` / scalar all agree).
//!   5. BF16 `gemm_bf16` (bf16 x bf16 -> f32) matches a scalar reference
//!      (AMX `TDPBF16PS` / AVX-512BF16 / scalar all agree).

use bevy::prelude::*;
use ndarray::backend::{cblas_sgemm, gemm_bf16, gemm_f32, gemm_i8};
use ndarray::hpc::simd_caps::simd_caps;

/// Row-major scalar reference: C(m x n) = A(m x k) . B(k x n), f32.
fn ref_gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Row-major scalar reference: C(m x n) = A(u8) . B(i8) -> i32.
fn ref_gemm_i8(m: usize, n: usize, k: usize, a: &[u8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p] as i32;
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j] as i32;
            }
        }
    }
    c
}

/// Truncate f32 -> bf16 bits (top 16 bits). Exact for the small integers
/// used here, so the bf16 product equals the f32 product bit-for-bit.
fn f32_to_bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Decode bf16 bits -> f32 (the kernel accumulates in f32; the reference
/// must do the same to match).
fn bf16_bits_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Row-major scalar reference for the bf16 -> f32 GEMM.
fn ref_gemm_bf16(m: usize, n: usize, k: usize, a: &[u16], b: &[u16]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = bf16_bits_to_f32(a[i * k + p]);
            for j in 0..n {
                c[i * n + j] += a_ip * bf16_bits_to_f32(b[p * n + j]);
            }
        }
    }
    c
}

fn main() {
    // 1. Which kernel tier will the GEMM dispatch to on this host?
    let caps = simd_caps();
    println!(
        "[gemm] caps: avx512f={} avx512vnni={} avx512bf16={} amx_tile={} avx2={} fma={} neon={}",
        caps.avx512f, caps.avx512vnni, caps.avx512bf16, caps.amx_tile, caps.avx2, caps.fma, caps.neon
    );

    // Shared operands: A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2).
    // Row-major. C = A . B is 2x2.
    let (m, n, k) = (2usize, 2usize, 3usize);

    // 2. f32 GEMM — native pure-Rust Goto-BLAS kernel (matrixmultiply),
    //    no external lib. Anchor against ndarray's own backend test value.
    let a_f32 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_f32 = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c_f32 = vec![0.0f32; m * n];
    // gemm_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) — row-major.
    gemm_f32(m, n, k, 1.0, &a_f32, k, &b_f32, n, 0.0, &mut c_f32, n);
    let anchor = [58.0f32, 64.0, 139.0, 154.0];
    let reference = ref_gemm_f32(m, n, k, &a_f32, &b_f32);
    for i in 0..m * n {
        assert!(
            (c_f32[i] - anchor[i]).abs() < 1e-4,
            "gemm_f32 != anchor at {i}: got {}, want {}",
            c_f32[i],
            anchor[i]
        );
        assert!(
            (c_f32[i] - reference[i]).abs() < 1e-4,
            "gemm_f32 != scalar ref at {i}: got {}, want {}",
            c_f32[i],
            reference[i]
        );
    }
    println!("[gemm] f32  gemm_f32 ok -> {c_f32:?} (anchor [58,64,139,154], no external lib)");

    // 3. cblas_sgemm drop-in alias must be bit-identical to gemm_f32.
    let mut c_cblas = vec![0.0f32; m * n];
    cblas_sgemm(m, n, k, 1.0, &a_f32, k, &b_f32, n, 0.0, &mut c_cblas, n);
    for i in 0..m * n {
        assert_eq!(
            c_cblas[i].to_bits(),
            c_f32[i].to_bits(),
            "cblas_sgemm != gemm_f32 at {i}"
        );
    }
    println!("[gemm] f32  cblas_sgemm drop-in bit-identical to gemm_f32");

    // 4. INT8 GEMM (u8 x i8 -> i32): AMX TDPBUSD / VNNI VPDPBUSD / scalar.
    let a_i8 = [1u8, 2, 3, 4, 5, 6];
    let b_i8 = [7i8, 8, 9, 10, 11, 12];
    let mut c_i8 = vec![0i32; m * n];
    gemm_i8(&a_i8, &b_i8, &mut c_i8, m, n, k);
    let ref_i8 = ref_gemm_i8(m, n, k, &a_i8, &b_i8);
    assert_eq!(c_i8, ref_i8, "gemm_i8 != scalar ref");
    println!("[gemm] int8 gemm_i8 ok -> {c_i8:?} (matches scalar ref)");

    // 5. BF16 GEMM (bf16 x bf16 -> f32): AMX TDPBF16PS / AVX-512BF16 / scalar.
    let a_bf16: Vec<u16> = a_f32.iter().copied().map(f32_to_bf16_bits).collect();
    let b_bf16: Vec<u16> = b_f32.iter().copied().map(f32_to_bf16_bits).collect();
    let mut c_bf16 = vec![0.0f32; m * n];
    gemm_bf16(&a_bf16, &b_bf16, &mut c_bf16, m, n, k);
    let ref_bf16 = ref_gemm_bf16(m, n, k, &a_bf16, &b_bf16);
    for i in 0..m * n {
        assert!(
            (c_bf16[i] - ref_bf16[i]).abs() < 1e-3,
            "gemm_bf16 != scalar ref at {i}: got {}, want {}",
            c_bf16[i],
            ref_bf16[i]
        );
    }
    println!("[gemm] bf16 gemm_bf16 ok -> {c_bf16:?} (matches scalar ref)");

    println!("[gemm] ALL OK — ndarray native GEMM (f32 + int8 + bf16) reachable from bevy, zero external libs");

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
