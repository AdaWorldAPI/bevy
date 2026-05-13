# ndarray Graph Plugin for Bevy

## What this is

`ndarray_graph_plugin` is a Bevy example that shows how to wire the
AdaWorldAPI/ndarray SIMD polyfill (`crate::simd::F32x16`, `Framebuffer`,
`compose_neo4j`, `GLOBAL_RENDERER`) directly into a Bevy `App` as a
first-class `Plugin`. Each Bevy `Update` tick advances a 64-node /
80-edge force-directed graph through `ndarray::hpc::renderer`'s
double-buffer integrator, rasterizes the result into a 512x512 palette-indexed
`Framebuffer` using `compose_neo4j`, converts the palette indices to RGBA via a
compile-time LUT, uploads the result as a `bevy::asset::Image`, and displays it
on a `Sprite`. The SIMD path (`F32x16::mul_add`, `U8x64::pairwise_avg`) is
selected at compile time from the `target-cpu` flag and confirmed at runtime
via `simd_caps()`.

---

## Build

### Prerequisites

**Rust toolchain**

```
rustup toolchain install 1.95.0
rustup override set 1.95.0
```

**System libraries** (Debian/Ubuntu)

```
sudo apt-get update -y
sudo apt-get install -y libwayland-dev libasound2-dev libudev-dev
```

**Sibling ndarray checkout**

The Bevy `Cargo.toml` depends on ndarray as a local path dependency
(`../ndarray`). The ndarray tree must be checked out next to the bevy
tree before building:

```
git clone https://github.com/AdaWorldAPI/ndarray.git ../ndarray
```

Both repos must be on matching branches for the feature flags to align.
The CI workflow clones the same-named branch if it exists, falling back
to `master`.

---

## Run

### CI-safe build (x86-64-v3, AVX2 baseline)

This is the default. It works on every GitHub Actions runner. The ndarray
polyfill picks the 8-lane AVX2 path; `PREFERRED_F32_LANES` is 8.

```
cargo run --example ndarray_graph_plugin
```

### AVX-512 build (x86-64-v4, Sapphire Rapids / Ice Lake-SP / Zen 4+)

The `run-avx512` alias is defined in `.cargo/config_ndarray_simd.toml`.
Copy or merge that file into `.cargo/config.toml` before using it.
This build will SIGILL on any host without AVX-512F; do not run it in CI
on stock GitHub Actions runners.

```
cargo run-avx512 --example ndarray_graph_plugin
```

---

## What it shows

On startup the plugin seeds `GLOBAL_RENDERER` with 64 nodes arranged in a
circle and 80 directed edges forming a random sparse graph. Each `Update`
tick:

1. `GLOBAL_RENDERER.tick(dt, damping)` integrates node positions via
   `integrate_simd` — `F32x16::mul_add` fused multiply-add over the
   position/velocity SoA buffers, one AVX-512 (or AVX2) pass per 16
   floats.

2. `compose_neo4j(&mut fb, frame, &edges, scale, offset, node_color, edge_color)`
   rasterizes the front buffer into a 512x512 `Framebuffer`:
   - Edges drawn as Bresenham lines with palette index `edge_color`.
   - Nodes drawn as dot sprites with palette index `node_color`.
   - Pixel values are u8 palette indices (0–15 for AVX-512 tier, 0–7
     for AVX2 tier, 0–3 for NEON/scalar tier).

3. A compile-time RGBA lookup table (`ndarray_graph_palette.rs`) maps
   each palette index to a 4-byte RGBA value. The 512x512 pixel array is
   expanded to a 1048576-byte RGBA buffer suitable for `bevy::asset::Image`.

4. The `Image` is uploaded to the Bevy asset server and bound to a `Sprite`
   component, which Bevy's 2D renderer displays in the window.

The window title shows the current tick count, SIMD tier, and frame time
so the polyfill path is visible at a glance.

---

## Architecture

```
Bevy App
  └── NdarrayGraphPlugin
        ├── Resource<Renderer>        (wraps GLOBAL_RENDERER or a local instance)
        │     └── ndarray::hpc::renderer::GLOBAL_RENDERER
        │           ├── RenderFrame (front)  ← readers here
        │           └── RenderFrame (back)   ← integrate_simd writes here
        │
        ├── System: tick_renderer
        │     calls Renderer::tick(dt, damping)
        │     → F32x16::mul_add via crate::simd polyfill
        │
        ├── System: rasterize_to_framebuffer
        │     calls compose_neo4j(&mut fb, frame, edges, ...)
        │     → Framebuffer { pixels: Vec<u8> }  (palette indices)
        │
        ├── System: palette_blit
        │     expands palette indices → RGBA bytes via LUT
        │     → bevy::asset::Image (Rgba8UnormSrgb, 512×512)
        │
        └── Sprite  ← displays the Image in the 2D world
```

Data flows in one direction: `Renderer` produces a `RenderFrame`, which
`compose_neo4j` reads to fill a `Framebuffer`, which the palette LUT
converts to an `Image`, which Bevy renders. No `&mut self` during any
compute step; all mutation is via the renderer's internal `RwLock`
double-buffer and Bevy's `ResMut`.

---

## Compile-time vs runtime tier

The polyfill exposes two orthogonal tier signals that can disagree:

| Signal | Where | Value on AVX2 build | Value on AVX-512 build |
|--------|-------|---------------------|------------------------|
| `PREFERRED_F32_LANES` | compile-time const (`crate::simd`) | `8` | `16` |
| `simd_caps().avx512f` | runtime CPUID (`LazyLock`) | `true` (if Sapphire Rapids) | `true` |

The smoke test caught exactly this mismatch: building with
`target-cpu=x86-64-v3` (the CI default) on a Sapphire Rapids host
produces `PREFERRED_F32_LANES=8` but `simd_caps().avx512f=true`. The two
signals are not automatically reconciled.

**What controls which path runs:**

- `target-cpu=x86-64-v3` (the default in `.cargo/config.toml`): the
  compiler emits AVX2 code; `cfg(target_feature = "avx512f")` is false
  at compile time; `F32x16::mul_add` compiles to 8-lane AVX2 FMA;
  `PREFERRED_F32_LANES = 8`. The runtime tier reported by `simd_caps()`
  is informational only — no code path switches based on it.

- `target-cpu=x86-64-v4` (via `cargo run-avx512` alias): the compiler
  emits AVX-512 code; `cfg(target_feature = "avx512f")` is true at
  compile time; `F32x16::mul_add` compiles to 16-lane `_mm512_fmadd_ps`;
  `PREFERRED_F32_LANES = 16`. The runtime `simd_caps()` tier now agrees
  with compile time.

The plugin prints both values at startup:

```
[ndarray_graph_plugin] compile-time: PREFERRED_F32_LANES=8
[ndarray_graph_plugin] runtime:      avx512f=true avx2=true
```

A mismatch is not an error — it is expected on Sapphire Rapids with a
CI-safe x86-64-v3 binary — but it means you are leaving AVX-512 throughput
on the table. Pass `-C target-cpu=x86-64-v4` (via the `run-avx512` alias)
to close the gap.

---

## Companion files

The full plugin is split across four files generated by the round-2 CCA2A
fleet:

| File | Agent | Contents |
|------|-------|----------|
| `bevy/examples/ndarray_graph_plugin.rs` | agent #1 plugin-core | `NdarrayGraphPlugin` struct and impl, Bevy systems (`tick_renderer`, `rasterize_to_framebuffer`, `palette_blit`), `Cargo.toml` `[[example]]` entry |
| `bevy/examples/ndarray_graph_palette.rs` | agent #2 plugin-palette | Compile-time RGBA LUT, `palette_to_rgba` expansion function, tier-keyed color definitions for nodes / edges / background |
| `bevy/.github/workflows/ndarray-smoke.yml` | agent #3 plugin-ci | GitHub Actions workflow: clones ndarray sibling, installs system deps, sets Rust 1.95.0, runs `cargo check` on both `ndarray_simd_smoke` and `ndarray_graph_plugin` examples on every push/PR to `claude/**` branches |
| `bevy/examples/README_NDARRAY_PLUGIN.md` | agent #4 plugin-readme | This file |

The existing smoke test at `bevy/examples/ndarray_simd_smoke.rs` remains
the canonical end-to-end correctness check. The graph plugin builds on the
same ndarray API surface that the smoke test exercises; see the smoke test's
assertion 5 (`compose_neo4j`) and assertions 3–4 (`integrate_simd`,
`integrate_simd_par`) for the tested contracts.

---

## Known limitations

- `integrate_simd_par` (rayon) is deliberately not used in the per-frame
  tick at 64 nodes. The documented crossover is 65536 floats; at 64 nodes
  (192 floats) rayon overhead dominates. Use `integrate_simd` for scenes
  under ~5000 nodes and switch to `integrate_simd_par` only when profiling
  confirms the crossover is reached.

- `PaletteTier::detect()` currently proxies off `PREFERRED_F32_LANES` (a
  f32 lane count) to select u8 palette depth. On an AVX2 build
  (`PREFERRED_F32_LANES=8`) the framebuffer uses `Mid8` (8 colors) even
  though AVX2 has 32 u8 lanes. This is a known issue in `framebuffer.rs`;
  the plugin uses whichever tier `PaletteTier::detect()` returns.

- The `GLOBAL_RENDERER` singleton is initialized once per process at 4096
  node capacity. It cannot be resized at runtime. For larger scenes,
  construct a local `Renderer::with_capacity(n)` and store it as a Bevy
  `Resource` instead of using `GLOBAL_RENDERER`.
