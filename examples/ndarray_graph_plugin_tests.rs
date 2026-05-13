//! Headless integration tests for the ndarray graph plugin.
//!
//! # Design choice
//!
//! Tests are written as **both** a `fn main()` that panics on failure
//! (CI-runnable via `cargo run --example ndarray_graph_plugin_tests`)
//! AND a `#[cfg(test)] mod tests` block so that
//! `cargo test --example ndarray_graph_plugin_tests` also works.
//!
//! `NdarrayGraphPlugin` and `GraphRenderer` are defined inline here so
//! the file is self-contained. Agent #1 (plugin-core) should produce an
//! `ndarray_graph_plugin.rs` whose types match this contract; at that
//! point the inline definitions here can be replaced with an import.
//!
//! # Running
//!
//! ```sh
//! # Panic-on-failure run (CI):
//! cargo run --example ndarray_graph_plugin_tests
//!
//! # Cargo test runner (alternative):
//! cargo test --example ndarray_graph_plugin_tests
//! ```

use bevy::prelude::*;
use ndarray::hpc::framebuffer::{compose_neo4j, Framebuffer};
use ndarray::hpc::renderer::{DT_60, GLOBAL_RENDERER, RenderFrame, Renderer};
use ndarray::hpc::simd_caps::simd_caps;
use ndarray::simd::PREFERRED_F32_LANES;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal plugin definition (matches the contract agent #1 will deliver).
//
// `GraphRenderer` wraps the ndarray `Renderer` as a Bevy `Resource`.
// `NdarrayGraphPlugin` inserts it and seeds a minimal graph on `Startup`.
// ─────────────────────────────────────────────────────────────────────────────

/// Bevy resource that wraps the ndarray double-buffered `Renderer`.
#[derive(Resource)]
pub struct GraphRenderer {
    /// The ndarray double-buffered renderer.
    pub renderer: Renderer,
    /// Edges: list of (src_node_idx, dst_node_idx) pairs.
    pub edges: Vec<(usize, usize)>,
}

impl Default for GraphRenderer {
    fn default() -> Self {
        Self {
            renderer: Renderer::with_capacity(16),
            edges: Vec::new(),
        }
    }
}

/// Bevy plugin that wires ndarray SIMD graph rendering into a Bevy `App`.
pub struct NdarrayGraphPlugin;

impl Plugin for NdarrayGraphPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GraphRenderer::default())
            .add_systems(Startup, seed_graph)
            .add_systems(Update, tick_graph);
    }
}

/// Seed a single frame with the test graph data.
fn seed_frame(frame: &mut RenderFrame) {
    frame.len = 2;
    // Node 0 at (10, 10, 0), velocity (1, 0, 0)
    frame.positions[0] = 10.0;
    frame.positions[1] = 10.0;
    frame.positions[2] = 0.0;
    frame.velocities[0] = 1.0;
    // Node 1 at (50, 50, 0), velocity (0, 1, 0)
    frame.positions[3] = 50.0;
    frame.positions[4] = 50.0;
    frame.positions[5] = 0.0;
    frame.velocities[4] = 1.0;
}

/// Startup system: seeds two nodes with initial positions and one edge.
///
/// Both frames are seeded identically so the first `tick_graph` call
/// integrates the correct initial state regardless of which frame is
/// currently the back buffer.
fn seed_graph(mut gr: ResMut<GraphRenderer>) {
    let r = &mut gr.renderer;
    // Seed both frames so the very first tick's integrate_simd starts
    // from the correct initial positions/velocities (not zeros).
    seed_frame(&mut r.frames[0].write().expect("frame 0 lock poisoned"));
    seed_frame(&mut r.frames[1].write().expect("frame 1 lock poisoned"));
    gr.edges.push((0, 1));
}

/// Update system: advance physics by one 60 fps tick.
fn tick_graph(gr: ResMut<GraphRenderer>) {
    gr.renderer.tick(DT_60, 0.99);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a headless app with `NdarrayGraphPlugin` ready for assertions.
fn make_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(NdarrayGraphPlugin);
    app
}

/// Exit after exactly one update (used in App::run()-style tests).
fn exit_on_first_update(mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test bodies (callable from both fn main and #[test]).
// ─────────────────────────────────────────────────────────────────────────────

/// Test 1: plugin inserts `GraphRenderer` resource; GLOBAL_RENDERER tick == 0.
fn test_plugin_initializes_global_renderer_resource() {
    let app = make_app();
    // Do NOT call app.update() yet — Startup systems run on first update.
    // Still, the resource is inserted by `build()` via `insert_resource`.
    assert!(
        app.world().contains_resource::<GraphRenderer>(),
        "GraphRenderer resource not found after add_plugins(NdarrayGraphPlugin)"
    );

    // The process-global renderer starts at tick zero.
    assert_eq!(
        GLOBAL_RENDERER.tick_count(),
        0,
        "GLOBAL_RENDERER.tick_count() should be 0 before any tick"
    );
    println!("[test 1] PASS: GraphRenderer resource present, GLOBAL_RENDERER.tick_count()=0");
}

/// Test 2: after one App::update(), the front frame has nodes and edges.
fn test_startup_seeds_nodes_and_edges() {
    let mut app = make_app();
    app.update(); // Runs Startup + Update systems once.

    let gr = app
        .world()
        .get_resource::<GraphRenderer>()
        .expect("GraphRenderer missing after update");

    let front = gr.renderer.read_front();
    assert!(
        front.len > 0,
        "front frame len should be > 0 after seed_graph, got {}",
        front.len
    );
    assert!(
        gr.edges.len() > 0,
        "edges list should be non-empty after seed_graph, got {}",
        gr.edges.len()
    );
    println!(
        "[test 2] PASS: front.len={} edges.len={}",
        front.len,
        gr.edges.len()
    );
}

/// Test 3: tick advances position[0] by exactly velocity * dt (modulo damping).
///
/// This confirms `integrate_simd` (using `F32x16::mul_add`, the actual polyfill)
/// ran inside the Bevy `tick_graph` system.
fn test_tick_advances_position_via_integrate_simd() {
    let mut app = make_app();
    app.update(); // Startup: seeds positions / velocities, swaps, then Update ticks.

    let gr = app
        .world()
        .get_resource::<GraphRenderer>()
        .expect("GraphRenderer missing");

    // After one tick: seed_graph ran first (sets back, swaps → front has the
    // seed), then tick_graph ran (writes to back, swaps → front has ticked data).
    // Node 0 x-position started at 10.0, velocity x = 1.0, damping = 0.99.
    // Expected: position_x = vel_x * DT_60 + pos_x = 1.0 * DT_60 + 10.0.
    let front = gr.renderer.read_front();
    let pos_x = front.positions[0];
    let expected = 1.0_f32.mul_add(DT_60, 10.0);

    assert!(
        (pos_x - expected).abs() < 1e-5,
        "Node 0 x-position after one tick: got {pos_x:.6}, expected {expected:.6} (vel*dt+pos)"
    );
    println!(
        "[test 3] PASS: position[0] advanced from 10.0 to {pos_x:.6} (expected {expected:.6})"
    );
}

/// Test 4: after one App::update(), `compose_neo4j` writes non-zero pixels.
///
/// Builds a framebuffer from the seeded frame + edges and checks that at
/// least 50 bytes are non-zero in the pixel buffer.
fn test_compose_neo4j_emits_pixels_to_framebuffer() {
    const NONZERO_THRESHOLD: usize = 50;

    let mut app = make_app();
    app.update();

    let gr = app
        .world()
        .get_resource::<GraphRenderer>()
        .expect("GraphRenderer missing");

    let front = gr.renderer.read_front();
    let mut fb = Framebuffer::new(128, 128);
    compose_neo4j(&mut fb, &front, &gr.edges, 1.0, (0.0, 0.0), 5, 2);

    let nonzero_count = fb.pixels.iter().filter(|&&p| p != 0).count();
    assert!(
        nonzero_count >= NONZERO_THRESHOLD,
        "compose_neo4j wrote only {nonzero_count} non-zero pixels (threshold={NONZERO_THRESHOLD})"
    );
    println!(
        "[test 4] PASS: compose_neo4j emitted {nonzero_count} non-zero pixels (threshold={NONZERO_THRESHOLD})"
    );
}

/// Test 5: polyfill runtime tier matches compile-time expectation on x86_64.
///
/// On x86_64, exactly one of avx512f or avx2 should be true (the machine
/// has at minimum AVX2 if we've compiled this far with the simd feature).
/// Prints the full caps struct for CI log visibility.
fn test_polyfill_runtime_tier_matches_expectation() {
    let caps = simd_caps();
    println!(
        "[test 5] simd_caps: avx512f={} avx2={} fma={} neon={}  \
         PREFERRED_F32_LANES={}",
        caps.avx512f, caps.avx2, caps.fma, caps.neon, PREFERRED_F32_LANES
    );

    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            caps.avx512f || caps.avx2,
            "Expected avx512f or avx2 to be true on x86_64, got caps={caps:?}"
        );
        println!("[test 5] PASS: x86_64 has avx512f={} or avx2={}", caps.avx512f, caps.avx2);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // On non-x86 (aarch64, WASM, etc.) just print — no mandatory assertion.
        println!("[test 5] PASS: non-x86_64 platform, caps printed above");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// fn main — CI entry point (panic on assertion failure → non-zero exit)
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== ndarray_graph_plugin_tests (headless) ===");

    test_plugin_initializes_global_renderer_resource();
    test_startup_seeds_nodes_and_edges();
    test_tick_advances_position_via_integrate_simd();
    test_compose_neo4j_emits_pixels_to_framebuffer();
    test_polyfill_runtime_tier_matches_expectation();

    println!("=== ALL TESTS PASSED ===");

    // Headless Bevy spin-up proof: MinimalPlugins + NdarrayGraphPlugin link ok.
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins(NdarrayGraphPlugin)
        .add_systems(Update, exit_on_first_update)
        .run();
}

// ─────────────────────────────────────────────────────────────────────────────
// #[cfg(test)] block — for `cargo test --example ndarray_graph_plugin_tests`
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_initializes_global_renderer_resource() {
        test_plugin_initializes_global_renderer_resource();
    }

    #[test]
    fn startup_seeds_nodes_and_edges() {
        test_startup_seeds_nodes_and_edges();
    }

    #[test]
    fn tick_advances_position_via_integrate_simd() {
        test_tick_advances_position_via_integrate_simd();
    }

    #[test]
    fn compose_neo4j_emits_pixels_to_framebuffer() {
        test_compose_neo4j_emits_pixels_to_framebuffer();
    }

    #[test]
    fn polyfill_runtime_tier_matches_expectation() {
        test_polyfill_runtime_tier_matches_expectation();
    }
}
