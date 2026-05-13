//! # NdarrayGraphPlugin — Bevy plugin for SIMD-accelerated graph rendering
//!
//! Visualises a force-directed graph using `ndarray::hpc::renderer::Renderer`
//! (double-buffered, SIMD-integrated) and `ndarray::hpc::framebuffer::Framebuffer`
//! (palette-indexed rasteriser). Each frame:
//!
//! 1. `tick_renderer` — advances physics via `Renderer::tick(dt, 0.98)`.
//! 2. `render_to_framebuffer` — rasterises via `compose_neo4j` into a
//!    long-lived 512×512 `Framebuffer`, expands palette→RGBA8 via the
//!    shared `ndarray_graph_palette::PALETTE_LUT`, and blits into a
//!    long-lived Bevy `Image`.
//!
//! Run headless (no window required for compile checks):
//! ```
//! cargo check --example ndarray_graph_plugin
//! ```

use std::f32::consts::TAU;

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use ndarray::hpc::framebuffer::{compose_neo4j, Framebuffer};
use ndarray::hpc::renderer::{Renderer, DT_60};

// Share the canonical 16-entry RGBA8 palette with the smoke / tests examples.
#[path = "ndarray_graph_palette.rs"]
mod palette;
use palette::{blit_u8_palette_to_rgba, PALETTE_LUT};

// ── Constants ────────────────────────────────────────────────────────────────

/// Side length of the off-screen framebuffer in pixels.
const FB_SIZE: u32 = 512;
/// Number of seed nodes placed in the circle layout on startup.
const NODE_COUNT: usize = 64;
/// Radius of the circle layout in logical units.
const LAYOUT_RADIUS: f32 = 20.0;
/// Node renderer capacity (must be ≥ NODE_COUNT, padded to SIMD lanes).
const RENDERER_CAPACITY: usize = 1024;

/// Palette index used for node dot sprites.
const NODE_COLOR: u8 = 15;
/// Palette index used for Bresenham edge lines.
const EDGE_COLOR: u8 = 8;
/// Scale factor: logical units → framebuffer pixels.
const SCALE: f32 = 8.0;
/// Offset that maps the graph origin to the centre of the 512×512 framebuffer.
const OFFSET: (f32, f32) = (256.0, 256.0);
/// Physics damping applied each tick (≈ 2 % velocity bleed per frame at 60 Hz).
const DAMPING: f32 = 0.98;

// ── Resources ────────────────────────────────────────────────────────────────

/// Bevy `Resource` wrapping the double-buffered SIMD renderer.
///
/// Heap-allocated via `Box` so the `RwLock`-guarded frames don't move.
#[derive(Resource)]
pub struct GraphRenderer {
    renderer: Box<Renderer>,
    /// Flat edge list shared between the seeder and the rasteriser.
    edges: Vec<(usize, usize)>,
}

/// Long-lived per-frame resources so we never allocate inside `Update`.
#[derive(Resource)]
struct RenderSurface {
    /// 512×512 palette-indexed framebuffer (re-cleared each tick by `compose_neo4j`).
    framebuffer: Framebuffer,
    /// Handle to the Bevy `Image` asset we upload palette pixels into.
    image_handle: Handle<Image>,
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Bevy plugin: SIMD-accelerated force-directed graph → Sprite display.
///
/// # Systems
///
/// | Schedule  | System                 | Purpose                        |
/// |-----------|------------------------|--------------------------------|
/// | `Startup` | `seed_graph`           | Place nodes + edges, swap once |
/// | `Update`  | `tick_renderer`        | Physics step (SIMD)            |
/// | `Update`  | `render_to_framebuffer`| Rasterise + blit to GPU Image  |
pub struct NdarrayGraphPlugin;

impl Plugin for NdarrayGraphPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_camera, setup_render_surface, seed_graph).chain())
            .add_systems(
                Update,
                (tick_renderer, render_to_framebuffer).chain(),
            );
    }
}

// ── Startup systems ───────────────────────────────────────────────────────────

/// Spawn a 2-D camera so the sprite is visible.
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Allocate the long-lived `Framebuffer` and the Bevy `Image`, then spawn
/// the `Sprite` that displays it.
fn setup_render_surface(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // Allocate a 512×512 RGBA8 image filled with black (palette index 0).
    let rgba = PALETTE_LUT[0];
    let image = Image::new_fill(
        Extent3d {
            width: FB_SIZE,
            height: FB_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &rgba,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    let image_handle = images.add(image);

    // Spawn the sprite that displays the image.
    commands.spawn(Sprite::from_image(image_handle.clone()));

    commands.insert_resource(RenderSurface {
        framebuffer: Framebuffer::new(FB_SIZE as usize, FB_SIZE as usize),
        image_handle,
    });
}

/// Seed 64 nodes in a circle layout with ~80 random edges, write into the
/// back frame, then swap front↔back so the first tick sees live data.
fn seed_graph(mut commands: Commands) {
    let renderer = Box::new(Renderer::with_capacity(RENDERER_CAPACITY));

    // Write node positions into the back frame.
    {
        let mut back = renderer.write_back();
        back.len = NODE_COUNT;
        for i in 0..NODE_COUNT {
            let angle = TAU * (i as f32) / (NODE_COUNT as f32);
            let x = LAYOUT_RADIUS * angle.cos();
            let y = LAYOUT_RADIUS * angle.sin();
            back.positions[i * 3] = x;
            back.positions[i * 3 + 1] = y;
            back.positions[i * 3 + 2] = 0.0;
            // Small tangential velocity to kick off the simulation.
            back.velocities[i * 3] = -angle.sin() * 0.5;
            back.velocities[i * 3 + 1] = angle.cos() * 0.5;
            // Uniform charge so all nodes repel equally.
            back.charges[i] = 1.0;
        }
    }
    // Swap so the front frame (read by `render_to_framebuffer`) is populated.
    renderer.swap();

    // Build ~80 edges: ring edges + a handful of cross-links.
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(96);
    // Ring edges (64)
    for i in 0..NODE_COUNT {
        edges.push((i, (i + 1) % NODE_COUNT));
    }
    // Cross-links (~16) using a simple deterministic stride pattern.
    for i in 0..16 {
        let a = (i * 4) % NODE_COUNT;
        let b = (i * 4 + NODE_COUNT / 2) % NODE_COUNT;
        if a != b {
            edges.push((a, b));
        }
    }

    commands.insert_resource(GraphRenderer {
        renderer,
        edges,
    });
}

// ── Update systems ────────────────────────────────────────────────────────────

/// Advance the physics simulation by one frame.
///
/// Calls `Renderer::tick(dt, damping)` which: integrates velocities into
/// positions via `F32x16::mul_add` (SIMD), then atomically swaps front/back.
fn tick_renderer(graph: ResMut<GraphRenderer>, time: Res<Time>) {
    // Use the real frame delta but clamp to avoid explosion on first frame.
    let dt = time.delta_secs().clamp(0.001, DT_60 * 4.0);
    graph.renderer.tick(dt, DAMPING);
}

/// Rasterise the current front frame into the `Framebuffer`, expand to RGBA8
/// via the palette LUT, and upload into the Bevy `Image`.
///
/// Neither the `Framebuffer` nor the `Image` buffer is reallocated — only
/// the pixel data is overwritten.
fn render_to_framebuffer(
    graph: Res<GraphRenderer>,
    mut surface: ResMut<RenderSurface>,
    mut images: ResMut<Assets<Image>>,
) {
    // Borrow split: read front frame, then rasterise into surface.framebuffer.
    let front = graph.renderer.read_front();
    compose_neo4j(
        &mut surface.framebuffer,
        &front,
        &graph.edges,
        SCALE,
        OFFSET,
        NODE_COLOR,
        EDGE_COLOR,
    );
    drop(front); // release read-lock before the blit

    // Expand palette u8 → RGBA8 directly into the Bevy image data buffer.
    let Some(mut image) = images.get_mut(&surface.image_handle) else {
        return;
    };
    let Some(data) = image.data.as_mut() else {
        return;
    };

    let pixels = &surface.framebuffer.pixels;
    debug_assert_eq!(
        data.len(),
        pixels.len() * 4,
        "Image data length mismatch: expected {} bytes for {} palette pixels",
        pixels.len() * 4,
        pixels.len()
    );

    // Shared palette expander from `ndarray_graph_palette.rs`. Equivalent to
    // the inline loop but the LUT lives in one place so the smoke test and
    // tests pick up the same colours.
    blit_u8_palette_to_rgba(pixels, data);
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(NdarrayGraphPlugin)
        .run();
}
