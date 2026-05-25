# 3DGS Implementation Plan Index — bevy

This directory contains the Bevy-side implementation plans for the 3DGS / HHTL / certified-rendering stack.

## Bevy responsibility

Bevy owns the interactive runtime and viewport shell:

- ECS mirror of active tiles, splat blocks, features, certificates, and decisions.
- Camera, interaction, picking, inspection UI, and debug overlays.
- Runtime display of ndarray CPU-rendered previews through Bevy `Image` assets.
- Optional WGPU / RenderGraph integration after CPU-preview and certificate paths are stable.
- Hot-reload and asset workflow for tiny scenes, fixtures, and local demos.
- Developer-visible debugging for depth, occlusion, HHTL tiers, and certificate failures.

## Markdown convention

Program-related material should use fenced Markdown blocks so Claude Code, GitHub review, and future handovers can parse it cleanly.

Use fences for:

```text
crate/module layouts
commands
Cargo feature sets
Rust DTO sketches
schema sketches
endpoint lists
call-flow diagrams
file paths when shown as groups
```

Use inline code only for short identifiers such as `Bevy`, `Image`, `Component`, or `TileId`.

## Plans

```text
3DGS-Bevy-viewport-runtime-plan.md
```

## Cross-repo boundary

Bevy should not own numerical kernels or durable graph storage.

The intended flow is:

```text
lance-graph
  durable graph, tiles, features, queries, certificates
        ->
bevy
  ECS mirror, camera, picking, UI, runtime viewport
        ->
ndarray
  CPU-SIMD 3DGS / HHTL / render-depth kernels
        ->
bevy
  texture upload, overlays, interaction feedback
```

Central principle: Bevy is where the certified world becomes touchable.
