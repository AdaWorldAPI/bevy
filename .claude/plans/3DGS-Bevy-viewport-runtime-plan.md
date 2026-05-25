# 3DGS Bevy Viewport Runtime Plan

## Goal

Use Bevy as the interactive viewport/runtime shell for the 3DGS / HHTL / certified-rendering stack.

Bevy should make the substrate visible and inspectable:

```text
lance-graph world graph
  -> active tile / feature / certificate set
  -> Bevy ECS mirror
  -> ndarray CPU-SIMD projection / framebuffer / certificate kernels
  -> Bevy Image / overlays / picking / UI
```

## Existing seed

The repository already contains an ndarray integration seed:

```text
examples/ndarray_graph_plugin.rs
examples/ndarray_graph_palette.rs
```

The current pattern is valuable:

```text
ndarray renderer / framebuffer
  -> palette-indexed CPU raster
  -> palette expansion to RGBA
  -> long-lived Bevy Image
  -> sprite display
```

This is the first bridge pattern for 3DGS CPU preview.

## Non-goals

Do not implement 3DGS numerical kernels in Bevy.

Do not implement durable tile/feature graph storage in Bevy.

Do not start with custom RenderGraph complexity before CPU-preview and certificate overlays are stable.

Do not couple Bevy demos directly to ArcGIS, Cesium, Blender, or PR-X12 internals. Use narrow DTOs.

## Phase 1: CPU-preview plugin

Add a small example plugin:

```text
examples/3dgs_viewport_plugin.rs
```

It should:

```text
create a Bevy camera
create a long-lived Bevy Image
create a synthetic splat/tile fixture
call ndarray CPU projection or framebuffer path
upload pixels into the Bevy Image
show a debug overlay with certificate values
```

Initial target is not visual perfection. It is proving the runtime bridge.

## Phase 2: ECS mirror

Define minimal components for active viewport state:

```rust
#[derive(Component)]
pub struct TileEntity {
    pub tile_id: String,
}

#[derive(Component)]
pub struct SplatBlockEntity {
    pub block_id: String,
    pub tile_id: String,
}

#[derive(Component)]
pub struct FeatureEntity {
    pub feature_id: String,
}

#[derive(Component)]
pub struct CertificateBadge {
    pub certificate_id: String,
    pub confidence: f32,
    pub error_px: f32,
    pub passed: bool,
}

#[derive(Component)]
pub struct DepthUncertainty {
    pub min_depth: f32,
    pub max_depth: f32,
    pub ordering_uncertainty: f32,
}
```

These are viewport mirrors only. The durable source of truth remains `lance-graph`.

## Phase 3: HHTL traversal bridge

Add systems that can receive or simulate traversal decisions:

```text
run_hhtl_traversal
update_visible_entities
apply_tile_decisions
request_ndarray_projection
upload_projection_result
```

Decision actions should mirror the lance-graph plan:

```text
skip
keep coarse
refine
load content
project exact
render exact
```

## Phase 4: Certificate overlays

Render certificate state visibly:

```text
green  -> passed / below budget
amber  -> uncertain / needs refinement
red    -> failed / rejected / invalid covariance
blue   -> hidden but query-relevant
purple -> depth-ordering uncertainty
```

Overlays should show:

```text
screen-space error
certified error
confidence
depth interval
occlusion confidence
reason codes
```

## Phase 5: Picking to query focus

Use Bevy interaction as query input:

```text
user clicks entity
  -> TileId / FeatureId / SplatBlockId
  -> query focus request
  -> lance-graph traversal refinement
  -> updated Bevy ECS mirror
```

This enables graph-driven rendering:

```text
selected asset
  -> refine nearby splats
  -> hydrate exact geometry
  -> show provenance and certificates
  -> dim unrelated regions
```

## Phase 6: AssetLoader experiments

Possible Bevy asset types:

```text
TilesetAsset
SplatBlockAsset
LanceSceneIndexAsset
BlenderSceneAsset
Prx12SceneAnchorAsset
CertificateReportAsset
```

Start with synthetic/local files. Avoid network dependency in examples.

## Phase 7: WGPU / RenderGraph path

Only after CPU-preview and certificate overlays work:

```text
ndarray CPU:
  traversal planning
  HHTL prefilter
  projection estimates
  depth certificates

Bevy / WGPU:
  final splat compositing
  mesh overlays
  material visualization
```

This keeps Bevy rendering fast without hiding the certified CPU decisions.

## Suggested module path

Start as examples:

```text
examples/3dgs_viewport_plugin.rs
examples/3dgs_certificate_overlay.rs
examples/3dgs_hhtl_debug_view.rs
```

Later extract to a crate only after the API stabilizes:

```text
crates/bevy_adaworld_viewer/
  lib.rs
  components.rs
  resources.rs
  systems.rs
  overlays.rs
  asset_loaders.rs
```

## Feature gating

Use the existing pattern:

```text
ndarray-examples
```

Do not enable ndarray-heavy examples in upstream-like CI by default.

Potential future feature:

```text
adaworld-3dgs-examples
```

## Acceptance criteria

- A Bevy example can display an ndarray-generated CPU framebuffer.
- A synthetic 3DGS/tile fixture can produce a visible certificate overlay.
- ECS components mirror tile/block/feature/certificate concepts without owning durable storage.
- No heap allocations occur in the per-frame pixel upload path beyond expected Bevy asset mutation.
- The example can run without ArcGIS/Cesium/Blender network dependencies.
- The plan leaves room for WGPU later without requiring it first.

## First demo

```text
one synthetic tile
one synthetic splat block
one camera
ndarray computes:
  projected preview
  depth interval
  certificate
Bevy displays:
  image preview
  bounding box / overlay
  UI text with confidence and reason codes
```

## Wall sentence

```text
Bevy is where the certified world becomes touchable.
```
