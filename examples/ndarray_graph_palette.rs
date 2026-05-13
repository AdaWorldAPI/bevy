//! Palette-index → RGBA conversion helper for the ndarray graph plugin.
//!
//! This module is a standalone library with no Bevy or ndarray dependencies.
//! It is imported by `ndarray_graph_plugin.rs` which uses the
//! `ndarray::simd::PaletteTier::Full16` tier (16-color palette).
//!
//! The [`PALETTE_LUT`] maps each of the 16 `PaletteTier::Full16` palette
//! indices to an RGBA byte quad.  The palette is Neo4j/Palantir-inspired:
//! index 0 is a dark-navy background, indices 1–12 graduate through
//! deep-blue → cyan → white, and indices 13–15 are hot accent colours
//! (amber → hot-orange → crimson-red).
//!
//! # Usage from the plugin
//! ```rust,ignore
//! // In ndarray_graph_plugin.rs:
//! mod ndarray_graph_palette;
//! use ndarray_graph_palette::blit_u8_palette_to_rgba;
//!
//! let mut rgba = vec![0u8; palette_pixels.len() * 4];
//! blit_u8_palette_to_rgba(&palette_pixels, &mut rgba);
//! ```

/// 16-entry RGBA look-up table for the `PaletteTier::Full16` palette.
///
/// Each entry is `[R, G, B, A]` with A always 255 (fully opaque).
///
/// Palette rationale (Neo4j/Palantir graph aesthetic):
/// - Index  0 — dark navy background  (#0D1B2A)
/// - Index  1 — deep navy             (#1A2D45)
/// - Index  2 — cobalt blue           (#1E3A5F)
/// - Index  3 — medium blue           (#1B4F8A)
/// - Index  4 — royal blue            (#1A6BB5)
/// - Index  5 — sky blue              (#2389DA)
/// - Index  6 — steel blue            (#41A9E0)
/// - Index  7 — light cyan            (#6DC8E8)
/// - Index  8 — pale cyan             (#9DE0EF)
/// - Index  9 — ice blue              (#C2EEF7)
/// - Index 10 — near-white blue       (#E0F6FD)
/// - Index 11 — pure white            (#FFFFFF)
/// - Index 12 — pale amber            (#FFE08A)
/// - Index 13 — warm amber            (#FFC33B)
/// - Index 14 — hot orange            (#FF7A00)
/// - Index 15 — crimson accent        (#E8001A)
pub const PALETTE_LUT: [[u8; 4]; 16] = [
    [0x0D, 0x1B, 0x2A, 0xFF], // 0  dark navy background
    [0x1A, 0x2D, 0x45, 0xFF], // 1  deep navy
    [0x1E, 0x3A, 0x5F, 0xFF], // 2  cobalt blue
    [0x1B, 0x4F, 0x8A, 0xFF], // 3  medium blue
    [0x1A, 0x6B, 0xB5, 0xFF], // 4  royal blue
    [0x23, 0x89, 0xDA, 0xFF], // 5  sky blue
    [0x41, 0xA9, 0xE0, 0xFF], // 6  steel blue
    [0x6D, 0xC8, 0xE8, 0xFF], // 7  light cyan
    [0x9D, 0xE0, 0xEF, 0xFF], // 8  pale cyan
    [0xC2, 0xEE, 0xF7, 0xFF], // 9  ice blue
    [0xE0, 0xF6, 0xFD, 0xFF], // 10 near-white blue
    [0xFF, 0xFF, 0xFF, 0xFF], // 11 pure white
    [0xFF, 0xE0, 0x8A, 0xFF], // 12 pale amber
    [0xFF, 0xC3, 0x3B, 0xFF], // 13 warm amber
    [0xFF, 0x7A, 0x00, 0xFF], // 14 hot orange
    [0xE8, 0x00, 0x1A, 0xFF], // 15 crimson accent
];

/// Expand a palette-indexed byte buffer into a 32-bit RGBA buffer.
///
/// Each byte in `palette_pixels` is treated as a 4-bit palette index
/// (bits 3:0; the upper nibble is masked off via `& 0x0F`).  The
/// corresponding [`PALETTE_LUT`] entry is copied into four consecutive
/// bytes of `rgba_out`.
///
/// # Panics
/// Panics (debug) / produces a short write (release) if
/// `rgba_out.len() < palette_pixels.len() * 4`.  The caller is
/// responsible for pre-allocating an output buffer of the correct size.
///
/// # Note on SIMD acceleration
/// The per-byte LUT lookup pattern (`permute_bytes`-style) is directly
/// supported by `crate::simd::U8x64::permute_bytes` in ndarray's SIMD
/// polyfill, which maps to `_mm512_permutexvar_epi8` on VBMI hardware.
/// For Round-2 scope the implementation uses a scalar `for` loop; a
/// vectorised path can be added once `permute_bytes` carries the
/// `#[target_feature(enable = "avx512vbmi")]` gate required by the
/// round-1 fleet review.
///
/// # Examples
/// ```
/// use ndarray_graph_palette::{blit_u8_palette_to_rgba, PALETTE_LUT};
///
/// let palette = [0u8, 15u8, 11u8];
/// let mut rgba = [0u8; 12];
/// blit_u8_palette_to_rgba(&palette, &mut rgba);
/// assert_eq!(&rgba[0..4],  &PALETTE_LUT[0]);   // index 0 → dark navy
/// assert_eq!(&rgba[4..8],  &PALETTE_LUT[15]);  // index 15 → crimson
/// assert_eq!(&rgba[8..12], &PALETTE_LUT[11]);  // index 11 → white
/// ```
#[inline]
pub fn blit_u8_palette_to_rgba(palette_pixels: &[u8], rgba_out: &mut [u8]) {
    debug_assert!(
        rgba_out.len() >= palette_pixels.len() * 4,
        "rgba_out too short: need {} bytes, got {}",
        palette_pixels.len() * 4,
        rgba_out.len(),
    );
    for (i, &p) in palette_pixels.iter().enumerate() {
        rgba_out[i * 4..i * 4 + 4].copy_from_slice(&PALETTE_LUT[p as usize & 0x0F]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a 64-byte palette buffer expands to 256 RGBA bytes,
    /// and that the first and last pixels map to the expected LUT entries.
    #[test]
    fn palette_lut_roundtrip() {
        // Build a 64-byte input: index 0 at position 0, index 15 at position 63,
        // and a ramp through 0-15 in between.
        let mut palette_pixels = [0u8; 64];
        for (i, byte) in palette_pixels.iter_mut().enumerate() {
            *byte = (i & 0x0F) as u8;
        }
        // position 0 → index 0, position 63 → index 15 (63 & 0x0F = 15)

        let mut rgba_out = [0u8; 256]; // 64 * 4
        blit_u8_palette_to_rgba(&palette_pixels, &mut rgba_out);

        // Output length must be exactly 256 bytes.
        assert_eq!(rgba_out.len(), 256);

        // First pixel must match palette index 0 (dark navy background).
        assert_eq!(
            &rgba_out[0..4],
            &PALETTE_LUT[0],
            "pixel 0 should be index 0 (dark navy)"
        );

        // Last pixel must match palette index 15 (crimson accent).
        assert_eq!(
            &rgba_out[252..256],
            &PALETTE_LUT[15],
            "pixel 63 should be index 15 (crimson accent)"
        );

        // Spot-check: position 11 → index 11 (white).
        assert_eq!(
            &rgba_out[11 * 4..11 * 4 + 4],
            &PALETTE_LUT[11],
            "pixel 11 should be index 11 (white)"
        );

        // Alpha channel is always 255 for every entry.
        for chunk in rgba_out.chunks_exact(4) {
            assert_eq!(chunk[3], 0xFF, "alpha must be 255");
        }
    }
}
