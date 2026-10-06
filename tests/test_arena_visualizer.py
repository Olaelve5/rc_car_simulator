"""Tests for 2D ArenaVisualizer coordinate mapping and rendering."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulation.arena_visualizer import ArenaVisualizer  # noqa: E402


def test_coordinate_mapping() -> None:
    """Verifies forward and inverse world-to-pixel coordinate transforms."""
    print("Testing coordinate transformations...")
    viz = ArenaVisualizer(
        arena_width_m=2.0,
        arena_height_m=1.5,
        pixels_per_meter=500.0,
        margin_px=50,
        top_bar_height_px=40,
        render_mode="rgb_array",
        origin_bottom_left=True,
    )

    test_points = [
        (0.0, 0.0),
        (2.0, 1.5),
        (1.0, 0.75),
        (0.5, 0.25),
        (1.8, 1.2),
    ]

    for wx, wy in test_points:
        px, py = viz.world_to_pixel(wx, wy)
        rx, ry = viz.pixel_to_world(px, py)
        # Due to pixel quantization round-trip error is <= 1 px = 1/500 = 0.002m
        err = max(abs(wx - rx), abs(wy - ry))
        assert err < 0.003, f"Round-trip failed for ({wx}, {wy}): got ({rx}, {ry})"

    # Origin checks with bottom-left convention
    px_00, py_00 = viz.world_to_pixel(0.0, 0.0)
    assert px_00 == 50
    # Top bar (40) + margin (50) + arena_height (750) = 840
    assert py_00 == 50 + 40 + 750

    px_top_right, py_top_right = viz.world_to_pixel(2.0, 1.5)
    # margin (50) + arena_width (1000) = 1050
    assert px_top_right == 50 + 1000
    # Top bar (40) + margin (50) = 90
    assert py_top_right == 50 + 40

    viz.close()
    print("[OK] Coordinate transforms verified.")


def test_rgb_array_render() -> None:
    """Verifies headless offscreen rendering returns valid RGB image array."""
    print("Testing rgb_array rendering...")
    viz = ArenaVisualizer(
        arena_width_m=2.0,
        arena_height_m=1.5,
        pixels_per_meter=300.0,
        margin_px=30,
        top_bar_height_px=30,
        render_mode="rgb_array",
    )

    expected_w = int(2.0 * 300) + 2 * 30  # 660
    expected_h = int(1.5 * 300) + 2 * 30 + 30  # 510

    frame = viz.render()
    assert frame is not None, "Render returned None in rgb_array mode!"
    assert isinstance(frame, np.ndarray), f"Expected np.ndarray, got {type(frame)}"
    assert frame.shape == (
        expected_h,
        expected_w,
        3,
    ), f"Expected shape ({expected_h}, {expected_w}, 3), got {frame.shape}"
    assert frame.dtype == np.uint8, f"Expected dtype uint8, got {frame.dtype}"

    viz.close()
    print(
        f"[OK] rgb_array frame rendered with shape {frame.shape} "
        f"and dtype {frame.dtype}."
    )


def main() -> None:
    print("=" * 60)
    print("Running ArenaVisualizer Phase 1 Verification")
    print("=" * 60)
    test_coordinate_mapping()
    test_rgb_array_render()
    print("=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
