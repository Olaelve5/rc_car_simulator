"""Integration test: Connects live ArUco marker tracking to 2D ArenaVisualizer.

Streams camera frames, resolves 4-point arena homography calibration and car pose,
and renders the live vehicle position, heading, and motion trail in the Pygame arena.

Usage:
    python tests/test_tracking_visualizer.py             # Uses default camera
    python tests/test_tracking_visualizer.py --no-camera # Hide OpenCV window
    python tests/test_tracking_visualizer.py --mock      # Simulated demo without camera
    python tests/test_tracking_visualizer.py --no-flip   # Disable 180deg camera flip
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
import warnings
import cv2

# Suppress benign pkg_resources warning from pygame internals on newer Python
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Ensure project root is discoverable on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DEFAULT_SETTINGS  # noqa: E402
from simulation.arena_visualizer import ArenaVisualizer  # noqa: E402
from vision.camera_stream import CameraStream  # noqa: E402
from vision.track_vehicle import (  # noqa: E402
    ArenaTracker,
    draw_overlay,
)


def run_mock_simulation(visualizer: ArenaVisualizer) -> None:
    """Runs a simulated circular vehicle trajectory to verify visualizer movement."""
    print("\n--- Running in MOCK Mode (Demo Trajectory) ---")
    print("Press 'ESC' or 'Q' in the visualizer window to exit.\n")

    center_x = visualizer.arena_width_m / 2.0
    center_y = visualizer.arena_height_m / 2.0
    radius = min(visualizer.arena_width_m, visualizer.arena_height_m) * 0.32
    omega = 0.8  # angular velocity (rad/s)
    t0 = time.time()

    while visualizer.is_open:
        t = time.time() - t0
        # Circular orbit
        x = center_x + radius * math.cos(omega * t)
        y = center_y + radius * math.sin(omega * t)
        yaw = omega * t + math.pi / 2.0  # Tangent heading
        # Normalize yaw to [-pi, pi]
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        steer = 0.25 * math.sin(omega * t)

        status = (
            f"MOCK DEMO | Pos: ({x:.2f}m, {y:.2f}m) | "
            f"Yaw: {math.degrees(yaw):.1f}°"
        )
        visualizer.render(
            vehicle_pose=(x, y, yaw),
            steering_angle=steer,
            status_text=status,
        )


def run_live_tracking(
    source: str,
    arena_width: float,
    arena_height: float,
    show_camera: bool = True,
    flip_camera: bool = True,
) -> None:
    """Runs live ArUco detection pipeline connected to 2D ArenaVisualizer."""
    arena_cfg = DEFAULT_SETTINGS.arena
    print("\n" + "=" * 65)
    print("AUTONOMOUS TESTBED: LIVE TRACKING + ARENA VISUALIZER")
    print("=" * 65)
    print(f"Source: {source} (Flip 180°: {flip_camera})")
    print(f"Arena Dimensions: {arena_width:.2f}m x {arena_height:.2f}m")
    print(f"Corner Marker IDs: {arena_cfg.corner_marker_ids}")
    print(f"Vehicle Marker ID: {arena_cfg.vehicle_marker_id}")
    print("Controls:")
    print("  'q' / 'ESC' : Quit test")
    print("  'l'         : Lock / unlock homography calibration (in camera window)")
    print("  'c'         : Clear motion breadcrumb trail (in visualizer window)")
    print("=" * 65 + "\n")

    # Initialize tracker and visualizer
    tracker = ArenaTracker(
        arena_width_m=arena_width,
        arena_height_m=arena_height,
        corner_ids=arena_cfg.corner_marker_ids,
        vehicle_marker_id=arena_cfg.vehicle_marker_id,
        dictionary=arena_cfg.aruco_dict_name,
        flip_camera=flip_camera,
    )

    visualizer = ArenaVisualizer(
        arena_width_m=arena_width,
        arena_height_m=arena_height,
        render_mode="human",
    )

    cam_cfg = DEFAULT_SETTINGS.camera
    cam = CameraStream(
        source=source,
        resolution=cam_cfg.resolution,
        fps=cam_cfg.fps,
        flip_video=flip_camera,
        buffer_size=1,
    )

    try:
        cam.start()
    except Exception as exc:
        print(f"[ERROR] Failed to open camera stream: {exc}")
        visualizer.close()
        return

    last_print = -math.inf
    print_period = 0.5  # print every 500ms
    last_frame_id = -1
    last_pose = None
    last_detections = ([], None, [])

    try:
        while visualizer.is_open:
            # Non-blocking retrieval of newest camera frame
            has_new, frame, frame_id = cam.read_latest(last_frame_id)

            if has_new and frame is not None:
                last_frame_id = frame_id
                # Run detection & pose calculation only on fresh frames
                last_pose, last_detections = tracker.update(frame, time.monotonic())

                # Optional OpenCV camera feed preview (updated on new frames)
                if show_camera:
                    camera_overlay = draw_overlay(
                        frame, tracker, last_pose, last_detections
                    )
                    cv2.imshow("Overhead Camera Feed (OpenCV)", camera_overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("l"):
                        tracker.locked = not tracker.locked
                        print(
                            f"Homography {'LOCKED' if tracker.locked else 'UNLOCKED'}"
                        )
                    elif key == ord("c"):
                        visualizer.clear_trail()

            # Format status text displaying both Visualizer FPS and Camera FPS
            vis_fps = visualizer.fps
            cam_fps = cam.measured_fps
            if last_pose is not None:
                status = (
                    f"TRACKING: ({last_pose.x:.2f}m, {last_pose.y:.2f}m) "
                    f"{math.degrees(last_pose.yaw):.1f}° | "
                    f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps:.0f} FPS"
                )
                visualizer.render(
                    vehicle_pose=(last_pose.x, last_pose.y, last_pose.yaw),
                    status_text=status,
                )
            else:
                if not tracker.transformer.is_calibrated:
                    corners_seen = len(tracker.visible_corners)
                    status = (
                        f"CALIBRATING: {corners_seen}/4 CORNERS | "
                        f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps:.0f} FPS"
                    )
                else:
                    status = (
                        f"ARENA CALIBRATED | SEARCHING CAR | "
                        f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps:.0f} FPS"
                    )
                visualizer.render(status_text=status)

            # Console heartbeat
            now = time.monotonic()
            if now - last_print >= print_period:
                last_print = now
                if not tracker.transformer.is_calibrated:
                    print(
                        "Waiting for 4 corners... Visible:",
                        list(tracker.visible_corners),
                        f"(Cam: {cam_fps:.1f} FPS)",
                    )
                elif last_pose is None:
                    print(
                        "Arena calibrated. Waiting for vehicle marker... "
                        f"(Cam: {cam_fps:.1f} FPS)"
                    )
                else:
                    print(
                        f"[LOCKED] Vehicle: x={last_pose.x:6.3f}m, "
                        f"y={last_pose.y:6.3f}m, "
                        f"yaw={math.degrees(last_pose.yaw):6.1f}° | "
                        f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps:.0f} FPS"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cam.stop()
        if show_camera:
            cv2.destroyAllWindows()
        visualizer.close()
        print("Visualizer and camera closed cleanly.")


def main() -> None:
    arena = DEFAULT_SETTINGS.arena
    cam = DEFAULT_SETTINGS.camera

    parser = argparse.ArgumentParser(
        description="Live ArUco tracking integrated with 2D Arena Visualizer."
    )
    parser.add_argument(
        "--source",
        default=str(cam.camera_id),
        help=(
            f"Camera device index (e.g. 0) or video file path "
            f"(default: {cam.camera_id})."
        ),
    )

    parser.add_argument(
        "--width",
        type=float,
        default=arena.arena_width_m,
        help=f"Arena width in meters (default: {arena.arena_width_m}).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=arena.arena_height_m,
        help=f"Arena height in meters (default: {arena.arena_height_m}).",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Hide OpenCV camera preview window (only display Pygame visualizer).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable 180-degree camera rotation (flip is on by default).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run animated mock trajectory to test visualizer without camera.",
    )
    args = parser.parse_args()

    if args.mock:
        with ArenaVisualizer(
            arena_width_m=args.width,
            arena_height_m=args.height,
            render_mode="human",
        ) as visualizer:
            run_mock_simulation(visualizer)
    else:
        run_live_tracking(
            source=args.source,
            arena_width=args.width,
            arena_height=args.height,
            show_camera=not args.no_camera,
            flip_camera=not args.no_flip,
        )


if __name__ == "__main__":
    main()
