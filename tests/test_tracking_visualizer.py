"""Integration test: Connects live ArUco marker tracking to 2D ArenaVisualizer.

Streams camera frames via threaded acquisition, resolves 4-point arena homography
calibration and car pose, and renders live vehicle position, heading, and motion
trail in the Pygame arena at a decoupled, smooth 60 FPS.

Usage:
    python tests/test_tracking_visualizer.py             # Uses default camera
    python tests/test_tracking_visualizer.py --no-camera # Pure Pygame (zero Cocoa lag)
    python tests/test_tracking_visualizer.py --res 640x480 # Higher FPS format
    python tests/test_tracking_visualizer.py --mock      # Simulated demo without camera
    python tests/test_tracking_visualizer.py --no-flip   # Disable 180deg camera flip
    python tests/test_tracking_visualizer.py --profile   # Microsecond stage diagnostics
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import warnings
import cv2
import pygame

# Suppress benign pkg_resources warning from pygame internals on newer Python
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Ensure project root is discoverable on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DEFAULT_SETTINGS  # noqa: E402
from simulation.arena_visualizer import ArenaVisualizer  # noqa: E402
from vision.camera_stream import CameraStream  # noqa: E402
from vision.marker_tracker import Detections, VehiclePose  # noqa: E402
from vision.track_vehicle import ArenaTracker, draw_overlay  # noqa: E402


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parses WIDTHxHEIGHT resolution string (e.g. '1280x720' -> (1280, 720))."""
    try:
        parts = res_str.lower().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    raise argparse.ArgumentTypeError(
        f"Invalid resolution '{res_str}'. Expected format: WIDTHxHEIGHT (e.g. 1280x720)"
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
            f"Yaw: {math.degrees(yaw):.1f}° | Vis: {visualizer.fps:.0f} FPS"
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
    resolution: Tuple[int, int] = (1280, 720),
    cam_fps: int = 60,
    show_camera: bool = True,
    show_pip: bool = True,
    flip_camera: bool = True,
    profile: bool = False,
) -> None:
    """Runs decoupled live tracking pipeline connected to 2D ArenaVisualizer."""
    arena_cfg = DEFAULT_SETTINGS.arena
    print("\n" + "=" * 65)
    print("AUTONOMOUS TESTBED: DECOUPLED TRACKING + ARENA VISUALIZER")
    print("=" * 65)
    print(f"Source: {source} (Flip 180°: {flip_camera})")
    print(f"Target Camera Format: {resolution[0]}x{resolution[1]} @ {cam_fps} FPS")
    print(f"Arena Dimensions: {arena_width:.2f}m x {arena_height:.2f}m")
    print(f"Corner Marker IDs: {arena_cfg.corner_marker_ids}")
    print(f"Vehicle Marker ID: {arena_cfg.vehicle_marker_id}")
    print("Controls:")
    print("  'q' / 'ESC' : Quit test")
    print("  'l'         : Lock / unlock homography calibration")
    print("  'c'         : Clear motion breadcrumb trail")
    print("  'v'         : Toggle in-window live camera preview (PiP)")
    if show_camera:
        print("Tip: Pass --no-camera to run solely in Pygame for lowest latency.")
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
        render_fps=60,
        render_mode="human",
    )
    visualizer.show_camera_pip = show_pip

    # Start threaded acquisition to eliminate blocking cap.read()
    cam = CameraStream(
        source=source,
        resolution=resolution,
        fps=cam_fps,
        flip_video=flip_camera,
        buffer_size=1,
    )

    try:
        cam.start()
    except Exception as exc:
        print(f"[ERROR] Failed to initialize camera stream: {exc}")
        visualizer.close()
        return

    last_frame_id = -1
    last_pose: Optional[VehiclePose] = None
    last_detections: Detections = ([], None, [])
    last_overlay: Optional[cv2.typing.MatLike] = None

    last_print = -math.inf
    print_period = 0.5

    # Profiling timers
    t_detect_sum = 0.0
    t_detect_count = 0

    try:
        while visualizer.is_open:
            # 1. Non-blocking retrieval of latest camera frame
            has_new, frame, frame_id = cam.read_latest(last_frame_id)

            if has_new and frame is not None:
                last_frame_id = frame_id
                t0_detect = time.perf_counter()

                # Process detection & homography on fresh frames
                last_pose, last_detections = tracker.update(frame, time.monotonic())

                t_detect = time.perf_counter() - t0_detect
                t_detect_sum += t_detect
                t_detect_count += 1

                # Generate camera overlay when requested for PiP or OpenCV window
                if visualizer.show_camera_pip or show_camera:
                    last_overlay = draw_overlay(
                        frame, tracker, last_pose, last_detections
                    )

                # Optional OpenCV Cocoa window (updated only on new camera frames)
                if show_camera and last_overlay is not None:
                    cv2.imshow("Overhead Camera Feed (OpenCV)", last_overlay)
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

            # 2. Check for Pygame window keyboard shortcuts
            if visualizer.was_key_pressed(pygame.K_l):
                tracker.locked = not tracker.locked
                print(f"Homography {'LOCKED' if tracker.locked else 'UNLOCKED'}")

            # 3. Build telemetry string showing BOTH display FPS and camera FPS
            vis_fps = visualizer.fps
            cam_fps_val = cam.measured_fps
            if last_pose is not None:
                status = (
                    f"TRACKING: ({last_pose.x:.2f}m, {last_pose.y:.2f}m) "
                    f"{math.degrees(last_pose.yaw):.1f}° | "
                    f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps_val:.0f} FPS"
                )
                pose_tuple = (last_pose.x, last_pose.y, last_pose.yaw)
            else:
                if not tracker.transformer.is_calibrated:
                    corners_seen = len(tracker.visible_corners)
                    status = (
                        f"CALIBRATING: {corners_seen}/4 CORNERS | "
                        f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps_val:.0f} FPS"
                    )
                else:
                    status = (
                        f"ARENA CALIBRATED | SEARCHING CAR | "
                        f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps_val:.0f} FPS"
                    )
                pose_tuple = None

            # 4. Render visualizer frame at steady 60 FPS
            visualizer.render(
                vehicle_pose=pose_tuple,
                status_text=status,
                camera_overlay=last_overlay,
            )

            # 5. Console heartbeat and performance diagnostics
            now = time.monotonic()
            if now - last_print >= print_period:
                last_print = now
                if profile and t_detect_count > 0:
                    avg_det_ms = (t_detect_sum / t_detect_count) * 1000.0
                    print(
                        f"[PROFILE] Detect: {avg_det_ms:4.1f}ms | "
                        f"Vis: {vis_fps:4.1f} FPS | "
                        f"Cam: {cam_fps_val:4.1f} FPS"
                    )
                    t_detect_sum = 0.0
                    t_detect_count = 0
                else:
                    if not tracker.transformer.is_calibrated:
                        print(
                            "Waiting for 4 corners... Visible:",
                            list(tracker.visible_corners),
                            f"(Cam: {cam_fps_val:.1f} FPS)",
                        )
                    elif last_pose is None:
                        print(
                            "Arena calibrated. Waiting for vehicle marker... "
                            f"(Cam: {cam_fps_val:.1f} FPS)"
                        )
                    else:
                        print(
                            f"[LOCKED] Vehicle: x={last_pose.x:6.3f}m, "
                            f"y={last_pose.y:6.3f}m, "
                            f"yaw={math.degrees(last_pose.yaw):6.1f}° | "
                            f"Vis: {vis_fps:.0f} FPS | Cam: {cam_fps_val:.0f} FPS"
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
        description="High-framerate ArUco tracking integrated with 2D Arena Visualizer."
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
        "--res",
        type=parse_resolution,
        default=cam.resolution,
        help=(
            f"Capture resolution (e.g. 1280x720, 640x480). "
            f"Default: {cam.resolution[0]}x{cam.resolution[1]}."
        ),
    )
    parser.add_argument(
        "--cam-fps",
        type=int,
        default=cam.fps,
        help=f"Target camera capture FPS requested from driver (default: {cam.fps}).",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Hide separate OpenCV window (render solely in Pygame for maximum FPS).",
    )
    parser.add_argument(
        "--no-pip",
        action="store_true",
        help=(
            "Disable default Picture-in-Picture camera inset inside Pygame "
            "(toggleable with 'v')."
        ),
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable 180-degree camera rotation (flip is on by default).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable live performance profiling and stage latency readouts.",
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
            resolution=args.res,
            cam_fps=args.cam_fps,
            show_camera=not args.no_camera,
            show_pip=not args.no_pip,
            flip_camera=not args.no_flip,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()
