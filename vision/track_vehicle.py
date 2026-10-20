"""Live arena tracking: homography from 4 corner ArUco markers + car marker pose.

Corner markers define the arena frame by their centers, in the order given by
``ArenaConfig.corner_marker_ids``:

    id[0] -> (0, 0)      id[1] -> (W, 0)      id[2] -> (W, H)      id[3] -> (0, H)

Place them counter-clockwise when viewed from above, otherwise the frame is
mirrored and yaw changes sign. W and H are the distances between marker
centers (``ArenaConfig.arena_width_m`` / ``arena_height_m``).

Usage:
    python -m vision.track_vehicle                  # camera from config
    python -m vision.track_vehicle --source 1       # camera index 1
    python -m vision.track_vehicle --source run.mp4 # recorded video
    python -m vision.track_vehicle --no-display     # print only

Keys (display mode): q = quit, l = lock/unlock the homography.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterator, Optional, Tuple
import cv2
import numpy as np

from config.settings import DEFAULT_SETTINGS
from .arena_tracker import ArenaTracker
from .marker_tracker import Detections, VehiclePose


def open_source(source: str) -> cv2.VideoCapture:
    """Opens a camera index (e.g. "0") or a video file path."""
    if source.isdigit():
        cam = DEFAULT_SETTINGS.camera
        cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, cam.fps)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    return cap


def iter_poses(
    cap: cv2.VideoCapture, tracker: ArenaTracker
) -> Iterator[Tuple[np.ndarray, Optional[VehiclePose], Detections]]:
    """Yields (frame, pose, detections) for each frame until the stream ends."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        if tracker.flip_camera:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        pose, detections = tracker.update(frame, time.monotonic())
        yield frame, pose, detections


def draw_overlay(
    frame: np.ndarray,
    tracker: ArenaTracker,
    pose: Optional[VehiclePose],
    detections: Detections,
) -> np.ndarray:
    """Draws detected markers, arena outline and car heading on the raw frame."""
    out = frame.copy()
    corners, ids, _ = detections
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

    t = tracker.transformer
    if t.is_calibrated:
        outline = np.array(
            [t.world_to_pixel(tuple(p)) for p in t.default_world_corners()],
            dtype=np.int32,
        )
        color = (0, 165, 255) if tracker.locked else (0, 255, 0)
        cv2.polylines(out, [outline], True, color, 2)

    if pose is not None:
        tip = (pose.x + 0.1 * math.cos(pose.yaw), pose.y + 0.1 * math.sin(pose.yaw))
        cv2.arrowedLine(
            out,
            t.world_to_pixel((pose.x, pose.y)),
            t.world_to_pixel(tip),
            (0, 0, 255),
            3,
            tipLength=0.3,
        )
        text = f"x={pose.x:.3f} y={pose.y:.3f} yaw={math.degrees(pose.yaw):.1f}deg"
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def main() -> None:
    arena = DEFAULT_SETTINGS.arena
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SETTINGS.camera.camera_id),
        help="Camera index or video file path.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=arena.arena_width_m,
        help="Distance between corner marker centers along X (m).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=arena.arena_height_m,
        help="Distance between corner marker centers along Y (m).",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=10.0,
        help="Max console print rate; 0 prints every frame.",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable OpenCV preview windows."
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable 180-degree camera rotation (flip is on by default).",
    )
    args = parser.parse_args()

    tracker = ArenaTracker.from_config(
        arena, not args.no_flip, args.width, args.height
    )
    cap = open_source(args.source)
    print_period = 1.0 / args.print_hz if args.print_hz > 0 else 0.0
    last_print = -math.inf

    print(
        f"Tracking car marker {arena.vehicle_marker_id}, "
        f"corners {arena.corner_marker_ids}, "
        f"arena {args.width} x {args.height} m. Ctrl+C to stop."
    )
    try:
        for frame, pose, detections in iter_poses(cap, tracker):
            now = time.monotonic()
            if now - last_print >= print_period:
                last_print = now
                if not tracker.transformer.is_calibrated:
                    print(
                        "waiting for all 4 corners, visible:",
                        list(tracker.visible_corners),
                    )
                elif pose is None:
                    print("car marker not visible")
                else:
                    print(
                        f"x={pose.x:7.3f} m  y={pose.y:7.3f} m  "
                        f"yaw={math.degrees(pose.yaw):7.1f} deg"
                    )

            if args.no_display:
                continue
            cv2.imshow("camera", draw_overlay(frame, tracker, pose, detections))
            if tracker.transformer.is_calibrated:
                cv2.imshow("arena (bird's-eye)", tracker.transformer.warp_image(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                tracker.locked = not tracker.locked
                print(f"homography {'locked' if tracker.locked else 'unlocked'}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
