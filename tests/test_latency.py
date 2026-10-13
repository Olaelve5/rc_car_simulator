"""Latency and timing benchmark for the cyber-physical vision pipeline.

Measures both end-to-end physical glass-to-detection latency (via automated optical
loopback) and internal microsecond software pipeline processing stages.

Usage:
    python tests/test_latency.py              # Optical latency (camera -> screen)
    python tests/test_latency.py --benchmark  # Software benchmark (no camera)
    python tests/test_latency.py --trials 30  # Run 30 optical flash trials
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import statistics
import sys
import time
from typing import List, Optional
import warnings
import cv2
import numpy as np
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
from vision.homography import HomographyTransformer  # noqa: E402
from vision.marker_tracker import MarkerTracker  # noqa: E402


# Visual palette
COLOR_BG = (24, 24, 27)
COLOR_PANEL = (39, 39, 42)
COLOR_TEXT = (228, 228, 231)
COLOR_MUTED = (161, 161, 170)
COLOR_SUCCESS = (34, 197, 94)
COLOR_ACCENT = (59, 130, 246)
COLOR_BORDER = (82, 82, 91)


def generate_marker_surface(
    aruco_dict: cv2.aruco.Dictionary,
    marker_id: int,
    size_px: int = 240,
) -> pygame.Surface:
    """Generates a high-contrast ArUco marker Pygame surface with a quiet zone."""
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    # Add quiet zone margin (white border)
    margin = int(size_px * 0.15)
    padded = np.ones(
        (size_px + 2 * margin, size_px + 2 * margin), dtype=np.uint8
    ) * 255
    padded[margin : margin + size_px, margin : margin + size_px] = marker_img
    padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    total_w, total_h = padded.shape[1], padded.shape[0]
    return pygame.image.frombuffer(padded_bgr.tobytes(), (total_w, total_h), "BGR")


def run_software_benchmark(num_frames: int = 150) -> None:
    """Benchmarks internal Python software pipeline using synthetic frames."""
    print("\n" + "=" * 65)
    print("SOFTWARE PIPELINE PERFORMANCE BENCHMARK (HEADLESS)")
    print("=" * 65)
    print(f"Synthesizing {num_frames} test frames with realistic markers...")

    arena_cfg = DEFAULT_SETTINGS.arena
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, arena_cfg.aruco_dict_name)
    )
    marker_tracker = MarkerTracker(
        dictionary_id=arena_cfg.aruco_dict_name,
        vehicle_marker_id=arena_cfg.vehicle_marker_id,
    )
    transformer = HomographyTransformer(
        arena_width_m=arena_cfg.arena_width_m,
        arena_height_m=arena_cfg.arena_height_m,
    )

    # Generate synthetic 1280x720 frame with 4 corners and car marker
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 180
    corner_positions = [
        (120, 80),
        (1160, 80),
        (1160, 640),
        (120, 640),
        (640, 360),
    ]
    all_ids = list(arena_cfg.corner_marker_ids) + [arena_cfg.vehicle_marker_id]
    for m_id, (cx, cy) in zip(all_ids, corner_positions):
        m = cv2.aruco.generateImageMarker(aruco_dict, m_id, 80)
        m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        frame[cy - 40 : cy + 40, cx - 40 : cx + 40] = m_bgr

    # Calibration step
    detections = marker_tracker.detect_markers(frame)
    corners = marker_tracker.corner_markers_from_detections(
        detections, arena_cfg.corner_marker_ids
    )
    pixel_pts = np.array([corners[i] for i in arena_cfg.corner_marker_ids])
    transformer.compute_matrix(pixel_pts)

    visualizer = ArenaVisualizer(render_mode="rgb_array")

    t_gray: List[float] = []
    t_detect: List[float] = []
    t_pose: List[float] = []
    t_render: List[float] = []
    t_total: List[float] = []

    for _ in range(num_frames):
        t0 = time.perf_counter()

        # 1. Grayscale conversion
        t_g0 = time.perf_counter()
        _ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_gray.append((time.perf_counter() - t_g0) * 1000.0)

        # 2. ArUco Marker detection
        t_d0 = time.perf_counter()
        dets = marker_tracker.detect_markers(frame)
        t_detect.append((time.perf_counter() - t_d0) * 1000.0)

        # 3. Homography & vehicle pose resolution
        t_p0 = time.perf_counter()
        pose = marker_tracker.vehicle_pose_from_detections(dets, transformer)
        t_pose.append((time.perf_counter() - t_p0) * 1000.0)

        # 4. Arena Visualizer headless rendering
        t_r0 = time.perf_counter()
        if pose:
            visualizer.render(vehicle_pose=(pose.x, pose.y, pose.yaw))
        t_render.append((time.perf_counter() - t_r0) * 1000.0)

        t_total.append((time.perf_counter() - t0) * 1000.0)

    visualizer.close()

    def stats_str(vals: List[float]) -> str:
        return (
            f"Mean: {statistics.mean(vals):5.2f} ms | "
            f"Min: {min(vals):5.2f} ms | "
            f"Max: {max(vals):5.2f} ms"
        )

    print("\n--- RESULTS: INTERNAL STAGE LATENCIES ---")
    print(f"1. Grayscale Conversion:      {stats_str(t_gray)}")
    print(f"2. ArUco Candidate & Decode:  {stats_str(t_detect)}")
    print(f"3. Homography Pose Mapping:   {stats_str(t_pose)}")
    print(f"4. Arena Digital Twin Render: {stats_str(t_render)}")
    print("-" * 65)
    mean_tot = statistics.mean(t_total)
    max_fps = 1000.0 / mean_tot if mean_tot > 0 else 0.0
    print(f"TOTAL SOFTWARE PIPELINE:      {stats_str(t_total)}")
    print(f"MAX THEORETICAL SOFTWARE FPS: {max_fps:.1f} FPS")
    print("=" * 65 + "\n")


def run_optical_latency_test(
    source: str,
    target_trials: int = 20,
    flip_camera: bool = True,
    marker_id: int = 0,
) -> None:
    """Runs automated optical loopback test by flashing an ArUco marker on screen."""
    print("\n" + "=" * 65)
    print("AUTONOMOUS TESTBED: OPTICAL END-TO-END LATENCY BENCHMARK")
    print("=" * 65)
    print(f"Video Source: {source} (Flip 180°: {flip_camera})")
    print(f"Target Trials: {target_trials}")
    print(f"Testing ArUco Marker ID: [{marker_id}]")
    print("Instructions:")
    print("  1. Aim the overhead camera at this monitor window.")
    print("  2. Verify your camera view in the top-right preview box.")
    print("  3. The marker will flash on/off at randomized intervals.")
    print("  4. Press 'q' or 'ESC' to end test early.")
    print("=" * 65 + "\n")

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
        print(f"[ERROR] Failed to start camera stream: {exc}")
        return

    pygame.init()
    window_w, window_h = 960, 640
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("End-to-End Latency Benchmark (Optical Loopback)")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont(["menlo", "consolas", "courier"], 16, bold=True)
    font_body = pygame.font.SysFont(["menlo", "consolas", "courier"], 13)
    font_large = pygame.font.SysFont(["menlo", "consolas", "courier"], 22, bold=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_surf = generate_marker_surface(aruco_dict, marker_id, size_px=220)
    marker_rect = marker_surf.get_rect()
    marker_rect.center = (window_w // 2 - 80, window_h // 2 + 20)

    marker_tracker = MarkerTracker(
        dictionary_id=cv2.aruco.DICT_4X4_50,
        vehicle_marker_id=marker_id,
    )

    # State machine variables
    STATE_IDLE_OFF = 0
    STATE_AWAIT_DETECTION = 1
    STATE_HOLD_SUCCESS = 2

    state = STATE_IDLE_OFF
    next_flash_time = time.monotonic() + 1.5
    flash_start_t: float = 0.0
    detected_latency_ms: Optional[float] = None
    state_hold_until: float = 0.0

    measurements: List[float] = []
    sw_latencies: List[float] = []
    last_frame_id = -1
    is_running = True

    try:
        while is_running and len(measurements) < target_trials:
            now = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        is_running = False
                    elif event.key == pygame.K_SPACE and state == STATE_IDLE_OFF:
                        next_flash_time = now

            # Fetch fresh camera frame
            has_new, frame, frame_id = cam.read_latest(last_frame_id)
            camera_sees_marker = False
            last_corners = None
            last_ids = None

            if has_new and frame is not None:
                last_frame_id = frame_id
                t_det0 = time.perf_counter()
                corners, ids, _ = marker_tracker.detect_markers(frame)
                t_det_ms = (time.perf_counter() - t_det0) * 1000.0
                sw_latencies.append(t_det_ms)
                last_corners, last_ids = corners, ids

                if ids is not None and marker_id in ids.ravel():
                    camera_sees_marker = True

            # State Machine Transitions
            if state == STATE_IDLE_OFF:
                if now >= next_flash_time and not camera_sees_marker:
                    # Trigger Flash ON
                    state = STATE_AWAIT_DETECTION
                    flash_start_t = time.perf_counter()

            elif state == STATE_AWAIT_DETECTION:
                if camera_sees_marker:
                    t_now = time.perf_counter()
                    detected_latency_ms = (t_now - flash_start_t) * 1000.0
                    measurements.append(detected_latency_ms)
                    print(
                        f"Trial {len(measurements):2d}/{target_trials}: "
                        f"Detected in {detected_latency_ms:5.1f} ms"
                    )
                    state = STATE_HOLD_SUCCESS
                    state_hold_until = now + 0.35  # Keep feedback on for 350ms

            elif state == STATE_HOLD_SUCCESS:
                if now >= state_hold_until:
                    # Turn off and wait random interval (0.4s to 0.8s)
                    state = STATE_IDLE_OFF
                    next_flash_time = now + random.uniform(0.4, 0.8)
                    detected_latency_ms = None

            # Render UI
            screen.fill(COLOR_BG)

            # 1. Header Bar
            pygame.draw.rect(screen, (15, 15, 18), (0, 0, window_w, 48))
            pygame.draw.line(screen, COLOR_BORDER, (0, 48), (window_w, 48), 1)
            title = font_title.render(
                "OPTICAL LATENCY BENCHMARK (GLASS-TO-DETECTION)", True, COLOR_TEXT
            )
            screen.blit(title, (16, 14))

            fps_str = (
                f"Vis: {clock.get_fps():.0f} FPS | "
                f"Cam: {cam.measured_fps:.0f} FPS | "
                f"Trial: {len(measurements)}/{target_trials}"
            )
            fps_surf = font_body.render(fps_str, True, COLOR_MUTED)
            screen.blit(fps_surf, (window_w - fps_surf.get_width() - 16, 16))

            # 2. Target Flash Area
            box_rect = pygame.Rect(0, 0, 300, 300)
            box_rect.center = marker_rect.center
            pygame.draw.rect(screen, COLOR_PANEL, box_rect, border_radius=8)

            if state in (STATE_AWAIT_DETECTION, STATE_HOLD_SUCCESS):
                # Flash marker surface
                screen.blit(marker_surf, marker_rect)
                border_color = (
                    COLOR_SUCCESS if state == STATE_HOLD_SUCCESS else COLOR_ACCENT
                )
                pygame.draw.rect(
                    screen, border_color, box_rect, width=3, border_radius=8
                )
            else:
                pygame.draw.rect(
                    screen, COLOR_BORDER, box_rect, width=1, border_radius=8
                )
                idle_txt = font_body.render(
                    "WAITING FOR FLASH...", True, COLOR_MUTED
                )
                screen.blit(
                    idle_txt, idle_txt.get_rect(center=box_rect.center)
                )

            # 3. Live Camera Preview Box (Top Right)
            preview_w, preview_h = 240, 135
            preview_x, preview_y = window_w - preview_w - 20, 65
            pygame.draw.rect(
                screen,
                (15, 23, 42),
                (preview_x - 2, preview_y - 2, preview_w + 4, preview_h + 4),
                border_radius=4,
            )

            if frame is not None:
                preview_frame = frame.copy()
                if last_ids is not None:
                    cv2.aruco.drawDetectedMarkers(
                        preview_frame, last_corners, last_ids
                    )
                thumb_bgr = cv2.resize(
                    preview_frame,
                    (preview_w, preview_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
                thumb_surf = pygame.image.frombuffer(
                    thumb_rgb.tobytes(), (preview_w, preview_h), "RGB"
                )
                screen.blit(thumb_surf, (preview_x, preview_y))
            pygame.draw.rect(
                screen,
                COLOR_BORDER,
                (preview_x - 2, preview_y - 2, preview_w + 4, preview_h + 4),
                width=1,
                border_radius=4,
            )

            preview_label = font_body.render(
                "CAMERA PREVIEW [AIM HERE]", True, COLOR_MUTED
            )
            screen.blit(preview_label, (preview_x, preview_y + preview_h + 6))

            # 4. Telemetry and Results Cards
            card_x = window_w - 320
            card_y = 240
            pygame.draw.rect(
                screen, COLOR_PANEL, (card_x, card_y, 300, 360), border_radius=8
            )
            pygame.draw.rect(
                screen,
                COLOR_BORDER,
                (card_x, card_y, 300, 360),
                width=1,
                border_radius=8,
            )

            t_card = font_title.render("LATENCY METRICS", True, COLOR_TEXT)
            screen.blit(t_card, (card_x + 16, card_y + 16))

            if measurements:
                cur_mean = statistics.mean(measurements)
                cur_min = min(measurements)
                cur_max = max(measurements)
                cur_jitter = (
                    statistics.stdev(measurements) if len(measurements) > 1 else 0.0
                )
                last_val = measurements[-1]

                m_lines = [
                    (f"Latest:       {last_val:5.1f} ms", COLOR_SUCCESS),
                    (f"Average:      {cur_mean:5.1f} ms", COLOR_TEXT),
                    (f"Minimum:      {cur_min:5.1f} ms", COLOR_MUTED),
                    (f"Maximum:      {cur_max:5.1f} ms", COLOR_MUTED),
                    (f"Jitter (Std): ±{cur_jitter:4.1f} ms", COLOR_MUTED),
                ]
            else:
                m_lines = [("Collecting flash trials...", COLOR_MUTED)]

            for idx, (txt, col) in enumerate(m_lines):
                line_surf = font_large.render(
                    txt, True, col
                ) if idx == 0 and measurements else font_body.render(txt, True, col)
                screen.blit(line_surf, (card_x + 16, card_y + 60 + idx * 30))

            if sw_latencies:
                avg_sw = statistics.mean(sw_latencies[-30:])
                sw_surf = font_body.render(
                    f"CV Compute:   {avg_sw:5.2f} ms", True, COLOR_ACCENT
                )
                screen.blit(sw_surf, (card_x + 16, card_y + 240))

            instr = font_body.render(
                "Press 'q' or 'ESC' to end test", True, COLOR_MUTED
            )
            screen.blit(instr, (card_x + 16, card_y + 320))

            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    finally:
        cam.stop()
        pygame.quit()

    # Final Terminal Report
    print("\n" + "=" * 65)
    print("OPTICAL LATENCY BENCHMARK: FINAL REPORT")
    print("=" * 65)
    if not measurements:
        print("No valid optical flash samples collected.")
        print("Tip: Ensure the camera can see the white ArUco marker box on screen.")
        print("=" * 65 + "\n")
        return

    mean_lat = statistics.mean(measurements)
    median_lat = statistics.median(measurements)
    min_lat = min(measurements)
    max_lat = max(measurements)
    jitter = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
    avg_sw = statistics.mean(sw_latencies) if sw_latencies else 2.5
    cam_transfer_est = max(0.0, mean_lat - avg_sw)

    print(f"Trials Completed:           {len(measurements)} of {target_trials}")
    print(f"Measured Camera FPS:        {cam.measured_fps:.1f} FPS")
    print("-" * 65)
    print(f"Average Latency (Mean):     {mean_lat:5.1f} ms")
    print(f"Median Latency:             {median_lat:5.1f} ms")
    print(f"Minimum Latency:            {min_lat:5.1f} ms")
    print(f"Maximum Latency:            {max_lat:5.1f} ms")
    print(f"Jitter (Standard Deviation): ±{jitter:4.1f} ms")
    print("-" * 65)
    print("ESTIMATED PIPELINE DELAY BREAKDOWN:")
    print(f"  • Physical Sensor & USB Ingestion: ~{cam_transfer_est:5.1f} ms")
    print(f"  • Computer Vision Pose Estimation:  {avg_sw:5.1f} ms")
    print("-" * 65)

    if mean_lat <= 40.0:
        verdict = "EXCELLENT (< 40ms) - Ultra-low latency for agile racing."
    elif mean_lat <= 60.0:
        verdict = "GOOD (40-60ms) - Solid performance for 1:28 trajectory tracking."
    else:
        verdict = "HIGH (> 60ms) - Check USB buffer or ambient lighting exposure."
    print(f"VERDICT: {verdict}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark latency of the cyber-physical vision pipeline."
    )
    cam_default_id = DEFAULT_SETTINGS.camera.camera_id
    parser.add_argument(
        "--source",
        default=str(cam_default_id),
        help=f"Camera index or file path (default: {cam_default_id}).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of automated flash trials (default: 20).",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=0,
        help="ArUco marker ID to flash (default: 0).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable 180-degree camera rotation.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run offline internal software benchmark without camera.",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_software_benchmark()
    else:
        run_optical_latency_test(
            source=args.source,
            target_trials=args.trials,
            flip_camera=not args.no_flip,
            marker_id=args.marker_id,
        )


if __name__ == "__main__":
    main()
