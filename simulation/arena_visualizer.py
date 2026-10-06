"""2D top-down arena visualizer and digital twin display for 1:28 scale testbed."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import warnings
import numpy as np
import pygame

# Suppress benign pkg_resources warning emitted by pygame internals on newer Python
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Ensure repository root is on sys.path when executed directly as a script
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.settings import DEFAULT_SETTINGS, ArenaConfig  # noqa: E402


class ArenaVisualizer:
    """Renders a metric 2D top-down view of the physical arena using Pygame.

    Translates physical world coordinates (meters) to screen pixel coordinates,
    providing visual representations of arena walls, calibration corner markers,
    metric grid lines, coordinate frames, and vehicle states.

    Attributes:
        arena_width_m: Width of arena along X axis in meters.
        arena_height_m: Height of arena along Y axis in meters.
        pixels_per_meter: Scale factor mapping meters to screen pixels.
        render_mode: Either 'human' (interactive display) or 'rgb_array'.
        render_fps: Target frame rate for display refresh.
        origin_bottom_left: If True, (0, 0) is bottom-left (standard Cartesian);
                            if False, (0, 0) is top-left (screen/image origin).
    """

    # Visual Theme Palette (Modern Dark Mode)
    COLOR_BG = (24, 24, 27)  # Zinc 900
    COLOR_ARENA_FLOOR = (39, 39, 42)  # Zinc 800
    COLOR_ARENA_BORDER = (248, 250, 252)  # Slate 50
    COLOR_GRID_MAJOR = (82, 82, 91)  # Zinc 600
    COLOR_GRID_MINOR = (63, 63, 70)  # Zinc 700
    COLOR_TEXT = (161, 161, 170)  # Zinc 400
    COLOR_HEADER_BG = (15, 15, 18)
    COLOR_HEADER_TEXT = (228, 228, 231)  # Zinc 200
    COLOR_CORNER_MARKER = (245, 158, 11)  # Amber 500

    # Vehicle Colors (1:28 scale chassis)
    COLOR_VEHICLE_BODY = (59, 130, 246)  # Blue 500
    COLOR_VEHICLE_CABIN = (30, 58, 138)  # Blue 900
    COLOR_VEHICLE_NOSE = (239, 68, 68)  # Red 500 (front bumper)
    COLOR_VEHICLE_BORDER = (241, 245, 249)  # Slate 100
    COLOR_VEHICLE_WHEEL = (15, 23, 42)  # Slate 900
    COLOR_VEHICLE_WHEEL_BORDER = (148, 163, 184)  # Slate 400
    COLOR_HEADING_LINE = (56, 189, 248)  # Sky 400

    def __init__(
        self,
        arena_config: Optional[ArenaConfig] = None,
        arena_width_m: Optional[float] = None,
        arena_height_m: Optional[float] = None,
        pixels_per_meter: float = 400.0,
        margin_px: int = 75,
        top_bar_height_px: int = 40,
        render_fps: int = 60,
        render_mode: str = "human",
        origin_bottom_left: bool = True,
        window_title: str = "Arena Visualizer",
    ) -> None:
        """Initializes geometry, screen scaling, and Pygame surface context."""
        cfg = arena_config or DEFAULT_SETTINGS.arena
        self.arena_width_m = (
            arena_width_m if arena_width_m is not None else cfg.arena_width_m
        )
        self.arena_height_m = (
            arena_height_m if arena_height_m is not None else cfg.arena_height_m
        )
        self.corner_marker_ids = cfg.corner_marker_ids
        self.vehicle_marker_id = cfg.vehicle_marker_id
        self.vehicle_marker_size_m = cfg.vehicle_marker_size_m

        # 1:28 Vehicle Physical Dimensions (meters)
        self.vehicle_length_m = 0.160
        self.vehicle_width_m = 0.075
        self.vehicle_wheelbase_m = 0.098
        self.vehicle_track_m = 0.065

        self.pixels_per_meter = float(pixels_per_meter)
        self.margin_px = int(margin_px)
        self.top_bar_height_px = int(top_bar_height_px)
        self.render_fps = int(render_fps)
        self.render_mode = render_mode
        self.origin_bottom_left = origin_bottom_left
        self.window_title = window_title

        # Compute arena display dimensions in pixels
        self.arena_width_px = int(round(self.arena_width_m * self.pixels_per_meter))
        self.arena_height_px = int(round(self.arena_height_m * self.pixels_per_meter))

        # Total window size including padding margins and top header bar
        self.window_width = self.arena_width_px + 2 * self.margin_px
        self.window_height = (
            self.arena_height_px + 2 * self.margin_px + self.top_bar_height_px
        )

        # Arena canvas bounding box on screen
        self.arena_rect_left = self.margin_px
        self.arena_rect_top = self.margin_px + self.top_bar_height_px

        self.is_open: bool = True
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._font_medium: Optional[pygame.font.Font] = None

        self._init_pygame()

    def _init_pygame(self) -> None:
        """Initializes Pygame display, fonts, and frame clock."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # Set up offscreen headless driver if in rgb_array mode with no existing display
        if self.render_mode == "rgb_array" and not pygame.display.get_init():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        if not pygame.display.get_init():
            pygame.display.init()

        if self.render_mode == "human":
            self._screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption(self.window_title)
        else:
            self._screen = pygame.Surface((self.window_width, self.window_height))

        self._clock = pygame.time.Clock()

        # Load fonts
        try:
            self._font_small = pygame.font.SysFont(
                ["menlo", "consolas", "dejavusansmono", "courier"], 11
            )
            self._font_medium = pygame.font.SysFont(
                ["menlo", "consolas", "dejavusansmono", "courier"], 14, bold=True
            )
        except Exception:
            self._font_small = pygame.font.Font(None, 14)
            self._font_medium = pygame.font.Font(None, 18)

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Converts metric arena world coordinates (meters) to screen pixel coordinates.

        Args:
            x: Position along arena X axis in meters [0.0, arena_width_m].
            y: Position along arena Y axis in meters [0.0, arena_height_m].

        Returns:
            Tuple of (pixel_x, pixel_y) screen coordinates.
        """
        px = int(round(self.arena_rect_left + x * self.pixels_per_meter))
        if self.origin_bottom_left:
            # Y points up in Cartesian space, inverted on screen
            py = int(
                round(
                    self.arena_rect_top
                    + (self.arena_height_m - y) * self.pixels_per_meter
                )
            )
        else:
            py = int(round(self.arena_rect_top + y * self.pixels_per_meter))
        return px, py

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Converts screen pixel coordinates back to metric arena coordinates.

        Args:
            px: Screen pixel X coordinate.
            py: Screen pixel Y coordinate.

        Returns:
            Tuple of (x, y) arena world coordinates in meters.
        """
        x = (px - self.arena_rect_left) / self.pixels_per_meter
        if self.origin_bottom_left:
            y = self.arena_height_m - (
                (py - self.arena_rect_top) / self.pixels_per_meter
            )
        else:
            y = (py - self.arena_rect_top) / self.pixels_per_meter
        return float(x), float(y)

    def handle_events(self) -> bool:
        """Processes OS and Pygame window events (e.g. window close, ESC/Q keys).

        Returns:
            True if visualizer should continue running, False if exit requested.
        """
        if self.render_mode != "human":
            return self.is_open

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.is_open = False
                    return False

        return self.is_open

    def _draw_top_header(self, surface: pygame.Surface) -> None:
        """Draws top information header bar with title and metadata chips."""
        header_rect = pygame.Rect(0, 0, self.window_width, self.top_bar_height_px)
        pygame.draw.rect(surface, self.COLOR_HEADER_BG, header_rect)
        pygame.draw.line(
            surface,
            self.COLOR_GRID_MAJOR,
            (0, self.top_bar_height_px - 1),
            (self.window_width, self.top_bar_height_px - 1),
            width=1,
        )

        if self._font_medium:
            title_surf = self._font_medium.render(
                "AUTONOMOUS RC TESTBED (1:28) | ARENA SIMULATOR",
                True,
                self.COLOR_HEADER_TEXT,
            )
            surface.blit(title_surf, (15, 12))

        if self._font_small:
            fps_val = self._clock.get_fps() if self._clock else 0.0
            meta_str = (
                f"Arena: {self.arena_width_m:.2f}m x {self.arena_height_m:.2f}m | "
                f"Scale: {int(self.pixels_per_meter)}px/m | "
                f"Target FPS: {self.render_fps} ({fps_val:.1f} live)"
            )
            meta_surf = self._font_small.render(meta_str, True, self.COLOR_TEXT)
            meta_rect = meta_surf.get_rect()
            meta_rect.right = self.window_width - 15
            meta_rect.centery = self.top_bar_height_px // 2
            surface.blit(meta_surf, meta_rect)

    def _draw_arena_grid(self, surface: pygame.Surface) -> None:
        """Draws arena floor surface, minor/major metric grid lines, and axis labels."""
        # 1. Fill arena surface
        arena_rect = pygame.Rect(
            self.arena_rect_left,
            self.arena_rect_top,
            self.arena_width_px,
            self.arena_height_px,
        )
        pygame.draw.rect(surface, self.COLOR_ARENA_FLOOR, arena_rect)

        # 2. Draw vertical metric grid lines (along X axis)
        grid_step_minor = 0.25  # meters
        grid_step_major = 0.5  # meters

        # X lines
        x_steps = int(round(self.arena_width_m / grid_step_minor))
        for i in range(x_steps + 1):
            x_m = i * grid_step_minor
            if x_m > self.arena_width_m + 1e-4:
                break
            px_top, py_top = self.world_to_pixel(x_m, self.arena_height_m)
            px_bot, py_bot = self.world_to_pixel(x_m, 0.0)

            is_major = abs(x_m % grid_step_major) < 1e-4 or abs(
                x_m % grid_step_major - grid_step_major
            ) < 1e-4
            line_color = self.COLOR_GRID_MAJOR if is_major else self.COLOR_GRID_MINOR
            line_width = 1

            pygame.draw.line(
                surface, line_color, (px_top, py_top), (px_bot, py_bot), line_width
            )

            # Axis tick labels along bottom margin (spaced below marker boxes)
            if is_major and self._font_small:
                label = self._font_small.render(f"{x_m:.2f}m", True, self.COLOR_TEXT)
                l_rect = label.get_rect()
                l_rect.centerx = px_bot
                l_rect.top = py_bot + 20
                surface.blit(label, l_rect)

        # Y lines
        y_steps = int(round(self.arena_height_m / grid_step_minor))
        for i in range(y_steps + 1):
            y_m = i * grid_step_minor
            if y_m > self.arena_height_m + 1e-4:
                break
            px_left, py_left = self.world_to_pixel(0.0, y_m)
            px_right, py_right = self.world_to_pixel(self.arena_width_m, y_m)

            is_major = abs(y_m % grid_step_major) < 1e-4 or abs(
                y_m % grid_step_major - grid_step_major
            ) < 1e-4
            line_color = self.COLOR_GRID_MAJOR if is_major else self.COLOR_GRID_MINOR
            line_width = 1

            pygame.draw.line(
                surface,
                line_color,
                (px_left, py_left),
                (px_right, py_right),
                line_width,
            )

            # Axis tick labels along left margin (spaced left of marker boxes)
            if is_major and self._font_small:
                label = self._font_small.render(f"{y_m:.2f}m", True, self.COLOR_TEXT)
                l_rect = label.get_rect()
                l_rect.right = px_left - 20
                l_rect.centery = py_left
                surface.blit(label, l_rect)

    def _draw_arena_boundaries(self, surface: pygame.Surface) -> None:
        """Renders solid arena perimeter border."""
        arena_rect = pygame.Rect(
            self.arena_rect_left,
            self.arena_rect_top,
            self.arena_width_px,
            self.arena_height_px,
        )
        pygame.draw.rect(surface, self.COLOR_ARENA_BORDER, arena_rect, width=2)

    def _draw_corner_markers(self, surface: pygame.Surface) -> None:
        """Renders ArUco corner calibration marker anchors at the four arena corners."""
        # 4 corners in metric coordinates:
        # [0]: (0, 0)
        # [1]: (width, 0)
        # [2]: (width, height)
        # [3]: (0, height)
        corners = [
            (0.0, 0.0),
            (self.arena_width_m, 0.0),
            (self.arena_width_m, self.arena_height_m),
            (0.0, self.arena_height_m),
        ]

        marker_size_px = 22
        half_sz = marker_size_px // 2

        for i, (cx, cy) in enumerate(corners):
            px, py = self.world_to_pixel(cx, cy)
            marker_rect = pygame.Rect(
                px - half_sz, py - half_sz, marker_size_px, marker_size_px
            )

            # Draw outer marker background and highlight border
            pygame.draw.rect(surface, (10, 10, 12), marker_rect)
            pygame.draw.rect(
                surface, self.COLOR_CORNER_MARKER, marker_rect, width=2
            )

            # Draw marker ID label
            if i < len(self.corner_marker_ids) and self._font_small:
                marker_id = self.corner_marker_ids[i]
                id_surf = self._font_small.render(
                    str(marker_id), True, self.COLOR_CORNER_MARKER
                )
                id_rect = id_surf.get_rect()
                id_rect.center = (px, py)
                surface.blit(id_surf, id_rect)

    def _transform_polygon(
        self,
        center_x: float,
        center_y: float,
        yaw: float,
        local_points: List[Tuple[float, float]],
    ) -> List[Tuple[int, int]]:
        """Rotates local metric offsets by yaw and translates to screen pixels.

        Args:
            center_x: Object center X position in meters.
            center_y: Object center Y position in meters.
            yaw: Object rotation angle in radians.
            local_points: List of (dx, dy) coordinate offsets in meters.

        Returns:
            List of (pixel_x, pixel_y) screen coordinate tuples.
        """
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pixel_pts: List[Tuple[int, int]] = []
        for dx, dy in local_points:
            wx = center_x + dx * cos_yaw - dy * sin_yaw
            wy = center_y + dx * sin_yaw + dy * cos_yaw
            pixel_pts.append(self.world_to_pixel(wx, wy))
        return pixel_pts

    def draw_vehicle(
        self,
        surface: pygame.Surface,
        x: float,
        y: float,
        yaw: float = 0.0,
        steering_angle: float = 0.0,
        body_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Renders 1:28 chassis, steerable wheels, heading vector, and ArUco marker.

        Args:
            surface: Target Pygame surface to draw upon.
            x: Vehicle center position along arena X axis in meters.
            y: Vehicle center position along arena Y axis in meters.
            yaw: Vehicle heading orientation angle in radians.
            steering_angle: Front wheel steer angle in radians (positive = left).
            body_color: Optional override for primary chassis color.
        """
        chassis_color = body_color or self.COLOR_VEHICLE_BODY
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 1. Four Wheels (2 rear fixed, 2 front steerable)
        half_wb = self.vehicle_wheelbase_m / 2.0
        # Track offset so wheels protrude cleanly on the sides of the chassis
        half_tr = (self.vehicle_width_m / 2.0) + 0.004
        hl_w = 0.032 / 2.0  # 32mm wheel length
        hw_w = 0.012 / 2.0  # 12mm wheel width
        wheel_local_box = [
            (-hl_w, -hw_w),
            (hl_w, -hw_w),
            (hl_w, hw_w),
            (-hl_w, hw_w),
        ]

        wheel_configs = [
            (-half_wb, half_tr, 0.0),             # Rear-Left
            (-half_wb, -half_tr, 0.0),            # Rear-Right
            (half_wb, half_tr, steering_angle),   # Front-Left
            (half_wb, -half_tr, steering_angle),  # Front-Right
        ]

        for w_dx, w_dy, w_steer in wheel_configs:
            w_center_x = x + w_dx * cos_yaw - w_dy * sin_yaw
            w_center_y = y + w_dx * sin_yaw + w_dy * cos_yaw
            w_yaw = yaw + w_steer
            w_pts = self._transform_polygon(
                w_center_x, w_center_y, w_yaw, wheel_local_box
            )
            pygame.draw.polygon(surface, self.COLOR_VEHICLE_WHEEL, w_pts)
            pygame.draw.polygon(
                surface, self.COLOR_VEHICLE_WHEEL_BORDER, w_pts, width=1
            )

        # 2. Main Chassis Body
        hl_b = self.vehicle_length_m / 2.0  # 0.080m
        hw_b = self.vehicle_width_m / 2.0   # 0.0375m
        body_box = [
            (hl_b, hw_b),
            (-hl_b, hw_b),
            (-hl_b, -hw_b),
            (hl_b, -hw_b),
        ]
        body_pts = self._transform_polygon(x, y, yaw, body_box)
        pygame.draw.polygon(surface, chassis_color, body_pts)
        pygame.draw.polygon(surface, self.COLOR_VEHICLE_BORDER, body_pts, width=2)

        # 3. Front Bumper / Directional Hood Accent (Red)
        bumper_box = [
            (hl_b - 0.025, hw_b),
            (hl_b, hw_b),
            (hl_b, -hw_b),
            (hl_b - 0.025, -hw_b),
        ]
        bumper_pts = self._transform_polygon(x, y, yaw, bumper_box)
        pygame.draw.polygon(surface, self.COLOR_VEHICLE_NOSE, bumper_pts)

        # 4. Windshield / Cabin (Dark Blue)
        cabin_box = [
            (-0.030, hw_b - 0.010),
            (0.025, hw_b - 0.010),
            (0.025, -hw_b + 0.010),
            (-0.030, -hw_b + 0.010),
        ]
        cabin_pts = self._transform_polygon(x, y, yaw, cabin_box)
        pygame.draw.polygon(surface, self.COLOR_VEHICLE_CABIN, cabin_pts)

        # 5. Roof ArUco Marker
        half_m = self.vehicle_marker_size_m / 2.0  # 0.025m (5cm marker)
        marker_box = [
            (-half_m, -half_m),
            (half_m, -half_m),
            (half_m, half_m),
            (-half_m, half_m),
        ]
        marker_pts = self._transform_polygon(x, y, yaw, marker_box)
        pygame.draw.polygon(surface, (10, 10, 12), marker_pts)
        pygame.draw.polygon(surface, self.COLOR_CORNER_MARKER, marker_pts, width=2)

        # Marker ID label
        if self._font_small:
            cx_px, cy_px = self.world_to_pixel(x, y)
            id_surf = self._font_small.render(
                str(self.vehicle_marker_id), True, self.COLOR_CORNER_MARKER
            )
            id_rect = id_surf.get_rect()
            id_rect.center = (cx_px, cy_px)
            surface.blit(id_surf, id_rect)

        # 6. Forward Heading Vector Line
        head_end_x = x + (hl_b + 0.060) * cos_yaw
        head_end_y = y + (hl_b + 0.060) * sin_yaw
        p_front = self.world_to_pixel(x + hl_b * cos_yaw, y + hl_b * sin_yaw)
        p_end = self.world_to_pixel(head_end_x, head_end_y)
        pygame.draw.line(surface, self.COLOR_HEADING_LINE, p_front, p_end, width=2)
        pygame.draw.circle(surface, self.COLOR_HEADING_LINE, p_end, 3)

    def render(
        self,
        vehicle_pose: Optional[Tuple[float, float, float]] = None,
        steering_angle: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Renders arena canvas, grid, boundaries, markers, and vehicle.

        Args:
            vehicle_pose: Optional (x, y, yaw) in meters and radians.
                          Defaults to center of the arena if None.
            steering_angle: Front wheel steer angle in radians.

        Returns:
            RGB numpy array of shape (H, W, 3) if render_mode is 'rgb_array',
            or None when rendered directly to screen in 'human' mode.
        """
        if not self.is_open or self._screen is None:
            return None

        # Process window events in human mode
        if not self.handle_events():
            return None

        # Clear background canvas
        self._screen.fill(self.COLOR_BG)

        # Draw visual components
        self._draw_top_header(self._screen)
        self._draw_arena_grid(self._screen)
        self._draw_arena_boundaries(self._screen)
        self._draw_corner_markers(self._screen)

        # Draw vehicle (defaults to middle of the arena)
        if vehicle_pose is not None:
            vx, vy, vyaw = vehicle_pose
        else:
            vx = self.arena_width_m / 2.0
            vy = self.arena_height_m / 2.0
            vyaw = 0.0

        self.draw_vehicle(
            self._screen,
            vx,
            vy,
            yaw=vyaw,
            steering_angle=steering_angle,
        )

        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock:
                self._clock.tick(self.render_fps)
            return None

        elif self.render_mode == "rgb_array":
            # Convert surface to (height, width, 3) numpy array
            img_arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(img_arr, (1, 0, 2))

        return None

    def close(self) -> None:
        """Releases Pygame surfaces and cleanly closes display windows."""
        self.is_open = False
        if pygame.display.get_init():
            pygame.display.quit()
        if pygame.get_init():
            pygame.quit()

    def __enter__(self) -> ArenaVisualizer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Launching ArenaVisualizer (Phase 1)")
    print("Press 'ESC' or 'Q' in the window to quit.")
    print("=" * 60)
    with ArenaVisualizer(render_mode="human") as visualizer:
        while visualizer.is_open:
            visualizer.render()
    print("Visualizer closed cleanly.")
