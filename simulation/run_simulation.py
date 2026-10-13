"""Interactive driving simulator for 1:28 cyber-physical RC car testbed.

Enables real-time keyboard piloting using either Dynamic (with tire slip & drift)
or Kinematic (ideal geometry) physics models in the Pygame 2D ArenaVisualizer.

Controls:
    W / UP          : Throttle forward
    S / DOWN        : Brake / Reverse
    A / LEFT        : Steer left
    D / RIGHT       : Steer right
    SPACE           : Handbrake (Emergency stop)
    M               : Toggle physics engine (Dynamic Drift <-> Kinematic Rails)
    R               : Reset vehicle to arena center
    C               : Clear breadcrumb motion trail
    1 / 2 / 3       : Switch speed profile (1: Crawl 35%, 2: Normal 60%, 3: Sport 100%)
    H               : Display controls help in terminal
    ESC / Q         : Exit simulator

Usage:
    python -m simulation.run_simulation                    # Dynamic model (default)
    python -m simulation.run_simulation --model kinematic  # Kinematic model
    python -m simulation.run_simulation --mode 3 --fps 60
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Union
import pygame

# Ensure project root is discoverable on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DEFAULT_SETTINGS, VehicleDynamicsConfig  # noqa: E402
from controllers.pid_controller import PIDController  # noqa: E402
from controllers.stanley_controller import StanleyController  # noqa: E402
from simulation.arena_visualizer import ArenaVisualizer  # noqa: E402
from simulation.bicycle_model import KinematicBicycleModel  # noqa: E402
from simulation.dynamic_bicycle_model import DynamicBicycleModel  # noqa: E402
from simulation.track_generator import TrackGenerator  # noqa: E402
from simulation.vehicle_state import VehicleState  # noqa: E402


class InteractiveSimulator:
    """Manages physics updates, keyboard input capture, and visual rendering loop."""

    SPEED_MODES = {
        1: ("Crawl (35%)", 0.35),
        2: ("Normal (60%)", 0.60),
        3: ("Sport (100%)", 1.00),
    }

    def __init__(
        self,
        arena_width_m: float = 2.0,
        arena_height_m: float = 1.5,
        fps: int = 50,
        speed_mode: int = 2,
        model_type: str = "dynamic",
        controller: str = "manual",
        initial_pose: Optional[Tuple[float, float, float]] = None,
        clamp_to_arena: bool = True,
    ) -> None:
        """Initializes simulation handles, physics parameters, and visualizer window."""
        self.arena_width_m = float(arena_width_m)
        self.arena_height_m = float(arena_height_m)
        self.fps = int(fps)
        self.dt = 1.0 / self.fps
        self.speed_mode = speed_mode
        self.clamp_to_arena = clamp_to_arena

        # Reference trajectory waypoints (Oval track centered in arena)
        cx = self.arena_width_m / 2.0
        cy = self.arena_height_m / 2.0
        self.waypoints = TrackGenerator.generate_oval(
            length_m=self.arena_width_m * 0.70,
            width_m=self.arena_height_m * 0.60,
            center=(cx, cy),
        )

        # Initial pose defaults to track start aligned with path tangent
        if initial_pose is not None:
            init_x, init_y, init_yaw = initial_pose
        else:
            init_x = float(self.waypoints[0, 0])
            init_y = float(self.waypoints[0, 1])
            dx = float(self.waypoints[1, 0] - init_x)
            dy = float(self.waypoints[1, 1] - init_y)
            init_yaw = math.atan2(dy, dx)

        self.initial_state = VehicleState(
            x=init_x,
            y=init_y,
            yaw=init_yaw,
            velocity=0.0,
            steering_angle=0.0,
            v_x=0.0,
            v_y=0.0,
            yaw_rate=0.0,
        )

        self.params = VehicleDynamicsConfig(dt=self.dt)
        self.kinematic_model = KinematicBicycleModel(
            params=self.params,
            initial_state=self.initial_state,
        )
        self.dynamic_model = DynamicBicycleModel(
            params=self.params,
            initial_state=self.initial_state,
        )

        # Set active model
        if model_type.strip().lower() == "kinematic":
            self.model: Union[KinematicBicycleModel, DynamicBicycleModel] = (
                self.kinematic_model
            )
        else:
            self.model = self.dynamic_model

        # Autonomous tracking controllers
        self.stanley_controller = StanleyController(
            waypoints=self.waypoints,
            dt=self.dt,
            target_velocity=0.60,
        )
        self.pid_controller = PIDController(
            waypoints=self.waypoints,
            dt=self.dt,
            target_velocity=0.55,
        )
        ctrl_mode = controller.strip().lower()
        if ctrl_mode == "pid":
            self.active_controller = "pid"
            self.autopilot = True
        elif ctrl_mode == "stanley":
            self.active_controller = "stanley"
            self.autopilot = True
        else:
            self.active_controller = "stanley"
            self.autopilot = False

        # Visualizer interface with track waypoints overlay
        self.visualizer = ArenaVisualizer(
            arena_width_m=self.arena_width_m,
            arena_height_m=self.arena_height_m,
            render_fps=self.fps,
            render_mode="human",
            window_title="RC Car Testbed | Interactive Simulator",
            waypoints=self.waypoints,
        )

        # Collision & status tracking
        self._collision_alert_timer: float = 0.0
        self._last_print_time: float = 0.0

    def print_banner(self) -> None:
        """Prints user keybindings and configuration summary to terminal."""
        print("\n" + "=" * 65)
        print("  1:28 RC CAR CYBER-PHYSICAL INTERACTIVE SIMULATOR")
        print("=" * 65)
        print(
            f"Arena Dimensions : {self.arena_width_m:.2f}m x "
            f"{self.arena_height_m:.2f}m"
        )
        print(f"Simulation Rate  : {self.fps} Hz (dt = {self.dt*1000:.1f}ms)")
        print(f"Active Physics   : {self.model.model_name.upper()} Model")
        mode_desc = self.SPEED_MODES[self.speed_mode][0]
        print(f"Speed Profile    : Mode {self.speed_mode} - {mode_desc}")
        print("-" * 65)
        print("KEYBOARD CONTROLS:")
        print("  [W] / [UP]     : Forward Throttle")
        print("  [S] / [DOWN]   : Active Brake / Reverse")
        print("  [A] / [LEFT]   : Steer Left")
        print("  [D] / [RIGHT]  : Steer Right")
        print("  [SPACE]        : Emergency Handbrake")
        print("  [M]            : Toggle Physics Model (Dynamic / Kinematic)")
        print("  [R]            : Reset to Arena Center")
        print("  [C]            : Clear Breadcrumb Trail")
        print("  [1] / [2] / [3]: Select Speed Profile (Crawl / Normal / Sport)")
        print("  [P]            : Toggle Autopilot ON/OFF")
        print("  [O]            : Switch Autopilot Controller (Stanley <-> PID)")
        print("  [H]            : Print Controls Help")
        print("  [ESC] / [Q]    : Quit Simulator")
        print("=" * 65 + "\n")

    def toggle_physics_model(self) -> None:
        """Seamlessly switches between Dynamic and Kinematic physics models."""
        curr_state = self.model.get_state()
        if isinstance(self.model, DynamicBicycleModel):
            self.model = self.kinematic_model
            mode_note = "KINEMATIC (Pure geometric tracking on rails)"
        else:
            self.model = self.dynamic_model
            mode_note = "DYNAMIC (Tire slip, lateral grip saturation & drift)"

        self.model.reset(curr_state)
        print(f"\n>> Switched physics engine to: {mode_note}")

    def _on_key_down(self, key: int) -> None:
        """Handles discrete single-press keyboard events."""
        if key == pygame.K_m:
            self.toggle_physics_model()
        elif key == pygame.K_p:
            self.autopilot = not self.autopilot
            status_desc = "ENABLED" if self.autopilot else "DISABLED"
            ctrl_name = self.active_controller.upper()
            print(f"\n>> Autopilot [{ctrl_name}]: {status_desc}")
        elif key == pygame.K_o:
            self.active_controller = (
                "pid" if self.active_controller == "stanley" else "stanley"
            )
            print(
                f"\n>> Switched Autopilot Controller to: "
                f"{self.active_controller.upper()}"
            )
        elif key == pygame.K_r:
            self.reset()
            print(">> Vehicle reset to center.")
        elif key == pygame.K_c:
            self.visualizer.clear_trail()
            print(">> Breadcrumb trail cleared.")
        elif key in (pygame.K_1, pygame.K_KP1):
            self.speed_mode = 1
            print(f">> Switched to Speed Mode 1: {self.SPEED_MODES[1][0]}")
        elif key in (pygame.K_2, pygame.K_KP2):
            self.speed_mode = 2
            print(f">> Switched to Speed Mode 2: {self.SPEED_MODES[2][0]}")
        elif key in (pygame.K_3, pygame.K_KP3):
            self.speed_mode = 3
            print(f">> Switched to Speed Mode 3: {self.SPEED_MODES[3][0]}")
        elif key == pygame.K_h:
            self.print_banner()

    def reset(self) -> None:
        """Resets the vehicle model to initial position and clears motion trail."""
        self.model.reset(self.initial_state)
        self.visualizer.clear_trail()

    def _handle_continuous_input(self) -> Tuple[float, float]:
        """Polls continuous keyboard states to compute steering and throttle demands.

        Returns:
            Tuple of (steering_target_rad, throttle_command).
        """
        keys = pygame.key.get_pressed()

        # 1. Steering target demand
        steer_target = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer_target += self.params.max_steer_rad
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer_target -= self.params.max_steer_rad

        # 2. Throttle command (scaled by selected speed profile)
        throttle_scale = self.SPEED_MODES[self.speed_mode][1]
        throttle = 0.0

        if keys[pygame.K_SPACE]:
            # Handbrake: zero throttle and strong active braking
            throttle = -1.0
        elif keys[pygame.K_w] or keys[pygame.K_UP]:
            throttle = 1.0 * throttle_scale
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            throttle = -1.0 * throttle_scale

        return steer_target, throttle

    def _handle_boundary_collisions(self, dt: float) -> bool:
        """Enforces arena perimeter wall constraints with physical bumper rebound.

        Args:
            dt: Simulation time step.

        Returns:
            True if collision occurred this step.
        """
        if not self.clamp_to_arena:
            return False

        # Chassis clearance boundaries (half vehicle length and width)
        half_len = 0.080  # 80mm half length
        half_wid = 0.038  # 38mm half width
        buffer = max(half_len, half_wid)

        min_x = buffer
        max_x = self.arena_width_m - buffer
        min_y = buffer
        max_y = self.arena_height_m - buffer

        state = self.model.state
        collided = False

        if state.x < min_x:
            state.x = min_x
            state.velocity = -state.velocity * 0.25  # Inelastic rebound
            state.v_x = -state.v_x * 0.25
            collided = True
        elif state.x > max_x:
            state.x = max_x
            state.velocity = -state.velocity * 0.25
            state.v_x = -state.v_x * 0.25
            collided = True

        if state.y < min_y:
            state.y = min_y
            state.velocity = -state.velocity * 0.25
            state.v_x = -state.v_x * 0.25
            collided = True
        elif state.y > max_y:
            state.y = max_y
            state.velocity = -state.velocity * 0.25
            state.v_x = -state.v_x * 0.25
            collided = True

        return collided

    def run(self) -> None:
        """Executes main interactive simulation loop."""
        self.print_banner()

        clock = pygame.time.Clock()

        try:
            while self.visualizer.is_open:
                # 1. Capture driving inputs (manual keyboard or autonomous autopilot)
                manual_steer, manual_throttle = self._handle_continuous_input()
                if (
                    self.autopilot
                    and abs(manual_steer) < 1e-3
                    and abs(manual_throttle) < 1e-3
                ):
                    active_agent = (
                        self.pid_controller
                        if self.active_controller == "pid"
                        else self.stanley_controller
                    )
                    act = active_agent.compute_action(
                        state=self.model.get_state(),
                        reference=self.waypoints,
                    )
                    steer_target = float(act[0]) * self.params.max_steer_rad
                    throttle_scale = self.SPEED_MODES[self.speed_mode][1]
                    throttle = float(act[1]) * throttle_scale
                else:
                    steer_target = manual_steer
                    throttle = manual_throttle

                # 2. Advance vehicle physics step
                self.model.step(
                    steering_target=steer_target,
                    throttle_input=throttle,
                    dt=self.dt,
                )

                # 3. Arena boundary collision handling
                collided = self._handle_boundary_collisions(self.dt)
                if collided:
                    self._collision_alert_timer = 0.4  # display alert for 0.4s

                # 4. Compose status telemetry bar
                state = self.model.get_state()
                steer_deg = math.degrees(state.steering_angle)
                yaw_deg = math.degrees(state.yaw)
                drift_deg = math.degrees(state.drift_angle_rad)

                if self._collision_alert_timer > 0:
                    self._collision_alert_timer -= self.dt
                    status = (
                        f"** WALL CONTACT ** | v: {state.velocity:+.2f} m/s | "
                        f"Pos: ({state.x:.2f}, {state.y:.2f})m"
                    )
                else:
                    mode_name = self.SPEED_MODES[self.speed_mode][0]
                    model_label = self.model.model_name.upper()

                    # Highlight drift when sideslip angle is significant
                    if (
                        isinstance(self.model, DynamicBicycleModel)
                        and abs(drift_deg) > 2.5
                        and abs(state.velocity) > 0.20
                    ):
                        physics_tag = f"[{model_label} DRIFT {drift_deg:+4.1f}°]"
                    else:
                        physics_tag = f"[{model_label}]"

                    if self.autopilot:
                        cte, _, _ = TrackGenerator.compute_tracking_errors(
                            self.waypoints, state.x, state.y, state.yaw
                        )
                        ctrl_tag = self.active_controller.upper()
                        pilot_tag = f"[AUTOPILOT: {ctrl_tag} | CTE: {cte:+.3f}m]"
                    else:
                        pilot_tag = f"[{mode_name}]"

                    status = (
                        f"{physics_tag} {pilot_tag} | "
                        f"v: {state.velocity:+.2f} m/s | "
                        f"Steer: {steer_deg:+4.1f}° | "
                        f"Pos: ({state.x:.2f}, {state.y:.2f})m | "
                        f"Yaw: {yaw_deg:+5.1f}°"
                    )

                # 5. Render to screen with discrete key handler
                self.visualizer.render(
                    vehicle_pose=(state.x, state.y, state.yaw),
                    steering_angle=state.steering_angle,
                    status_text=status,
                    key_callback=self._on_key_down,
                )

                # 6. Periodic terminal logging (every 1.5 seconds)
                now = time.monotonic()
                if now - self._last_print_time >= 1.5:
                    self._last_print_time = now
                    fps_live = self.visualizer.fps
                    drift_str = (
                        f" | drift={drift_deg:+4.1f}°"
                        if isinstance(self.model, DynamicBicycleModel)
                        else ""
                    )
                    print(
                        f"[{self.model.model_name[:3]}] v={state.velocity:+.2f}m/s | "
                        f"steer={steer_deg:+4.1f}°{drift_str} | "
                        f"pos=({state.x:.2f}m, {state.y:.2f}m) | "
                        f"FPS={fps_live:.0f}"
                    )

                clock.tick(self.fps)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            self.visualizer.close()
            print("Simulator closed cleanly.")


def main() -> None:
    """CLI entry point for running the interactive driving simulator."""
    arena = DEFAULT_SETTINGS.arena

    parser = argparse.ArgumentParser(
        description="Interactive 1:28 RC Car Simulation with Pygame."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dynamic", "kinematic"],
        default="dynamic",
        help=(
            "Physics model engine: 'dynamic' (tire slip & drift) "
            "or 'kinematic' (default: dynamic)."
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
        "--fps",
        type=int,
        default=50,
        help="Target simulation and rendering frame rate (default: 50).",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Initial speed mode: 1 (Crawl 35%%), 2 (Normal 60%%), 3 (Sport 100%%).",
    )
    parser.add_argument(
        "--controller",
        type=str,
        choices=["manual", "stanley", "pid"],
        default="manual",
        help="Initial driving control mode: 'manual' (default), 'stanley', or 'pid'.",
    )
    parser.add_argument(
        "--no-walls",
        action="store_true",
        help="Disable arena wall collision clamping.",
    )

    args = parser.parse_args()

    sim = InteractiveSimulator(
        arena_width_m=args.width,
        arena_height_m=args.height,
        fps=args.fps,
        speed_mode=args.mode,
        model_type=args.model,
        controller=args.controller,
        clamp_to_arena=not args.no_walls,
    )
    sim.run()


if __name__ == "__main__":
    main()
