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
    F               : Snap simulated car to the tracked real car (--tracking)
    L               : Lock / unlock the camera homography (--tracking)
    ESC / Q         : Exit simulator

Usage:
    python -m simulation.run_simulation                    # Dynamic model (default)
    python -m simulation.run_simulation --model kinematic  # Kinematic model
    python -m simulation.run_simulation --mode 3 --fps 60
    python -m simulation.run_simulation --tracking         # + real car from camera
    python -m simulation.run_simulation --tracking-only    # real car only
    python -m simulation.run_simulation --tracking --source run.mp4 --camera-preview
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import pygame

# Ensure project root is discoverable on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DEFAULT_SETTINGS, VehicleDynamicsConfig  
from controllers.input_manager import InputManager  
from controllers.pid_controller import PIDController  
from controllers.stanley_controller import StanleyController  
from simulation.arena_visualizer import ArenaVisualizer  
from simulation.bicycle_model import KinematicBicycleModel  
from simulation.dynamic_bicycle_model import DynamicBicycleModel  
from simulation.track_generator import TrackGenerator  
from simulation.vehicle_state import VehicleState  
from vision.arena_tracker import LiveTracker  # noqa: E402
from vision.marker_tracker import VehiclePose  # noqa: E402
from vision.track_vehicle import draw_overlay  # noqa: E402


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
        track_type: str = "oval",
        initial_pose: Optional[Tuple[float, float, float]] = None,
        clamp_to_arena: bool = True,
        tracker: Optional[LiveTracker] = None,
        show_sim_vehicle: bool = True,
        camera_preview: bool = False,
    ) -> None:
        """Initializes simulation handles, physics parameters, and visualizer window."""
        self.arena_width_m = float(arena_width_m)
        self.arena_height_m = float(arena_height_m)
        self.fps = int(fps)
        self.dt = 1.0 / self.fps
        self.speed_mode = speed_mode
        self.clamp_to_arena = clamp_to_arena
        self.track_type = track_type.strip().lower()

        # Camera tracking of the real car (digital twin overlay)
        self.tracker = tracker
        self.show_sim_vehicle = show_sim_vehicle
        self.camera_preview = camera_preview and tracker is not None
        self._preview_frame_id = -1

        # Reference trajectory waypoints
        self.waypoints = self._create_track(self.track_type)

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
        elif ctrl_mode == "simple_pid":
            self.active_controller = "simple_pid"
            self.autopilot = True
        elif ctrl_mode == "stanley":
            self.active_controller = "stanley"
            self.autopilot = True
        else:
            self.active_controller = "stanley"
            self.autopilot = False

        # Human-machine interface input manager (Xbox Controller + Keyboard)
        self.input_manager = InputManager(
            max_steer_rad=self.params.max_steer_rad,
            initial_speed_mode=self.speed_mode,
        )

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

    def _create_track(self, track_type: str) -> np.ndarray:
        """Generates reference track waypoints via TrackGenerator."""
        cx = self.arena_width_m / 2.0
        cy = self.arena_height_m / 2.0
        kind = track_type.strip().lower()

        if kind == "circle":
            radius = min(self.arena_width_m, self.arena_height_m) * 0.35
            return TrackGenerator.generate_circle(radius_m=radius, center=(cx, cy))
        elif kind == "figure_eight":
            return TrackGenerator.generate_figure_eight(
                length_m=self.arena_width_m * 0.70,
                width_m=self.arena_height_m * 0.60,
                center=(cx, cy),
            )
        else:
            return TrackGenerator.generate_oval(
                length_m=self.arena_width_m * 0.70,
                width_m=self.arena_height_m * 0.60,
                center=(cx, cy),
            )

    def switch_track(self, track_type: Optional[str] = None) -> None:
        """Switches or cycles through track layouts (Oval -> Circle -> Figure-8)."""
        track_sequence = ["oval", "circle", "figure_eight"]
        if track_type is not None:
            self.track_type = track_type.strip().lower()
        else:
            curr_idx = (
                track_sequence.index(self.track_type)
                if self.track_type in track_sequence
                else 0
            )
            self.track_type = track_sequence[(curr_idx + 1) % len(track_sequence)]

        self.waypoints = self._create_track(self.track_type)
        self.visualizer.set_waypoints(self.waypoints)
        self.stanley_controller.set_waypoints(self.waypoints)
        self.pid_controller.set_waypoints(self.waypoints)
        self.simple_pid_controller.set_waypoints(self.waypoints)
        print(f"\n>> Switched track layout to: {self.track_type.upper()}")

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
        print(f"Active Track     : {self.track_type.upper()}")
        mode_desc = self.SPEED_MODES[self.speed_mode][0]
        print(f"Speed Profile    : Mode {self.speed_mode} - {mode_desc}")
        ctrl_status = (
            f"🎮 {self.input_manager.device_name}"
            if self.input_manager.is_gamepad_connected
            else "⌨️ Keyboard (No Gamepad Detected)"
        )
        print(f"Input Device     : {ctrl_status}")
        print("-" * 65)
        print("XBOX CONTROLLER (BLUETOOTH / USB):")
        print("  [Left Stick X] : Proportional Steering (Analog with deadzone & curve)")
        print("  [RT]           : Forward Throttle (Proportional)")
        print("  [LT]           : Brake / Reverse (Proportional)")
        print("  [Left Stick Y] : Dual Throttle / Reverse (Stick alternative)")
        print("  [A Button]     : Emergency Handbrake Stop")
        print("  [B Button]     : Reset Vehicle to Center")
        print("  [X Button]     : Clear Breadcrumb Trail")
        print("  [Y Button]     : Toggle Autopilot ON/OFF")
        print("  [LB] / [RB]    : Step Speed Profile (Crawl / Normal / Sport)")
        print("  [D-Pad Left/Rt]: Cycle Track Layout (Oval -> Circle -> Fig-8)")
        print("  [D-Pad Up/Down]: Cycle Autopilot Controller (Stanley / PID)")
        print("-" * 65)
        print("KEYBOARD CONTROLS:")
        print("  [W] / [UP]     : Forward Throttle")
        print("  [S] / [DOWN]   : Active Brake / Reverse")
        print("  [A] / [LEFT]   : Steer Left")
        print("  [D] / [RIGHT]  : Steer Right")
        print("  [SPACE]        : Emergency Handbrake")
        print("  [M]            : Toggle Physics Model (Dynamic / Kinematic)")
        print("  [T]            : Cycle Track Layout (Oval -> Circle -> Figure-8)")
        print("  [R]            : Reset to Arena Center")
        print("  [C]            : Clear Breadcrumb Trail")
        print("  [1] / [2] / [3]: Select Speed Profile (Crawl / Normal / Sport)")
        print("  [P]            : Toggle Autopilot ON/OFF")
        print("  [O]            : Cycle Autopilot (Stanley -> PID -> Simple-PID)")
        print("  [H]            : Print Controls Help")
        if self.tracker is not None:
            print("  [F]            : Snap Simulated Car to Real Car")
            print("  [L]            : Lock / Unlock Camera Homography")
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

    def cycle_controller(self) -> None:
        """Cycles autopilot controllers (Stanley / PID / Simple-PID)."""
        ctrl_cycle = ["stanley", "pid", "simple_pid"]
        curr_i = (
            ctrl_cycle.index(self.active_controller)
            if self.active_controller in ctrl_cycle
            else 0
        )
        self.active_controller = ctrl_cycle[(curr_i + 1) % len(ctrl_cycle)]
        print(
            f"\n>> Switched Autopilot Controller to: "
            f"{self.active_controller.upper()}"
        )

    def _on_event(self, event: pygame.event.Event) -> None:
        """Handles Pygame discrete events, delegating to InputManager."""
        action = self.input_manager.handle_event(event)
        if action == "reset":
            self.reset()
            print(">> Vehicle reset to center (Gamepad B).")
        elif action == "clear_trail":
            self.visualizer.clear_trail()
            print(">> Breadcrumb trail cleared (Gamepad X).")
        elif action == "toggle_autopilot":
            self.autopilot = not self.autopilot
            status_desc = "ENABLED" if self.autopilot else "DISABLED"
            ctrl_name = self.active_controller.upper()
            print(f"\n>> Autopilot [{ctrl_name}]: {status_desc} (Gamepad Y)")
        elif action == "cycle_track":
            self.switch_track()
        elif action == "cycle_controller":
            self.cycle_controller()
        elif action == "toggle_physics":
            self.toggle_physics_model()

    def _on_key_down(self, key: int) -> None:
        """Handles discrete single-press keyboard events."""
        if key == pygame.K_m:
            self.toggle_physics_model()
        elif key == pygame.K_t:
            self.switch_track()
        elif key == pygame.K_p:
            self.autopilot = not self.autopilot
            status_desc = "ENABLED" if self.autopilot else "DISABLED"
            ctrl_name = self.active_controller.upper()
            print(f"\n>> Autopilot [{ctrl_name}]: {status_desc}")
        elif key == pygame.K_o:
            self.cycle_controller()
        elif key == pygame.K_r:
            self.reset()
            print(">> Vehicle reset to center.")
        elif key == pygame.K_c:
            self.visualizer.clear_trail()
            print(">> Breadcrumb trail cleared.")
        elif key in (pygame.K_1, pygame.K_KP1):
            self.speed_mode = 1
            self.input_manager.speed_mode = 1
            print(f">> Switched to Speed Mode 1: {self.SPEED_MODES[1][0]}")
        elif key in (pygame.K_2, pygame.K_KP2):
            self.speed_mode = 2
            self.input_manager.speed_mode = 2
            print(f">> Switched to Speed Mode 2: {self.SPEED_MODES[2][0]}")
        elif key in (pygame.K_3, pygame.K_KP3):
            self.speed_mode = 3
            self.input_manager.speed_mode = 3
            print(f">> Switched to Speed Mode 3: {self.SPEED_MODES[3][0]}")
        elif key == pygame.K_h:
            self.print_banner()
        elif key == pygame.K_f and self.tracker is not None:
            self.snap_to_tracked()
        elif key == pygame.K_l and self.tracker is not None:
            locked = self.tracker.toggle_lock()
            print(f">> Camera homography {'LOCKED' if locked else 'UNLOCKED'}")

    def snap_to_tracked(self) -> None:
        """Resets the simulated car to the tracked real car's pose and speed."""
        real = self.tracker.latest() if self.tracker is not None else None
        if real is None:
            print(">> Cannot snap: real car is not being tracked.")
            return
        self.model.reset(
            VehicleState(
                x=real.x,
                y=real.y,
                yaw=real.yaw,
                velocity=real.velocity,
                v_x=real.velocity,
            )
        )
        print(f">> Simulated car snapped to real car at ({real.x:.2f}, {real.y:.2f})m")

    def _show_camera_preview(self) -> None:
        """Shows the annotated camera frame in an OpenCV window when it changes."""
        assert self.tracker is not None
        frame_id, frame, detections = self.tracker.latest_frame()
        if frame is None or frame_id == self._preview_frame_id:
            return
        self._preview_frame_id = frame_id
        real = self.tracker.latest()
        pose = (
            VehiclePose(real.x, real.y, real.yaw, real.timestamp)
            if real is not None
            else None
        )
        overlay = draw_overlay(frame, self.tracker.tracker, pose, detections)
        cv2.imshow("Overhead camera", overlay)
        cv2.waitKey(1)

    def reset(self) -> None:
        """Resets the vehicle model to initial position and clears motion trail."""
        self.model.reset(self.initial_state)
        self.visualizer.clear_trail()
        self.stanley_controller.reset()
        self.pid_controller.reset()
        self.simple_pid_controller.reset()

    def _handle_continuous_input(self) -> Tuple[float, float, bool]:
        """Polls unified inputs from InputManager (Gamepad + Keyboard).

        Returns:
            Tuple of (steering_target_rad, throttle_command, brake_override).
        """
        demand = self.input_manager.poll()
        self.speed_mode = demand.speed_mode
        return demand.steering_rad, demand.throttle, demand.brake

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
                # 1. Capture driving inputs (manual or autonomous autopilot)
                manual_steer, manual_throttle, manual_brake = (
                    self._handle_continuous_input()
                )
                if (
                    self.autopilot
                    and abs(manual_steer) < 1e-3
                    and abs(manual_throttle) < 1e-3
                    and not manual_brake
                ):
                    active_agent: Union[
                        PIDController, StanleyController
                    ]
                    if self.active_controller == "pid":
                        active_agent = self.pid_controller
                    elif self.active_controller == "simple_pid":
                        active_agent = self.simple_pid_controller
                    else:
                        active_agent = self.stanley_controller
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

                    dev_label = (
                        "🎮 " + self.input_manager.device_name[:12]
                        if self.input_manager.is_gamepad_connected
                        else "⌨️ KEYS"
                    )

                    if self.autopilot:
                        cte, _, _ = TrackGenerator.compute_tracking_errors(
                            self.waypoints, state.x, state.y, state.yaw
                        )
                        ctrl_tag = self.active_controller.upper()
                        pilot_tag = f"[AUTOPILOT: {ctrl_tag} | CTE: {cte:+.3f}m]"
                    else:
                        pilot_tag = f"[{mode_name} | {dev_label}]"

                    status = (
                        f"{physics_tag} {pilot_tag} | "
                        f"v: {state.velocity:+.2f} m/s | "
                        f"Steer: {steer_deg:+4.1f}° | "
                        f"Pos: ({state.x:.2f}, {state.y:.2f})m | "
                        f"Yaw: {yaw_deg:+5.1f}°"
                    )

                # 5. Real car from the camera tracker, if enabled
                real = self.tracker.latest() if self.tracker is not None else None
                if self.tracker is not None:
                    real_tag = (
                        f"REAL v: {real.velocity:+.2f} m/s"
                        if real is not None
                        else f"REAL: {self.tracker.status}"
                    )
                    status = f"[{real_tag}] {status}"
                    if self.camera_preview:
                        self._show_camera_preview()

                # 6. Render to screen with discrete key and event handlers
                self.visualizer.render(
                    vehicle_pose=(state.x, state.y, state.yaw),
                    steering_angle=state.steering_angle,
                    status_text=status,
                    key_callback=self._on_key_down,
                    event_callback=self._on_event,
                    tracked_pose=(
                        (real.x, real.y, real.yaw) if real is not None else None
                    ),
                    show_vehicle=self.show_sim_vehicle,
                )

                # 7. Periodic terminal logging (every 1.5 seconds)
                now = time.monotonic()
                if now - self._last_print_time >= 1.5:
                    self._last_print_time = now
                    fps_live = self.visualizer.fps
                    drift_str = (
                        f" | drift={drift_deg:+4.1f}°"
                        if isinstance(self.model, DynamicBicycleModel)
                        else ""
                    )
                    input_label = (
                        f"controller={self.input_manager.device_name}"
                        if self.input_manager.is_gamepad_connected
                        else "input=keyboard"
                    )
                    if self.tracker is not None:
                        print(
                            f"[real] {self.tracker.status} | "
                            + (
                                f"pos=({real.x:.2f}m, {real.y:.2f}m) | "
                                f"yaw={math.degrees(real.yaw):+5.1f}° | "
                                f"v={real.velocity:+.2f}m/s | "
                                if real is not None
                                else ""
                            )
                            + f"cam={self.tracker.camera.measured_fps:.0f}fps"
                        )
                    print(
                        f"[{self.model.model_name[:3]}] v={state.velocity:+.2f}m/s | "
                        f"steer={steer_deg:+4.1f}°{drift_str} | "
                        f"pos=({state.x:.2f}m, {state.y:.2f}m) | "
                        f"{input_label} | "
                        f"FPS={fps_live:.0f}"
                    )

                clock.tick(self.fps)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            if self.tracker is not None:
                self.tracker.stop()
            if self.camera_preview:
                cv2.destroyAllWindows()
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
        choices=["manual", "stanley", "pid", "simple_pid"],
        default="manual",
        help=(
            "Initial driving control mode: 'manual' (default), "
            "'stanley', 'pid', or 'simple_pid'."
        ),
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=["oval", "circle", "figure_eight"],
        default="oval",
        help="Reference track layout: 'oval' (default), 'circle', or 'figure_eight'.",
    )
    parser.add_argument(
        "--no-walls",
        action="store_true",
        help="Disable arena wall collision clamping.",
    )

    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Track the real car with the overhead camera and draw it in green.",
    )
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Like --tracking, but hide the simulated car.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SETTINGS.camera.camera_id),
        help="Camera index or video file for --tracking (default: from config).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable the 180-degree camera rotation for --tracking.",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="Also show the annotated camera feed in an OpenCV window.",
    )

    args = parser.parse_args()

    tracker: Optional[LiveTracker] = None
    if args.tracking or args.tracking_only:
        tracker = LiveTracker(
            source=args.source,
            flip_camera=False if args.no_flip else None,
            arena_width_m=args.width,
            arena_height_m=args.height,
        )
        try:
            tracker.start()
        except RuntimeError as err:
            parser.exit(1, f"Camera tracking unavailable: {err}\n")

    sim = InteractiveSimulator(
        arena_width_m=args.width,
        arena_height_m=args.height,
        fps=args.fps,
        speed_mode=args.mode,
        model_type=args.model,
        controller=args.controller,
        track_type=args.track,
        clamp_to_arena=not args.no_walls,
        tracker=tracker,
        show_sim_vehicle=not args.tracking_only,
        camera_preview=args.camera_preview,
    )
    sim.run()


if __name__ == "__main__":
    main()
