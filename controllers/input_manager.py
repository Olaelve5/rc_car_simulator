"""Human-Machine Interface (HMI) input manager for cyber-physical RC car testbed.

Supports Xbox Wireless Controller (via Bluetooth or USB-C on macOS) and keyboard
arrow keys / WASD with deadzones, exponential steering curves, and hot-plugging.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
import numpy as np
import pygame


@dataclass
class InputDemand:
    """Normalized control demands emitted by the input manager.

    Attributes:
        steering_rad: Steering angle demand in radians (+left, -right).
        throttle: Longitudinal throttle command in range [-1.0, 1.0].
        brake: Emergency handbrake / cutoff boolean override.
        speed_mode: Active speed profile index (1: Crawl, 2: Normal, 3: Sport).
        source: Active input device description ('gamepad' or 'keyboard').
    """

    steering_rad: float = 0.0
    throttle: float = 0.0
    brake: bool = False
    speed_mode: int = 2
    source: str = "keyboard"


class InputManager:
    """Unifies Xbox Gamepad and Keyboard inputs with response curves and hot-plugging.

    Maps macOS SDL2 / Pygame joystick axes and buttons to physical vehicle actuation:
      - Left Thumbstick X : Proportional steering with center deadzone & expo curve.
      - Right Trigger (RT): Proportional forward throttle [0.0, 1.0].
      - Left Trigger (LT) : Proportional reverse / active braking [0.0, 1.0].
      - Left Stick Y      : Alternative throttle / reverse stick control.
      - A Button / Space  : Emergency handbrake stop.
      - B Button          : Reset vehicle to track / arena center.
      - X Button          : Clear motion breadcrumb trail.
      - Y Button          : Toggle autonomous autopilot ON / OFF.
      - LB / RB Bumpers   : Step speed profile Down / Up.
      - D-Pad (Hat)       : Cycle tracks (Left/Right) & controllers (Up/Down).
    """

    SPEED_MODES: Dict[int, Tuple[str, float]] = {
        1: ("Crawl (35%)", 0.35),
        2: ("Normal (60%)", 0.60),
        3: ("Sport (100%)", 1.00),
    }

    def __init__(
        self,
        max_steer_rad: float = 0.5236,  # ~30 degrees
        deadzone: float = 0.08,
        steer_expo: float = 1.3,
        initial_speed_mode: int = 2,
    ) -> None:
        """Initializes Pygame joystick subsystem and controller state."""
        self.max_steer_rad = float(max_steer_rad)
        self.deadzone = float(deadzone)
        self.steer_expo = float(steer_expo)
        self.speed_mode = int(initial_speed_mode)

        # Trigger calibration states (SDL2 triggers may rest at -1.0 or 0.0 initially)
        self._rt_calibrated = False
        self._lt_calibrated = False

        # Active joystick handle
        self.joystick: Optional[pygame.joystick.Joystick] = None
        self._init_joystick_subsystem()

    def _init_joystick_subsystem(self) -> None:
        """Initializes joystick module and attaches first available controller."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count > 0:
            self._attach_joystick(0)
        else:
            self.joystick = None

    def _attach_joystick(self, device_index: int) -> None:
        """Attaches and initializes a detected joystick device."""
        try:
            joy = pygame.joystick.Joystick(device_index)
            joy.init()
            self.joystick = joy
            self._rt_calibrated = False
            self._lt_calibrated = False
            name = joy.get_name()
            print(f"\n[InputManager] 🎮 Connected Game Controller: '{name}'")
            print(
                f"             Axes: {joy.get_numaxes()} | "
                f"Buttons: {joy.get_numbuttons()} | "
                f"Hats: {joy.get_numhats()}"
            )
        except pygame.error as err:
            print(f"[InputManager] Cannot attach controller {device_index}: {err}")
            self.joystick = None

    @property
    def is_gamepad_connected(self) -> bool:
        """Returns True if a physical game controller is currently initialized."""
        return self.joystick is not None and self.joystick.get_init()

    @property
    def device_name(self) -> str:
        """Returns the human-readable name of the active input controller."""
        if self.is_gamepad_connected and self.joystick is not None:
            return self.joystick.get_name()
        return "Keyboard"

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Processes discrete event signals (hotplug, button clicks, key presses).

        Returns:
            Optional string action identifier for simulator/teleop runner:
              'reset', 'clear_trail', 'toggle_autopilot', 'cycle_controller',
              'toggle_physics', 'cycle_track', 'print_help', or None.
        """
        # 1. Hotplugging events
        if event.type == pygame.JOYDEVICEADDED:
            self._attach_joystick(event.device_index)
            return None
        elif event.type == pygame.JOYDEVICEREMOVED:
            print("\n[InputManager] ⚠️ Controller detached. Reverting to keyboard.")
            self.joystick = None
            return None

        # 2. Gamepad discrete button presses
        if event.type == pygame.JOYBUTTONDOWN and self.is_gamepad_connected:
            btn = event.button
            # Xbox standard mapping:
            # 0: A (Handbrake / Stop)
            # 1: B (Reset to center)
            # 2: X (Clear trail)
            # 3: Y (Toggle Autopilot)
            # 4: LB (Speed mode down)
            # 5: RB (Speed mode up)
            # 6: View / Back (Toggle physics)
            # 7: Menu / Start (Cycle track)
            if btn == 1:
                return "reset"
            elif btn == 2:
                return "clear_trail"
            elif btn == 3:
                return "toggle_autopilot"
            elif btn == 4:
                self.step_speed_mode(-1)
                return None
            elif btn == 5:
                self.step_speed_mode(+1)
                return None
            elif btn == 6:
                return "toggle_physics"
            elif btn == 7:
                return "cycle_track"

        # 3. Gamepad D-pad (Hat) events
        if event.type == pygame.JOYHATMOTION and self.is_gamepad_connected:
            hat_x, hat_y = event.value
            if hat_x != 0:
                return "cycle_track"
            elif hat_y != 0:
                return "cycle_controller"

        return None

    def step_speed_mode(self, direction: int) -> int:
        """Increments or decrements speed profile within [1, 3]."""
        new_mode = max(1, min(3, self.speed_mode + direction))
        if new_mode != self.speed_mode:
            self.speed_mode = new_mode
            mode_name = self.SPEED_MODES[self.speed_mode][0]
            print(f">> [Input] Speed Profile: Mode {self.speed_mode} ({mode_name})")
        return self.speed_mode

    def _shape_axis(self, raw_value: float) -> float:
        """Applies deadzone thresholding and exponential scaling to an analog axis."""
        val = float(np.clip(raw_value, -1.0, 1.0))
        abs_val = abs(val)

        if abs_val <= self.deadzone:
            return 0.0

        # Rescale normalized range [deadzone, 1.0] -> [0.0, 1.0]
        rescaled = (abs_val - self.deadzone) / (1.0 - self.deadzone)
        rescaled = float(np.clip(rescaled, 0.0, 1.0))

        # Apply exponential curve for fine center fidelity
        curved = rescaled**self.steer_expo
        return math.copysign(curved, val)

    def _read_trigger(self, raw_value: float, is_rt: bool) -> float:
        """Normalizes macOS SDL2 trigger axis value [-1.0, 1.0] to [0.0, 1.0]."""
        # Handle uncalibrated rest state quirk on macOS
        if raw_value < -0.1:
            if is_rt:
                self._rt_calibrated = True
            else:
                self._lt_calibrated = True

        calibrated = self._rt_calibrated if is_rt else self._lt_calibrated
        if calibrated:
            norm = (raw_value + 1.0) / 2.0
        else:
            # If not yet calibrated and positive, use direct positive reading
            norm = max(0.0, raw_value)

        norm = float(np.clip(norm, 0.0, 1.0))
        return norm if norm > self.deadzone else 0.0

    def poll(self) -> InputDemand:
        """Polls continuous inputs from Gamepad and Keyboard.

        Returns:
            InputDemand with steering_rad, throttle, brake, and active speed_mode.
        """
        throttle_scale = self.SPEED_MODES[self.speed_mode][1]

        # -------------------------------------------------------------
        # 1. Gamepad Input
        # -------------------------------------------------------------
        gamepad_active = False
        gp_steer_norm = 0.0
        gp_throttle = 0.0
        gp_brake = False

        if self.is_gamepad_connected and self.joystick is not None:
            num_axes = self.joystick.get_numaxes()
            num_btns = self.joystick.get_numbuttons()

            # Axis 0: Left Stick X
            # Note: Steering convention in bicycle model:
            # Positive (+) steering turns LEFT. Negative (-) steering turns RIGHT.
            # Stick Left gives raw negative value -> invert to get positive steer angle!
            if num_axes > 0:
                raw_steer = self.joystick.get_axis(0)
                shaped_steer = self._shape_axis(raw_steer)
                # Invert: stick left (-raw) -> positive steer (+left)
                gp_steer_norm = -shaped_steer
                if abs(gp_steer_norm) > 1e-3:
                    gamepad_active = True

            # Throttle & Brake: Triggers (preferred on Xbox)
            rt_norm = 0.0
            lt_norm = 0.0
            if num_axes >= 6:
                # macOS SDL2 standard mapping: LT=Axis 4, RT=Axis 5
                lt_norm = self._read_trigger(self.joystick.get_axis(4), is_rt=False)
                rt_norm = self._read_trigger(self.joystick.get_axis(5), is_rt=True)
            elif num_axes >= 3:
                # Fallback mapping
                rt_norm = self._read_trigger(self.joystick.get_axis(2), is_rt=True)

            # Left Stick Y as fallback throttle if triggers untouched
            stick_y_throttle = 0.0
            if num_axes > 1:
                # Stick UP is negative in Pygame -> invert for forward throttle
                raw_y = -self.joystick.get_axis(1)
                stick_y_throttle = self._shape_axis(raw_y)

            if rt_norm > 0.01 or lt_norm > 0.01:
                # Triggers active
                gp_throttle = (rt_norm - lt_norm) * throttle_scale
                gamepad_active = True
            elif abs(stick_y_throttle) > 0.01:
                # Stick Y active
                gp_throttle = stick_y_throttle * throttle_scale
                gamepad_active = True

            # Button 0 (A): Emergency handbrake
            if num_btns > 0 and self.joystick.get_button(0):
                gp_brake = True
                gp_throttle = -1.0
                gamepad_active = True

        # -------------------------------------------------------------
        # 2. Keyboard Input (Arrow keys & WASD fallback)
        # -------------------------------------------------------------
        keys = pygame.key.get_pressed()
        kb_active = False
        kb_steer_norm = 0.0
        kb_throttle = 0.0
        kb_brake = False

        # Steering
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            kb_steer_norm += 1.0  # Turn left
            kb_active = True
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            kb_steer_norm -= 1.0  # Turn right
            kb_active = True

        # Throttle & Brake
        if keys[pygame.K_SPACE]:
            kb_brake = True
            kb_throttle = -1.0
            kb_active = True
        elif keys[pygame.K_w] or keys[pygame.K_UP]:
            kb_throttle = 1.0 * throttle_scale
            kb_active = True
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            kb_throttle = -1.0 * throttle_scale
            kb_active = True

        # Speed mode hotkeys
        if keys[pygame.K_1] or keys[pygame.K_KP1]:
            self.speed_mode = 1
        elif keys[pygame.K_2] or keys[pygame.K_KP2]:
            self.speed_mode = 2
        elif keys[pygame.K_3] or keys[pygame.K_KP3]:
            self.speed_mode = 3

        # -------------------------------------------------------------
        # 3. Fusion & Arbitration
        # -------------------------------------------------------------
        # If keyboard is actively used, it blends or overrides gamepad
        if kb_active:
            final_steer_norm = float(np.clip(kb_steer_norm + gp_steer_norm, -1.0, 1.0))
            final_throttle = kb_throttle
            final_brake = kb_brake or gp_brake
            source = "keyboard"
        elif gamepad_active:
            final_steer_norm = float(np.clip(gp_steer_norm, -1.0, 1.0))
            final_throttle = gp_throttle
            final_brake = gp_brake
            source = "gamepad"
        else:
            final_steer_norm = 0.0
            final_throttle = 0.0
            final_brake = False
            source = "idle"

        steering_rad = final_steer_norm * self.max_steer_rad

        return InputDemand(
            steering_rad=steering_rad,
            throttle=final_throttle,
            brake=final_brake,
            speed_mode=self.speed_mode,
            source=source,
        )
