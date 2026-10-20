"""Drive the physical RC car with a gamepad or keyboard via the USB transmitter.

Uses the same InputManager mapping as the simulator (left stick steers, RT/LT
forward/reverse, A = stop, LB/RB = speed profile) and streams commands to the
ESP32-C3 transmitter at the configured rate. Keyboard input needs the small
status window focused; the gamepad works in the background.

Close the Arduino Serial Monitor first - only one program can hold the port.

Usage:
    python -m comms.teleop --list                      # list serial ports
    python -m comms.teleop --axes                      # print raw gamepad axes
    python -m comms.teleop /dev/cu.usbmodem1101        # drive
    python -m comms.teleop /dev/cu.usbmodem1101 --bind # bind, then drive
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Keep reading the controller even when the window doesn't have focus.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame  # noqa: E402

# Ensure project root is discoverable on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from comms.esp32_client import ControlCommand  # noqa: E402
from comms.serial_transmitter import (  # noqa: E402
    SerialTransmitter,
    list_serial_ports,
)
from config.settings import DEFAULT_SETTINGS  # noqa: E402
from controllers.input_manager import InputManager  # noqa: E402


def show_axes() -> None:
    """Prints live axis values of the first gamepad, to verify the mapping."""
    pygame.init()
    if pygame.joystick.get_count() == 0:
        sys.exit("No controller found - plug it in and try again.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()} ({js.get_numaxes()} axes)")
    print("Move the sticks; Ctrl+C to quit.")
    try:
        while True:
            pygame.event.pump()
            vals = "  ".join(
                f"{i}:{js.get_axis(i):+.2f}" for i in range(js.get_numaxes())
            )
            print("\r" + vals, end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


def drive_loop(tx: SerialTransmitter, inputs: InputManager, rate_hz: float) -> None:
    """Polls inputs and streams drive commands until the window is closed."""
    screen = pygame.display.set_mode((420, 90))
    pygame.display.set_caption("RC Teleop")
    font = pygame.font.SysFont("menlo", 16)

    print(
        f"Driving at {rate_hz:.0f} Hz. Left stick: steer, RT/LT: forward/reverse, "
        "A: stop, LB/RB: speed. ESC or Ctrl+C to quit."
    )
    print("If a trigger doesn't respond, press and release it once.")

    period = 1.0 / rate_hz
    next_t = time.perf_counter()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE,
                pygame.K_q,
            ):
                return
            inputs.handle_event(event)

        demand = inputs.poll()
        command = ControlCommand(
            # InputDemand is +left; ControlCommand is +right.
            steering=-demand.steering_rad / inputs.max_steer_rad,
            throttle=demand.throttle,
            brake=demand.brake,
            timestamp=time.time(),
        )
        tx.send_command(command)

        mode = InputManager.SPEED_MODES[demand.speed_mode][0]
        status = [
            f"{demand.source:8s} {mode}",
            f"steer {command.steering:+.2f}  throttle {command.throttle:+.2f}"
            + ("  STOP" if command.brake else ""),
        ]
        screen.fill((20, 20, 24))
        for i, text in enumerate(status):
            screen.blit(font.render(text, True, (220, 220, 220)), (12, 14 + 28 * i))
        pygame.display.flip()

        next_t += period
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()  # Fell behind; don't try to catch up


def main() -> None:
    cfg = DEFAULT_SETTINGS.transmitter
    ap = argparse.ArgumentParser(description="Gamepad -> ESP32 transmitter -> car")
    ap.add_argument("port", nargs="?", default=cfg.port, help="serial port")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--axes", action="store_true", help="print gamepad axes and exit")
    ap.add_argument("--bind", action="store_true", help="bind before driving")
    ap.add_argument(
        "--mode", type=int, default=3, choices=(1, 2, 3), help="initial speed profile"
    )
    args = ap.parse_args()

    if args.list:
        for device, description in list_serial_ports():
            print(f"{device:28s} {description}")
        return
    if args.axes:
        show_axes()
        return
    if not args.port:
        ap.error("port is required (use --list to find it)")

    pygame.init()
    inputs = InputManager(initial_speed_mode=args.mode)
    tx = SerialTransmitter(
        args.port,
        baud=cfg.baud,
        max_throttle=cfg.max_throttle,
        invert_steering=cfg.invert_steering,
    )
    try:
        if args.bind:
            tx.bind()
            print("Binding for 3 s - power the car on now.")
            time.sleep(3.2)
        drive_loop(tx, inputs, cfg.send_rate_hz)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        tx.close()
        pygame.quit()


if __name__ == "__main__":
    main()
