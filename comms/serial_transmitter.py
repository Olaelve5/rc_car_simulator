"""USB serial client for the ESP32-C3 + nRF24L01 RC transmitter.

The ESP32 (firmware/rc_transmitter) emulates the WLtoys XK radio protocol, so
the stock car is driven directly without onboard electronics. The PC streams
newline-terminated text commands over USB:

    d STEER THR   drive, both -1000..1000 (THR > 0 = forward)
    n             neutral
    b             bind (3 s, power the car on during this)

Each ``d`` line doubles as a heartbeat: the firmware goes to neutral if none
arrives for 300 ms, so commands must be streamed at a steady rate (~50 Hz).
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import serial
from serial.tools import list_ports

from .control_command import ControlCommand

WIRE_SCALE = 1000  # Firmware command range is -1000..1000

# ESP32-C3 built-in USB Serial/JTAG (Espressif vendor ID)
ESP32C3_USB_IDS = ((0x303A, 0x1001),)


def list_serial_ports() -> List[Tuple[str, str]]:
    """Returns (device, description) for every serial port on the host."""
    return [(p.device, p.description) for p in list_ports.comports()]


def find_transmitter_port() -> Optional[str]:
    """Finds the ESP32-C3 transmitter by its USB vendor/product ID.

    macOS names the port after the physical USB socket, so the device path
    differs between ports and machines while the USB IDs stay the same.

    Returns:
        The device path if exactly one transmitter is connected, else None.

    Raises:
        RuntimeError: If more than one matching device is connected.
    """
    matches = [
        p.device
        for p in list_ports.comports()
        if (p.vid, p.pid) in ESP32C3_USB_IDS and not p.device.startswith("/dev/tty.")
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Several ESP32-C3 boards found ({', '.join(matches)}); pass the port."
        )
    return matches[0] if matches else None


class SerialTransmitter:
    """Streams drive commands to the ESP32-C3 transmitter over USB serial.

    Attributes:
        port: Serial device path (e.g. /dev/cu.usbmodem1101).
        max_throttle: Throttle magnitude cap in [0.0, 1.0].
        invert_steering: Flip steering sign on the wire. The firmware's
            positive STEER turns the 284131 left, so this is True for the
            ControlCommand convention (left < 0 < right).
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        max_throttle: float = 1.0,
        invert_steering: bool = True,
        echo: bool = True,
    ) -> None:
        """Opens the port without resetting the board and sends neutral."""
        self.port = port
        self.max_throttle = float(max_throttle)
        self.invert_steering = bool(invert_steering)
        self._echo = echo

        self._ser = serial.Serial()
        self._ser.port = port
        self._ser.baudrate = baud
        self._ser.timeout = 0.1
        self._ser.write_timeout = 0.1
        # Keep DTR/RTS low so opening the port doesn't reset the ESP32-C3.
        self._ser.dtr = False
        self._ser.rts = False
        self._ser.open()

        # Background reader keeps the ESP's output drained (and optionally echoed).
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        time.sleep(2.0)  # If the board did reset, let setup() finish
        self.neutral()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._ser.readline()
            except (serial.SerialException, OSError):
                break
            if line and self._echo:
                print("[esp]", line.decode(errors="replace").rstrip())

    def _send(self, text: str) -> None:
        self._ser.write((text + "\n").encode("ascii"))

    def send_command(self, command: ControlCommand) -> bool:
        """Sends one drive command. Brake overrides throttle to neutral.

        Args:
            command: Normalized steering (left < 0 < right) and throttle.

        Returns:
            True if the line was written to the serial port.
        """
        steer = max(-1.0, min(1.0, command.steering))
        if self.invert_steering:
            steer = -steer
        cap = self.max_throttle
        throttle = 0.0 if command.brake else max(-cap, min(cap, command.throttle))
        try:
            self._send(
                f"d {round(steer * WIRE_SCALE)} {round(throttle * WIRE_SCALE)}"
            )
        except (serial.SerialException, OSError):
            return False
        return True

    def neutral(self) -> None:
        """Centers steering and stops the motor."""
        self._send("n")

    def emergency_stop(self) -> bool:
        """Sends neutral immediately.

        Returns:
            True if the neutral command was written.
        """
        try:
            self.neutral()
        except (serial.SerialException, OSError):
            return False
        return True

    def bind(self) -> None:
        """Starts a 3 s bind burst; power the car on during it."""
        self._send("b")

    def close(self) -> None:
        """Sends neutral and closes the port."""
        try:
            self.emergency_stop()
            time.sleep(0.05)
        finally:
            self._stop.set()
            self._ser.close()

    def __enter__(self) -> "SerialTransmitter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
