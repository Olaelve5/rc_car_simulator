"""Smoke test: Iterates through camera device indices (0 to 3) for UVC capture."""

from __future__ import annotations

import time
import cv2


def check_camera_index(index: int, backend_name: str = "AVFOUNDATION") -> bool:
    """Attempts to connect to a camera index and grab a test frame.

    Args:
        index: Camera device integer index (0-3).
        backend_name: Name of backend used for logging.

    Returns:
        True if camera opened and successfully returned a valid frame.
    """
    print(f"\n--- Testing Camera Index {index} (Backend: {backend_name}) ---")
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        # Fallback to default backend
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"[-] Index {index}: Device could not be opened.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[+] Index {index}: Successfully opened!")
    print(
        f"    Reported Resolution: {width}x{height} @ {fps:.1f} FPS "
        f"(FOURCC: {fourcc_str})"
    )

    # Warm up sensor and attempt frame capture
    time.sleep(0.2)
    success, frame = cap.read()

    if success and frame is not None:
        print(f"    Captured Frame Shape: {frame.shape}, dtype={frame.dtype}")
        print(f"[OK] Camera index {index} is functional and streaming frames.")
        cap.release()
        return True
    else:
        print(f"[!] Camera index {index} opened, but failed to read a frame.")
        cap.release()
        return False


def main() -> None:
    """Iterates through indices 0 to 3 to discover video devices."""
    print("=" * 60)
    print("macOS UVC Camera Hardware Discovery & Verification")
    print("=" * 60)
    print("Scanning indices 0 through 3 using OpenCV AVFoundation backend...\n")

    found_devices = []
    for idx in range(4):
        if check_camera_index(idx):
            found_devices.append(idx)

    print("\n" + "=" * 60)
    if found_devices:
        print(f"SUCCESS: Functional camera(s) found at indices: {found_devices}")
        print(
            "Tip: Set `camera_id` in config/settings.py to your UVC camera index."
        )
    else:
        print("NOTICE: No active video streams responded on indices 0-3.")
        print("Notes for macOS:")
        print("  1. Ensure your UVC camera is plugged into a USB port.")
        print("  2. Check macOS Camera permissions:")
        print("     System Settings -> Privacy & Security -> Camera (allow terminal).")
    print("=" * 60)


if __name__ == "__main__":
    main()
