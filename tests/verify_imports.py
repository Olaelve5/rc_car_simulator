"""Smoke test: Verifies third-party dependencies and local modules import cleanly."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so local modules are discoverable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_third_party_imports() -> bool:
    """Verifies imports for all critical external libraries."""
    all_passed = True
    print("=" * 60)
    print("Verifying Third-Party Core Dependencies...")
    print("=" * 60)

    # 1. PyTorch & Apple Silicon MPS
    try:
        import torch

        mps_status = (
            "Available (Apple Silicon GPU acceleration enabled)"
            if torch.backends.mps.is_available()
            else "Not available"
        )
        print(f"[OK] torch: {torch.__version__} (MPS: {mps_status})")
    except Exception as exc:
        print(f"[FAIL] torch import failed: {exc}")
        all_passed = False

    # 2. OpenCV & ArUco
    try:
        import cv2

        if hasattr(cv2, "aruco"):
            print(
                f"[OK] cv2 (opencv-contrib-python): {cv2.__version__} with aruco module"
            )
        else:
            print(
                f"[FAIL] cv2 imported ({cv2.__version__}) but aruco submodule missing!"
            )
            all_passed = False
    except Exception as exc:
        print(f"[FAIL] cv2 / cv2.aruco import failed: {exc}")
        all_passed = False

    # 3. Gymnasium
    try:
        import gymnasium

        print(f"[OK] gymnasium: {gymnasium.__version__}")
    except Exception as exc:
        print(f"[FAIL] gymnasium import failed: {exc}")
        all_passed = False

    # 4. Stable-Baselines3
    try:
        import stable_baselines3

        print(f"[OK] stable_baselines3: {stable_baselines3.__version__}")
    except Exception as exc:
        print(f"[FAIL] stable_baselines3 import failed: {exc}")
        all_passed = False

    # 5. Pygame
    try:
        import pygame

        pg_version = getattr(pygame, "__version__", pygame.version.ver)
        print(f"[OK] pygame: {pg_version}")
    except Exception as exc:
        print(f"[FAIL] pygame import failed: {exc}")
        all_passed = False

    # 6. Additional Scientific Libraries
    try:
        import numpy as np
        import scipy
        import matplotlib

        print(f"[OK] numpy: {np.__version__}")
        print(f"[OK] scipy: {scipy.__version__}")
        print(f"[OK] matplotlib: {matplotlib.__version__}")
    except Exception as exc:
        print(f"[FAIL] scientific dependencies import failed: {exc}")
        all_passed = False

    return all_passed


def test_local_module_imports() -> bool:
    """Verifies internal packages import without circular dependencies."""
    all_passed = True
    print("\n" + "=" * 60)
    print("Verifying Local Repository Module Structure...")
    print("=" * 60)

    modules_to_test = [
        ("config.settings", "DEFAULT_SETTINGS"),
        ("vision.camera_stream", "CameraStream"),
        ("vision.homography", "HomographyTransformer"),
        ("vision.marker_tracker", "MarkerTracker"),
        ("comms.esp32_client", "ESP32Client"),
        ("simulation.bicycle_model", "KinematicBicycleModel"),
        ("simulation.arena_env", "ArenaEnv"),
        ("simulation.track_generator", "TrackGenerator"),
        ("controllers.base_controller", "BaseController"),
        ("controllers.pid_controller", "PIDController"),
        ("controllers.stanley_controller", "StanleyController"),
        ("controllers.rl_policy_agent", "RLPolicyAgent"),
    ]

    for mod_name, attr_name in modules_to_test:
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            getattr(mod, attr_name)
            print(f"[OK] {mod_name}.{attr_name}")
        except Exception as exc:
            print(f"[FAIL] {mod_name} import failed: {exc}")
            all_passed = False

    return all_passed


def main() -> None:
    """Main verification entry point."""
    print(f"Python interpreter: {sys.executable} ({sys.version.split()[0]})")
    third_party_ok = test_third_party_imports()
    local_ok = test_local_module_imports()

    print("\n" + "=" * 60)
    if third_party_ok and local_ok:
        print("RESULT: ALL IMPORTS VERIFIED SUCCESSFULLY! ✓")
        print("=" * 60)
        sys.exit(0)
    else:
        print("RESULT: SOME IMPORTS FAILED! ✗")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
