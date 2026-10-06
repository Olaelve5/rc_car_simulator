# Autonomous Cyber-Physical Testbed (1:28 Scale)

A modular software architecture for an autonomous cyber-physical RC car testbed running on Apple macOS. The system integrates overhead computer vision perception (ArUco markers via UVC camera), wireless UDP telemetry with an onboard ESP32-C3 microcontroller, classical/RL control pipelines (PID, Stanley, Stable-Baselines3), and a Gymnasium simulation environment for rapid prototyping and sim-to-real transfer.

---

## 1. System Architecture Overview

```
                      +-----------------------------+
                      |   Overhead UVC USB Camera   |
                      +--------------+--------------+
                                     |
                                     v
                       [vision.camera_stream]
                                     |
                                     v
                     [vision.homography / marker]
                                     |
                     (Vehicle Pose: x, y, yaw, v)
                                     |
                                     v
+------------------------- [controllers] --------------------------+
|  - controllers.pid_controller      (PID lateral & longitudinal)   |
|  - controllers.stanley_controller  (Stanley front-axle tracking)  |
|  - controllers.rl_policy_agent     (Stable-Baselines3 Policy)     |
+------------------------------------+-----------------------------+
                                     |
                       (Steering & Throttle Commands)
                                     |
                                     v
                          [comms.esp32_client]
                                     | (UDP over Wi-Fi / 50 Hz)
                                     v
                      +-----------------------------+
                      |    1:28 RC Car (ESP32-C3)   |
                      +-----------------------------+
```

---

## 2. Repository Layout

```
testbed_core/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration (camera, network, safety bounds)
├── vision/
│   ├── __init__.py
│   ├── camera_stream.py     # Threaded UVC capture grabber using cv2.VideoCapture
│   ├── homography.py        # 4-point perspective warp and metric arena mapping
│   └── marker_tracker.py    # ArUco detection and 2D vehicle pose estimation
├── comms/
│   ├── __init__.py
│   └── esp32_client.py      # UDP client streaming commands and receiving telemetry
├── simulation/
│   ├── __init__.py
│   ├── bicycle_model.py     # Discrete 2D kinematic bicycle model state updates
│   ├── arena_env.py         # Gymnasium.Env interface for RL and sim-to-real
│   └── track_generator.py   # Reference path generation (circles, figure-8, ovals)
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py   # Abstract base class: compute_action(state, reference)
│   ├── pid_controller.py    # Classical decoupled PID controller stub
│   ├── stanley_controller.py# Stanley non-linear path tracking controller stub
│   └── rl_policy_agent.py   # Stable-Baselines3 policy inference wrapper
├── tests/
│   ├── __init__.py
│   ├── verify_camera.py     # Camera discovery test (scans indices 0-3 via AVFoundation)
│   └── verify_imports.py    # Dependency & module smoke verification
├── .gitignore               # Python, macOS (.DS_Store), virtualenv, IDE exclusions
├── requirements.txt         # Core dependencies
└── README.md                # Documentation and run instructions
```

---

## 3. Environment Setup (macOS)

### Prerequisites
- macOS (Apple Silicon or Intel)
- Python 3.10 or 3.11 installed (e.g. via Homebrew: `brew install python@3.10`)
- Overhead USB UVC camera connected
- Terminal application with Camera permissions granted in macOS Settings

### Step 1: Create Virtual Environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Verification & Smoke Tests

### Test 1: Verify Dependencies and Local Module Imports
Verifies that `torch` (with Apple Silicon Metal Performance Shaders / MPS), `cv2.aruco`, `gymnasium`, `stable_baselines3`, `pygame`, and all local repository modules import without errors:

```bash
source .venv/bin/activate
python tests/verify_imports.py
```

### Test 2: Detect & Verify Overhead UVC Camera
Scans video indices 0 through 3 using OpenCV's macOS `CAP_AVFOUNDATION` backend:

```bash
source .venv/bin/activate
python tests/verify_camera.py
```

*Note: On macOS, ensure your terminal emulator (e.g., Terminal, iTerm2, VS Code) has permission to access the camera under `System Settings -> Privacy & Security -> Camera`.*

### Test 3: Live Vehicle Tracking (Homography)
Detects the four corner markers (IDs `0, 1, 2, 3`), computes the pixel -> arena homography, and continuously prints the car marker (ID `4`) pose in meters:

```bash
source .venv/bin/activate
python -m vision.track_vehicle                    # camera from config
python -m vision.track_vehicle --source run.mp4   # recorded video
python -m vision.track_vehicle --width 1.8 --height 1.2 --no-display
```

Corner marker centers map to `0 -> (0, 0)`, `1 -> (W, 0)`, `2 -> (W, H)`, `3 -> (0, H)`; place them counter-clockwise seen from above. `W`/`H` are the measured distances between marker centers. Mount the car marker with its top edge facing the front of the car (yaw = 0 along +X). In the preview, press `l` to lock the homography and `q` to quit.

---

## 5. Configuration & Safety Parameters

All operational settings are centralized in `config/settings.py`:
- **`CameraConfig`**: Device index (`camera_id`), resolution (`1280x720`), FPS target (`30`).
- **`NetworkConfig`**: ESP32 destination IP (`192.168.1.100`), UDP control port (`8888`), control loop frequency (`50 Hz`).
- **`SafetyConfig`**: Throttle cap limit (`max_throttle_pct = 0.50`), crawling speed limit (`0.30 m/s`), and watchdog timeout (`0.25 s`).
- **`ArenaConfig`**: Metric dimensions (`2.0m x 1.5m`), ArUco dictionary (`DICT_4X4_50`), and marker IDs.

---

## 6. Code Style & Static Analysis

Format and lint using standard tools:
```bash
source .venv/bin/activate

# Code formatting
black .

# Style and syntax checks
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Type checking
mypy --ignore-missing-imports .
```
