# Autonomous Cyber-Physical Testbed (1:28 Scale)

A modular software architecture for an autonomous cyber-physical RC car testbed running on Apple macOS. The system integrates overhead computer vision perception (ArUco markers via UVC camera), an ESP32-C3 + nRF24L01 USB radio transmitter that drives the stock RC car, classical/RL control pipelines (PID, Stanley, Stable-Baselines3), and a Gymnasium simulation environment for rapid prototyping and sim-to-real transfer.

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
                       [comms.serial_transmitter]
                                     | (USB serial / 50 Hz)
                                     v
                      +-----------------------------+
                      | ESP32-C3 + nRF24L01 (firmware)|
                      +--------------+--------------+
                                     | (2.4 GHz WLtoys XK protocol)
                                     v
                      +-----------------------------+
                      |   1:28 RC Car (stock RX)    |
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
│   ├── arena_tracker.py     # ArenaTracker (per frame) + LiveTracker (threaded camera)
│   ├── track_vehicle.py     # CLI: print/preview the tracked car pose
│   └── marker_tracker.py    # ArUco detection and 2D vehicle pose estimation
├── comms/
│   ├── __init__.py
│   ├── control_command.py   # ControlCommand: normalized steering/throttle/brake
│   ├── serial_transmitter.py# USB serial client for the ESP32-C3 RC transmitter
│   └── teleop.py            # Gamepad/keyboard teleop of the real car
├── firmware/
│   └── rc_transmitter/
│       └── rc_transmitter.ino # ESP32-C3 + nRF24L01 WLtoys (XK) transmitter
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

### Test 4: Drive the Real Car (USB Transmitter)
The car is driven through an ESP32-C3 + nRF24L01 that emulates the WLtoys XK radio protocol (`firmware/rc_transmitter`). Flash it from the Arduino IDE with `Tools -> USB CDC On Boot: Enabled`; wiring and the serial protocol are documented at the top of the sketch.

```bash
source .venv/bin/activate
python -m comms.teleop --axes      # check gamepad axes
python -m comms.teleop --bind      # bind (power the car on), then drive
python -m comms.teleop             # drive
python -m comms.teleop --list      # list ports, if auto-detect fails
python -m comms.teleop /dev/cu.usbmodem1101   # drive on a specific port
```

Controls match the simulator: left stick steers, RT/LT forward/reverse, A stops, LB/RB change the speed profile. The transmitter port is found automatically by its USB ID. Close the Arduino Serial Monitor first. The firmware goes to neutral if no command arrives for 300 ms.

From code, stream `ControlCommand`s at a steady rate (~50 Hz):

```python
from comms import ControlCommand, SerialTransmitter, find_transmitter_port

with SerialTransmitter(find_transmitter_port(), max_throttle=0.6) as tx:
    while running:
        tx.send_command(ControlCommand(steering=s, throttle=t))  # both -1..1, +right / +forward
```

### Test 5: Real Car in the Simulator (Digital Twin)
Tracks the real car with the overhead camera and draws it in **green** in the simulator, next to the simulated (blue) car:

```bash
source .venv/bin/activate
python -m simulation.run_simulation --tracking                  # sim car + real car
python -m simulation.run_simulation --tracking-only             # real car only
python -m simulation.run_simulation --tracking --camera-preview # + annotated camera window
python -m simulation.run_simulation --tracking --source run.mp4 # recorded video instead of camera
```

The header shows the tracking state (`CALIBRATING n/4 corners`, `SEARCHING CAR`, or the real car's speed). Press `F` to snap the simulated car onto the real car's pose and speed, and `L` to lock the homography once calibrated (useful when the car occludes a corner marker). Drive the real car at the same time with `python -m comms.teleop` in a second terminal.

From code, `LiveTracker` gives the newest pose without blocking:

```python
from vision import LiveTracker

with LiveTracker() as tracker:          # camera, arena and flip from config
    car = tracker.latest()              # TrackedVehicle or None if not seen for 0.5 s
    if car:
        print(car.x, car.y, car.yaw, car.velocity)
```

---

## 5. Configuration & Safety Parameters

All operational settings are centralized in `config/settings.py`:
- **`CameraConfig`**: Device index (`camera_id`), resolution (`1280x720`), FPS target (`30`).
- **`TransmitterConfig`**: Serial port (empty = auto-detect), send rate (`50 Hz`), throttle cap (`0.60`) and steering inversion for the USB RC transmitter.
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
