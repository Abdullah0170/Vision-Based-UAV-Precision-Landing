# Vision-Based UAV Precision Landing (ArUco) — Raspberry Pi 5 + Camera Module 3 + Pixhawk

Vision-based precision landing for a multirotor UAV. A downward-facing Raspberry Pi
Camera Module 3 detects an ArUco marker, estimates its pose with `solvePnP`
(IPPE_SQUARE), filters it with a constant-velocity Kalman filter, and a PID controller
steers the drone over the marker over MAVLink (`pymavlink`) until the final LAND hand-over.

Works with **ArduPilot** (GUIDED or GUIDED_NOGPS) and **PX4** (OFFBOARD).

## Features

- Pi Camera (picamera2, newest-frame / no-lag mode) or a USB camera / video file for testing
- Fast detection on a downscaled image with sub-pixel corner refinement at full resolution
- Attitude-compensated marker position (camera → body FRD → level heading frame)
- 6-state Kalman filter with outlier rejection (NIS gate)
- XY and Z PID controllers; ALIGN → DESCEND → LAND state machine with SEARCH / LOST timeouts
- No-GPS flight (ArduPilot GUIDED_NOGPS): velocity commands converted to roll/pitch with an integral wind trim
- Touchdown detection and automatic (forced if needed) disarm
- CSV flight log, optional annotated video recording and live camera window
- Bench tools: `--vision-only` and the guided `--fix-directions` camera-orientation test

## Hardware

- Raspberry Pi 5 with Camera Module 3 (pointing down)
- Pixhawk-class flight controller connected to the Pi UART (default `/dev/ttyAMA0` @ 921600)
- Printed ArUco marker (default `DICT_5X5_50`, ID 0, 0.10 m black square)

## Installation (Raspberry Pi OS)

```bash
sudo apt install -y python3-picamera2
python3 -m venv --system-site-packages ~/drone/venv
source ~/drone/venv/bin/activate
pip install -r requirements.txt
```

`picamera2` comes from apt, so the venv must be created with `--system-site-packages`.
Use the non-headless OpenCV build if you want the live camera window.

## Camera calibration

The script needs a camera calibration `.npz` containing the camera matrix and
distortion coefficients (default path `~/Desktop/camera_module3_pi5_calibration.npz`,
change with `--calib`). The calibration file is not included in this repository.

## Usage

```bash
# 1. Bench test - no drone connection, shows what the drone would do
python precision_land.py --vision-only

# 2. Find the camera orientation (saves ~/drone/directions.json, used automatically)
python precision_land.py --fix-directions

# 3. Ground arm test (arms for 5 s, camera check only)
python precision_land.py --alt 0

# 4. Flight: take off to 2 m, precision-land on the marker
python precision_land.py --alt 2

# Without GPS (ArduPilot GUIDED_NOGPS, attitude mode)
python precision_land.py --alt 1.5 --no-gps
```

Main options:

| Option | Default | Meaning |
|---|---|---|
| `--marker-size` | 0.1 | edge of the black marker square (m) |
| `--marker-id` / `--dict` | 0 / 5X5_50 | marker ID and ArUco dictionary |
| `--calib` | `~/Desktop/camera_module3_pi5_calibration.npz` | calibration file |
| `--device` / `--baud` | `/dev/ttyAMA0` / 921600 | MAVLink connection |
| `--alt` | 2 | take-off height (m), 0 = ground test |
| `--final-alt` | 0.35 | camera height where LAND mode takes over (m) |
| `--max-xy-speed` | 0.30 | horizontal speed limit (m/s) |
| `--cam-yaw` | from saved file | camera rotation vs. drone front (deg) |
| `--record` | – | save annotated video |
| `--no-gps`, `--no-rc`, `--no-checks` | – | relaxed / no-GPS operation |

Run `python precision_land.py --help` for the full list.

## Optional helper

If `set_arming_check.py` (parameter helpers: `read_param`, `checks_off`, `checks_restore`,
`save_backup`) is in the same folder, the script uses it to read `FRAME_CLASS` /
`PILOT_SPEED_UP/DN` and to temporarily disable pre-arm checks. Without it the script
falls back to force-arming.

## Output

- Flight logs: `~/drone/logs/pland_YYYYMMDD_HHMMSS.csv`
- Camera snapshot: `~/drone/camera_view.jpg`

## Safety

This is experimental research code. Test with `--vision-only` and `--alt 0` first, fly
in an open area, keep an RC transmitter ready to take over, and remove the propellers
for any bench test with the flight controller connected.

## Author

Abdullah R. Radam — Electrical Engineering Technical College, Middle Technical University, Baghdad, Iraq
