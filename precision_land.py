
import os as _os
if _os.environ.get("WAYLAND_DISPLAY") and not _os.environ.get("DISPLAY"):
    _os.environ["DISPLAY"] = ":0"                 # Pi desktop (Wayland) also runs XWayland on :0
_os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # pip OpenCV window toolkit needs X11/XWayland
import argparse
import collections
import csv
import math
import os
import threading
import time

import cv2
import numpy as np
from pymavlink import mavutil

try:
    import set_arming_check as checks      # same folder: parameter helpers (optional)
except ImportError:
    checks = None

M = mavutil.mavlink
NAN = float("nan")
VERSION = "v1 - Pi camera ArUco precision landing (based on arm_takeoff_land v7)"
FORCE = 10000

# =====================================================================================
# CONFIG (defaults - most can be changed on the command line)
# =====================================================================================
# >>> SET THIS: edge length of the BLACK square of your printed marker, in metres <<<
MARKER_SIZE = 0.1
# >>> take-off height in metres (camera must still see the marker from here) <<<
TAKEOFF_ALT = 2
# >>> True = swap FORWARD<->BACK and RIGHT<->LEFT (camera turned 180 deg) <<<
# NOTE: if you ran  python landing.py --fix-directions  its result (~/drone/directions.json)
#       is used instead and this switch is ignored.
REVERSE_DIRECTIONS = True
DIRECTIONS_FILE = "~/drone/directions.json"
# >>> size of the live camera window (width, height) in pixels <<<
WINDOW_SIZE = (640, 480)


def fit_to_window(img, size=None):
    """Scale the picture to fit inside the window WITHOUT stretching it; fill the rest black."""
    W, H = size or WINDOW_SIZE
    h, w = img.shape[:2]
    k = min(W / w, H / h)
    nw, nh = max(1, int(w * k)), max(1, int(h * k))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    return canvas
DEFAULT_CALIB = "~/Desktop/camera_module3_pi5_calibration.npz"
ARUCO_DICTS = {"4X4_50": cv2.aruco.DICT_4X4_50, "5X5_50": cv2.aruco.DICT_5X5_50,
               "5X5_100": cv2.aruco.DICT_5X5_100, "6X6_50": cv2.aruco.DICT_6X6_50,
               "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL}

CONTROL_DT = 0.08            # s, control loop period (12.5 Hz) - same as the sim
IMAGE_TIMEOUT = 0.6          # s, older detections count as "lost"
SEARCH_TIMEOUT = 20.0        # s without the marker (while high) -> normal LAND
LOST_HOLD_TIMEOUT = 4.0      # s marker lost during the approach -> normal LAND

FINAL_ALT_TO_LAND = 0.35     # m camera-to-marker height: switch to LAND mode
COMMIT_LAND_ALT = 0.80       # m: marker lost below this -> LAND immediately
LAND_TARGET_H = 0.35         # m: descent PID target height

CENTER_FRAC = 0.12           # centred if horizontal offset < CENTER_FRAC * height ...
CENTER_FRAC_FINAL = 0.18     # ... below 1 m
CENTER_MIN = 0.05            # m, never require better than this

MAX_XY_SPEED = 0.30          # m/s
DESCEND_SPEED = 0.25         # m/s minimum descent while centred (high)
DESCEND_SPEED_FINAL = 0.10   # m/s minimum descent below 1 m
MAX_Z_SPEED = 0.40           # m/s

# XY PID: input = horizontal offset in METRES, output = velocity m/s
PID_XY = dict(kp=0.55, ki=0.03, kd=0.10, output_limit=MAX_XY_SPEED, integral_limit=0.5, deadband=0.02)
# Z PID: input = height error (m) above LAND_TARGET_H, output = descent speed m/s
PID_Z = dict(kp=0.35, ki=0.01, kd=0.05, output_limit=MAX_Z_SPEED, integral_limit=0.8, deadband=0.03)

# attitude mode (no GPS): velocity error -> tilt angle
TILT_PER_MS = math.radians(9.0)   # rad of tilt per 1 m/s velocity error
TILT_I = math.radians(6.0)        # rad per (m/s * s): integral "wind trim" that removes steady drift
MAX_TILT_TRIM = math.radians(3.0) # rad, limit of the wind trim
MAX_TILT = math.radians(6.0)      # rad


# =====================================================================================
# CALIBRATION
# =====================================================================================
COMMON_SIZES = [(640, 480), (640, 360), (800, 600), (1024, 768), (1280, 720), (1280, 960),
                (1296, 972), (1536, 864), (1640, 1232), (1920, 1080), (2304, 1296),
                (2592, 1944), (3280, 2464), (4608, 2592)]


def load_calibration(path, width=None, height=None):
    """Load K and distortion from an .npz. Returns K, D, (w, h) the camera must run at."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"Calibration file not found: {path}\n  Use --calib /path/to/file.npz")
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    print(f"Calibration {path}  keys: {keys}")

    def pick(names):
        for k in keys:
            if k.lower() in names:
                return data[k]
        return None

    K = pick({"camera_matrix", "cameramatrix", "mtx", "k", "camera_mtx", "intrinsic",
              "intrinsics", "matrix", "cam_matrix", "camera_matrix_k"})
    D = pick({"dist_coeffs", "distcoeffs", "dist", "d", "distortion", "dist_coeff",
              "distortion_coefficients", "dist_coefs"})
    if K is None:     # last resort: any 3x3 array
        K = next((data[k] for k in keys if np.asarray(data[k]).shape == (3, 3)), None)
    if K is None:
        raise SystemExit(f"No 3x3 camera matrix found in {keys}")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64).ravel()

    size = pick({"image_size", "img_size", "resolution", "size", "imagesize", "image_shape",
                 "frame_size", "img_shape"})
    calib_size = None
    if size is not None:
        s = [int(v) for v in np.asarray(size).ravel()[:2]]
        calib_size = (s[0], s[1]) if s[0] >= s[1] else (s[1], s[0])   # (w, h)
    else:
        guess = (2 * K[0, 2], 2 * K[1, 2])
        calib_size = min(COMMON_SIZES, key=lambda wh: abs(wh[0] - guess[0]) + abs(wh[1] - guess[1]))
        print(f"  No image size stored - guessed {calib_size[0]}x{calib_size[1]} from the principal point."
              "  If wrong, pass --width/--height of the calibration images.")

    w, h = (width or calib_size[0]), (height or calib_size[1])
    if (w, h) != calib_size:
        sx, sy = w / calib_size[0], h / calib_size[1]
        if abs(sx - sy) > 0.01:
            print(f"  WARNING: {w}x{h} has a different shape than the calibration {calib_size} -"
                  " the camera crops differently and the pose will be WRONG. Use the calibration size.")
        K = K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        print(f"  Camera matrix scaled from {calib_size} to {w}x{h}")
    print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  dist={np.round(D[:5], 4)}")
    print(f"  Camera will run at {w}x{h}")
    return K, D, (w, h)


# =====================================================================================
# CAMERA SOURCES
# =====================================================================================
class PiCamera:
    """Camera Module 3 through picamera2. read() -> (gray image, capture time)."""

    def __init__(self, size, fps=30, lens_position=None):
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError as e:
            raise SystemExit("picamera2 not found. Install: sudo apt install -y python3-picamera2\n"
                             "and create the venv with --system-site-packages") from e
        self.w, self.h = size
        info = Picamera2.global_camera_info()
        if not info:
            raise SystemExit("No camera detected by picamera2. Run: python camera_check.py")
        print(f"  Camera: {info[0].get('Model', '?')}, capture size {self.w}x{self.h}")
        try:
            self.cam = Picamera2(0)
        except Exception as e:
            raise SystemExit(f"Cannot open the camera ({e}).\n  Is another program using it? "
                             "Close it (pkill -f rpicam; pkill -f landing.py) or reboot.") from e
        tries = [dict(main={"size": (self.w, self.h), "format": "YUV420"}, controls={"FrameRate": float(fps)},
                      queue=False, buffer_count=3),           # queue=False: always the NEWEST frame, no lag
                 dict(main={"size": (self.w, self.h), "format": "YUV420"}, controls={"FrameRate": float(fps)}),
                 dict(main={"size": (self.w, self.h), "format": "YUV420"}),
                 dict(main={"size": (self.w, self.h), "format": "RGB888"})]
        self.rgb = False
        for i, kw in enumerate(tries):
            try:
                self.cam.configure(self.cam.create_video_configuration(**kw))
                self.cam.start()
                self.rgb = kw["main"]["format"] == "RGB888"
                if i == 0:
                    print("  Camera: newest-frame mode (no lag), 3 buffers")
                if i:
                    print(f"  Camera: started with fallback setup #{i}")
                break
            except Exception as e:
                print(f"  Camera setup #{i} failed: {e}")
                try:
                    self.cam.stop()
                except Exception:
                    pass
        else:
            raise SystemExit("Camera would not start with any setup. Run: python camera_check.py")
        try:
            if lens_position is None:
                self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                print("  Camera: continuous autofocus")
            else:
                self.cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(lens_position)})
                print(f"  Camera: manual focus, LensPosition {lens_position} dioptres")
        except Exception as e:
            print(f"  Camera: focus control not available ({e}) - continuing")
        time.sleep(1.0)
        test = self.read()[0]
        print(f"  Camera: first frame OK {test.shape[1]}x{test.shape[0]}, brightness {test[::8, ::8].mean():.0f}")

    def read(self):
        arr = self.cam.capture_array("main")
        if self.rgb:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY), time.monotonic()
        return arr[:self.h, :self.w], time.monotonic()  # YUV420: first h rows = Y (gray) plane

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass


class OpenCVCamera:
    """USB camera index or a video file (for testing on a PC)."""

    def __init__(self, source, size):
        src = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera/video {source}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, time.monotonic()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), time.monotonic()

    def close(self):
        self.cap.release()


def make_camera(args, size):
    if args.camera == "pi":
        return PiCamera(size, args.fps, args.lens_position)
    return OpenCVCamera(args.camera, size)


# =====================================================================================
# FILTERS / CONTROLLERS
# =====================================================================================
class PIDController:
    """Same PID as the simulation."""

    def __init__(self, kp, ki, kd, output_limit, integral_limit, deadband=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit, self.integral_limit, self.deadband = abs(output_limit), abs(integral_limit), abs(deadband)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error, dt):
        if abs(error) < self.deadband:
            error = 0.0
        dt = max(dt, 1e-3)
        if not self.initialized:
            self.prev_error = error
            self.initialized = True
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, out))


class MarkerKF:
    """Constant-velocity Kalman filter. State: marker position (fwd, right, down) relative to
    the drone in the level, heading-aligned frame [m] and its rate [m/s]."""

    def __init__(self, accel_sigma=1.5):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.q = accel_sigma ** 2
        self.t = None
        self.ok = False

    def reset(self):
        self.ok = False
        self.t = None

    def update(self, z, t, r_sigma):
        if not self.ok or self.t is None or t - self.t > 1.0:
            self.x[:3], self.x[3:] = z, 0.0
            self.P = np.diag([r_sigma ** 2] * 3 + [1.0] * 3)
            self.t, self.ok = t, True
            return True
        dt = max(1e-3, t - self.t)
        self.t = t
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        G = np.zeros((6, 3))
        G[:3, :] = np.eye(3) * 0.5 * dt * dt
        G[3:, :] = np.eye(3) * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        R = np.eye(3) * r_sigma ** 2
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > 25.0:                      # outlier (bad pose) -> skip this measurement
            return False
        Kg = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.P = (np.eye(6) - Kg @ H) @ self.P
        return True


def rot_level_from_body(roll, pitch):
    """Rotation body(FRD) -> level heading frame = Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        return np.degrees([math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])])
    return np.degrees([math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0])


# =====================================================================================
# VISION THREAD
# =====================================================================================
class VisionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_time = 0.0       # monotonic time of the last processed frame
        self.det_time = 0.0         # monotonic time of the last accepted detection
        self.frames = 0
        self.fps = 0.0
        self.detected = False
        self.p = np.zeros(3)        # filtered marker position (fwd, right, down) [m], level frame
        self.v = np.zeros(3)        # filtered rate of p [m/s]  (drone velocity = -v)
        self.raw = np.zeros(3)      # unfiltered level-frame position
        self.cam = np.zeros(3)      # raw tvec in the camera frame
        self.px = (0, 0)
        self.marker_yaw = 0.0
        self.view = None            # last annotated camera image (BGR)
        self.error = ""             # last camera/vision error text
        self.proc_ms = 0.0          # average processing time per frame
        self.brightness = 0.0
        self.others = []            # other marker IDs in view


class Vision(threading.Thread):
    def __init__(self, camera, K, D, args, attitude_at=None):
        super().__init__(daemon=True)
        self.camera, self.K, self.D, self.args = camera, K, D, args
        self.attitude_at = attitude_at or (lambda t: (0.0, 0.0, 0.0))
        self.state = VisionState()
        self.kf = MarkerKF()
        self.running = True
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
        try:
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE   # refined at full res below
            self.detector = cv2.aruco.ArucoDetector(d, params)
            self.detect = lambda g: self.detector.detectMarkers(g)
        except AttributeError:                                    # OpenCV < 4.7
            params = cv2.aruco.DetectorParameters_create()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
            self.detect = lambda g: cv2.aruco.detectMarkers(g, d, parameters=params)
        s = args.marker_size / 2.0
        # corner order of ArUco: top-left, top-right, bottom-right, bottom-left
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        psi = math.radians(args.cam_yaw)
        # image-right and image-up directions in the body (fwd, right) plane
        self.img_right = np.array([-math.sin(psi), math.cos(psi)])
        self.img_up = np.array([math.cos(psi), math.sin(psi)])
        self.writer = None
        self.last_log = 0.0
        self.misses = 0
        self.t_proc = 0.0                    # average processing time per frame [s]

    def cam_to_body(self, t):
        """camera tvec (x right, y down, z forward/down) -> body FRD."""
        horiz = t[0] * self.img_right + (-t[1]) * self.img_up
        fix = getattr(self.args, "dir_fix", None)
        if fix:                                   # from --fix-directions (rotation, then mirror)
            a = math.radians(fix["rotate_deg"])
            f, r = horiz
            horiz = np.array([f * math.cos(a) - r * math.sin(a), f * math.sin(a) + r * math.cos(a)])
            if fix["mirror"]:
                horiz[1] = -horiz[1]
        return np.array([horiz[0], horiz[1], t[2]])

    def run(self):
        import traceback
        n0, t0 = 0, time.monotonic()
        errors = 0
        while self.running:
            try:
                gray, t_cap = self.camera.read()
                if gray is None:
                    time.sleep(0.01)
                    continue
                if gray.ndim == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                gray = np.ascontiguousarray(gray)
                self.process(gray, t_cap - self.args.cam_latency)
            except Exception as e:
                errors += 1
                with self.state.lock:
                    self.state.error = f"{type(e).__name__}: {e}"
                if errors <= 3:
                    print("\n  CAMERA/VISION ERROR (send this to fix it):")
                    traceback.print_exc()
                time.sleep(0.2)
                continue
            n0 += 1
            if time.monotonic() - t0 >= 2.0:
                with self.state.lock:
                    self.state.fps = n0 / (time.monotonic() - t0)
                n0, t0 = 0, time.monotonic()

    def fast_detect(self, gray):
        """Detect on a smaller copy (much faster on the Pi), refine corners on the full image.
        Every 4th frame without a marker also tries full resolution (small marker / very high)."""
        scale = self.args.detect_scale or (0.5 if gray.shape[1] > 800 else 1.0)
        if scale < 0.99 and (self.misses % 4 != 3):
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            corners, ids, rej = self.detect(small)
            corners = [c / scale for c in corners] if ids is not None else corners
        else:
            corners, ids, rej = self.detect(gray)
        if ids is not None:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            refined = []
            for c in corners:
                c = np.ascontiguousarray(c.reshape(-1, 1, 2), dtype=np.float32)
                cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), crit)
                refined.append(c.reshape(1, 4, 2))
            corners = refined
            self.misses = 0
        else:
            self.misses += 1
        return corners, ids, rej

    def process(self, gray, t_frame):
        a = self.args
        t_start = time.perf_counter()
        corners, ids, _ = self.fast_detect(gray)
        found = None
        others = []
        if ids is not None:
            for c, mid in zip(corners, ids.ravel()):
                if int(mid) == a.marker_id and found is None:
                    found = c.reshape(4, 2).astype(np.float64)
                else:
                    others.append(int(mid))
        brightness = float(gray[::8, ::8].mean())
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w = gray.shape[:2]
        cv2.drawMarker(display, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 30, 2)
        if found is None:
            msg = f"NO MARKER (looking for DICT_{a.dict} ID {a.marker_id})"
            if others:
                msg += f" - other IDs seen: {others}"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display, f"brightness {brightness:.0f}  {w}x{h}  {self.state.fps:.0f} fps", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        accepted = False
        if found is not None:
            ok, rvec, tvec = cv2.solvePnP(self.obj, found, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok and tvec[2, 0] > 0.05:
                tc = tvec.ravel()
                roll, pitch, _yaw = self.attitude_at(t_frame)
                p_body = self.cam_to_body(tc)
                p_level = rot_level_from_body(roll, pitch) @ p_body
                rng = float(np.linalg.norm(tc))
                accepted = self.kf.update(p_level, t_frame, r_sigma=0.01 + 0.015 * rng)
                R, _ = cv2.Rodrigues(rvec)
                myaw = rotation_matrix_to_euler_angles(R)[2]
                with self.state.lock:
                    st = self.state
                    st.cam, st.raw = tc, p_level
                    st.px = tuple(found.mean(axis=0).astype(int))
                    st.marker_yaw = myaw
                    if accepted:
                        st.p, st.v = self.kf.x[:3].copy(), self.kf.x[3:].copy()
                        st.det_time = t_frame
                if True:
                    cv2.aruco.drawDetectedMarkers(display, [found.reshape(1, 4, 2).astype(np.float32)],
                                                  np.array([[a.marker_id]]))
                    cv2.drawFrameAxes(display, self.K, self.D, rvec, tvec, a.marker_size * 0.5)
                    p = self.kf.x[:3]
                    cv2.putText(display, f"fwd {p[0]:+.2f} right {p[1]:+.2f} h {p[2]:.2f} m",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        with self.state.lock:
            self.state.frame_time = t_frame
            self.state.frames += 1
            self.state.detected = accepted or (self.state.detected and t_frame - self.state.det_time < IMAGE_TIMEOUT)
            if t_frame - self.state.det_time > 1.0:
                self.kf.reset()
            self.t_proc = 0.9 * self.t_proc + 0.1 * (time.perf_counter() - t_start)
            self.state.proc_ms = self.t_proc * 1000
            self.state.view = display            # shown / saved by the MAIN thread
            self.state.brightness = brightness
            self.state.others = others

        if a.record:
            if self.writer is None:
                hh, ww = display.shape[:2]
                self.writer = cv2.VideoWriter(os.path.expanduser(a.record),
                                              cv2.VideoWriter_fourcc(*"mp4v"), 15, (ww, hh))
            self.writer.write(display)

    def snapshot(self):
        with self.state.lock:
            st = self.state
            fresh = st.det_time > 0 and (time.monotonic() - st.det_time) < IMAGE_TIMEOUT
            return dict(fresh=fresh, p=st.p.copy(), v=st.v.copy(), raw=st.raw.copy(), px=st.px,
                        frames=st.frames, fps=st.fps, frame_time=st.frame_time,
                        brightness=st.brightness, others=list(st.others), error=st.error,
                        proc_ms=st.proc_ms,
                        age=time.monotonic() - st.det_time if st.det_time else 99.0,
                        det_time=st.det_time)

    def show(self):
        """Show the camera window. Call from the MAIN thread only (GUI rule on the Pi)."""
        if not self.args.show:
            return
        with self.state.lock:
            view = self.state.view
            n = self.state.frames
        if view is None or n == getattr(self, "_shown", -1):
            return                                   # nothing new to draw
        self._shown = n
        try:
            if not getattr(self, "_win", False):
                cv2.namedWindow("precision landing - camera", cv2.WINDOW_AUTOSIZE)
                self._win = True
            cv2.imshow("precision landing - camera", fit_to_window(view, self.args.window_size))
            cv2.waitKey(1)
        except Exception as e:
            print(f"  Cannot open the camera window: {str(e).splitlines()[0][:120]}\n"
                  "  -> if OpenCV is the 'headless' version it has no windows: "
                  "pip uninstall -y opencv-contrib-python-headless opencv-python-headless\n"
                  "     then: pip install --no-deps opencv-contrib-python==4.10.0.84\n"
                  f"  Meanwhile see the saved picture: {self.args.save_view}")
            self.args.show = False

    def save_view_every(self, path, period):
        now = time.monotonic()
        if now - getattr(self, "_last_save", 0.0) > period and self.save_view(path):
            self._last_save = now

    def save_view(self, path):
        with self.state.lock:
            view = self.state.view
        if view is not None:
            cv2.imwrite(os.path.expanduser(path), view)
            return True
        return False

    def stop(self):
        self.running = False
        if self.writer is not None:
            self.writer.release()


# =====================================================================================
# DRONE (from arm_takeoff_land.py v7, plus velocity / attitude commands)
# =====================================================================================
class Drone:
    def __init__(self, device, baud):
        print(f"Connecting to {device} @ {baud} ...")
        self.m = mavutil.mavlink_connection(device, baud=baud, source_system=255, source_component=190)
        self._last_hb = 0.0
        self.apm = False
        self.pos = self.gps = self.sys = self.att = self.hud = None
        self.landed_state = M.MAV_LANDED_STATE_UNDEFINED
        self.alt0 = None
        self.yaw0 = 0.0
        self.no_gps = False
        self.texts = []
        self.att_hist = collections.deque(maxlen=300)   # (monotonic time, roll, pitch, yaw)
        self.ui = None                                   # camera window refresh (main thread)
        self._last_ui = 0.0
        self.lock = threading.Lock()
        self.climb_speed = 2.5                           # m/s, PILOT_SPEED_UP (GUIDED_NOGPS scaling)
        self.descent_speed = 2.5                         # m/s, PILOT_SPEED_DN (0 -> = UP)

    # ------------------------------------------------------------ low level
    def _heartbeat(self):
        if time.time() - self._last_hb >= 1.0:
            self.m.mav.heartbeat_send(M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._last_hb = time.time()

    def update(self, timeout=0.1):
        self._heartbeat()
        if self.ui is not None and time.monotonic() - self._last_ui > 0.04:
            self._last_ui = time.monotonic()
            self.ui()
        if timeout and timeout > 0:
            msg = self.m.recv_match(blocking=True, timeout=timeout)
        else:                              # NOTE: recv_match(timeout=0) never reads - use non-blocking
            msg = self.m.recv_match(blocking=False)
        if msg is None:
            return None
        t = msg.get_type()
        if t == "GLOBAL_POSITION_INT":
            self.pos = msg
        elif t == "GPS_RAW_INT":
            self.gps = msg
        elif t == "SYS_STATUS":
            self.sys = msg
        elif t == "ATTITUDE":
            self.att = msg
            self.att_hist.append((time.monotonic(), msg.roll, msg.pitch, msg.yaw))
        elif t == "VFR_HUD":
            self.hud = msg
        elif t == "EXTENDED_SYS_STATE":
            self.landed_state = msg.landed_state
        elif t == "STATUSTEXT":
            print(f"  FC: {msg.text}")
            self.texts.append(msg.text)
        return msg

    def drain(self, max_msgs=50):
        """Read everything waiting, without blocking (used inside the control loop)."""
        for _ in range(max_msgs):
            if self.update(timeout=0) is None:
                break
        self._heartbeat()

    def attitude_at(self, t):
        """Roll, pitch, yaw at monotonic time t (linear interpolation of the ATTITUDE history)."""
        hist = list(self.att_hist)
        if not hist:
            return 0.0, 0.0, 0.0
        if t <= hist[0][0]:
            return hist[0][1:]
        for (t0, r0, p0, y0), (t1, r1, p1, y1) in zip(hist, hist[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / max(1e-6, t1 - t0)
                return r0 + f * (r1 - r0), p0 + f * (p1 - p0), y0
        return hist[-1][1:]

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.update()

    def armed(self):
        return bool(self.m.motors_armed())

    def rel_alt(self):
        return self.pos.relative_alt / 1000 if self.pos else 0.0

    def command(self, cmd, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0, timeout=3.0):
        name = M.enums["MAV_CMD"][cmd].name
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     cmd, 0, p1, p2, p3, p4, p5, p6, p7)
        end = time.time() + timeout
        while time.time() < end:
            msg = self.update()
            if msg is not None and msg.get_type() == "COMMAND_ACK" and msg.command == cmd:
                print(f"  {name}: {M.enums['MAV_RESULT'][msg.result].name}")
                return msg.result == M.MAV_RESULT_ACCEPTED
        print(f"  {name}: no ACK")
        return False

    def set_mode(self, name, timeout=5.0):
        mapping = self.m.mode_mapping() or {}
        if name not in mapping:
            print(f"  Unknown mode {name}")
            return False
        print(f"  Mode -> {name}")
        self.command(M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mapping[name])
        end = time.time() + timeout
        while time.time() < end:
            self.update()
            if self.m.flightmode == name:
                return True
        print(f"  Mode change to {name} not confirmed (now {self.m.flightmode})")
        return False

    # ------------------------------------------------------------ steps (unchanged from v7)
    def connect(self):
        hb = self.m.wait_heartbeat(timeout=15)
        if hb is None:
            raise SystemExit("No heartbeat - check wiring, baud rate, SERIAL2/MAV_1 settings.")
        self.apm = hb.autopilot == M.MAV_AUTOPILOT_ARDUPILOTMEGA
        fw = "ArduPilot" if self.apm else ("PX4" if hb.autopilot == M.MAV_AUTOPILOT_PX4 else str(hb.autopilot))
        print(f"  Heartbeat OK (system {self.m.target_system}), firmware {fw}, mode {self.m.flightmode}")
        rates = {M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 10, M.MAVLINK_MSG_ID_GPS_RAW_INT: 2,
                 M.MAVLINK_MSG_ID_SYS_STATUS: 2, M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE: 5,
                 M.MAVLINK_MSG_ID_ATTITUDE: 50, M.MAVLINK_MSG_ID_VFR_HUD: 10}
        for msg_id, hz in rates.items():
            self.command(M.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, int(1e6 / hz))

    def gps_ok(self):
        return self.gps is not None and self.gps.fix_type >= 3 and self.pos is not None

    def battery_volts(self):
        if self.sys is None or self.sys.voltage_battery in (0, 65535):
            return None
        return self.sys.voltage_battery / 1000

    def preflight(self, min_volt, strict=False, wait=60):
        wait = wait if strict else 10
        print(f"Pre-flight status (waiting up to {wait} s for GPS) ...")
        end = time.time() + wait
        while time.time() < end:
            self.update()
            volt = self.battery_volts()
            if self.gps_ok() and volt is not None and volt >= min_volt:
                break
        volt = self.battery_volts()
        sats = self.gps.satellites_visible if self.gps else 0
        fix = self.gps.fix_type if self.gps else 0
        problems = []
        if not self.gps_ok():
            problems.append(f"no 3D GPS fix (fix type {fix}, {sats} satellites)")
        if volt is None:
            problems.append("no battery reading (battery monitor not set up)")
        elif volt < min_volt:
            problems.append(f"battery low: {volt:.2f} V < {min_volt} V")
        if not problems:
            print(f"  OK: battery {volt:.2f} V, GPS fix {fix}, {sats} satellites")
        for p in problems:
            print(f"  WARNING: {p}" + ("" if strict else " - ignored"))
        if problems and strict:
            raise SystemExit("Pre-flight failed (--strict).")

    def arm(self, force=False, strict=False):
        print("ARMING ..." + ("  (FORCED - pre-arm checks bypassed)" if force else ""))
        if self.apm:
            mode = "GUIDED_NOGPS" if self.no_gps else "GUIDED"
            if not self.set_mode(mode):
                if mode == "GUIDED" and not strict:
                    print("  GUIDED refused (no position) -> switching to no-GPS flight")
                    self.no_gps = True
                    if not self.set_mode("GUIDED_NOGPS"):
                        raise SystemExit("Could not enter GUIDED_NOGPS mode either.")
                else:
                    raise SystemExit(f"Could not enter {mode} mode.")
        if not self.command(M.MAV_CMD_COMPONENT_ARM_DISARM, 1, FORCE if force else 0):
            if force or strict:
                self.spin(0.5)
                self.explain()
                raise SystemExit("Arming rejected - see the FC messages above (PreArm: ...).")
            print("  Arming refused by pre-arm checks -> retrying with FORCE-arm")
            if not self.command(M.MAV_CMD_COMPONENT_ARM_DISARM, 1, FORCE):
                self.spin(0.5)
                self.explain()
                raise SystemExit("Force-arm also rejected - see the FC messages above.")
        end = time.time() + 5
        while time.time() < end and not self.armed():
            self.update()
        if self.armed():
            self.spin(2)
        print(f"  Armed: {self.armed()}")
        if not self.armed():
            self.explain()
        return self.armed()

    def explain(self):
        text = " | ".join(self.texts).lower()
        hints = []
        if "frame class" in text:
            hints.append("FRAME not active: set FRAME_CLASS = 1, FRAME_TYPE = 1 and REBOOT the Pixhawk:\n"
                         "      python set_arming_check.py --frame quad-x --reboot")
        if "radio failsafe" in text or "rc not" in text or "throttle failsafe" in text:
            hints.append("NO RC SIGNAL: switch on the RC transmitter, or add --no-rc (FS_THR_ENABLE = 0).")
        if "safety" in text and "switch" in text:
            hints.append("SAFETY SWITCH: press the red button, or python set_arming_check.py --reboot")
        if hints:
            print("WHY IT DID NOT ARM:")
            for h in hints:
                print("  - " + h)

    def disarm(self):
        print("DISARMING ...")
        self.command(M.MAV_CMD_COMPONENT_ARM_DISARM, 0)

    def takeoff(self, alt):
        if self.apm:
            target = alt
            print(f"TAKE OFF to {alt:.1f} m ...")
        else:
            if self.pos is None:
                print("  PX4 has no position estimate (no GPS) - cannot take off.")
                return False
            target = self.pos.alt / 1000 + alt
            print(f"TAKE OFF to {alt:.1f} m (AMSL {target:.1f} m) ...")
        if not self.command(M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, NAN, NAN, NAN, target):
            return False
        end = time.time() + 40
        last = 0
        while time.time() < end:
            self.update()
            h = self.rel_alt()
            if time.time() - last > 1:
                print(f"  altitude {h:.2f} m")
                last = time.time()
            if h >= 0.9 * alt:
                print("  Take-off altitude reached")
                return True
            if not self.armed():
                print("  Disarmed during take-off!")
                return False
        print("  Take-off timeout")
        return False

    # ------------------------------------------------------------ no-GPS flight (GUIDED_NOGPS)
    def send_attitude(self, thrust, roll=0.0, pitch=0.0):
        """Attitude target. In GUIDED_NOGPS 'thrust' is a CLIMB RATE: 0.5 hold, >0.5 up, <0.5 down."""
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(self.yaw0 / 2), math.sin(self.yaw0 / 2)
        q = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
             cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
        self.m.mav.set_attitude_target_send(0, self.m.target_system, self.m.target_component,
                                            0b00000111, q, 0, 0, 0, float(thrust))

    def baro_alt(self):
        return (self.hud.alt - self.alt0) if (self.hud and self.alt0 is not None) else 0.0

    def fly_nogps(self, seconds, thrust, stop_alt=None):
        end = time.time() + seconds
        last_send = last_print = 0
        while time.time() < end:
            if time.time() - last_send >= 0.1:
                self.send_attitude(thrust)
                last_send = time.time()
            self.update(0.02)
            h = self.baro_alt()
            if time.time() - last_print > 1:
                print(f"  baro height {h:.2f} m")
                last_print = time.time()
            if stop_alt is not None and h >= 0.9 * stop_alt:
                return True
            if not self.armed():
                print("  Disarmed in flight!")
                return False
        return stop_alt is None

    def takeoff_nogps(self, alt, climb=0.65):
        self.yaw0 = self.att.yaw if self.att else 0.0
        self.alt0 = self.hud.alt if self.hud else None
        max_time = alt / ((climb - 0.5) * 2 * self.climb_speed) * 1.5 + 2
        print(f"TAKE OFF (no GPS) to {alt:.1f} m, climb thrust {climb}, max {max_time:.0f} s ...")
        ok = self.fly_nogps(max_time, climb, stop_alt=alt if self.alt0 is not None else None)
        print("  Take-off altitude reached" if ok else "  Take-off: time limit reached")
        return self.armed()

    # ------------------------------------------------------------ precision-landing commands
    def send_velocity_body(self, vx, vy, vz):
        """vx forward, vy right, vz DOWN [m/s], heading-aligned level frame, yaw rate 0."""
        frame = M.MAV_FRAME_BODY_OFFSET_NED if self.apm else M.MAV_FRAME_BODY_NED
        mask = 0b0000_0101_1100_0111          # use velocity + yaw rate (0)
        self.m.mav.set_position_target_local_ned_send(0, self.m.target_system, self.m.target_component,
                                                      frame, mask, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)

    def start_offboard_px4(self):
        """PX4 needs OFFBOARD mode for velocity commands (setpoints must already be streaming)."""
        for _ in range(20):
            self.send_velocity_body(0, 0, 0)
            self.update(0.05)
        ok = self.command(M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6)   # 6 = OFFBOARD
        print("  PX4 OFFBOARD " + ("started" if ok else "REFUSED"))
        return ok

    def land(self, height_fn=None, timeout=60.0):
        """LAND, then make sure the motors stop.
        Touchdown = Pixhawk reports ON_GROUND for 1 s, OR height < 0.30 m and no vertical motion
        (|climb| < 0.10 m/s) for 2 s. Then: normal disarm, and FORCE disarm if the Pixhawk refuses.
        height_fn() may return the camera height above the marker (or None)."""
        print("LANDING ...")
        if self.apm:
            self.set_mode("LAND")
        else:
            self.command(M.MAV_CMD_NAV_LAND, 0, 0, 0, NAN, NAN, NAN, NAN)
        t0 = time.time()
        last = 0.0
        ground_since = None
        still_since = None
        while time.time() - t0 < timeout:
            self.update()
            if not self.armed():
                print("  Landed and disarmed")
                return True
            # best height estimate: camera > GPS relative altitude > barometer
            h = None
            if height_fn is not None:
                try:
                    h = height_fn()
                except Exception:
                    h = None
            if h is None and self.pos is not None:
                h = self.rel_alt()
            if h is None and self.alt0 is not None and self.hud is not None:
                h = self.baro_alt()
            climb = self.hud.climb if self.hud is not None else None
            now = time.time()

            if self.landed_state == M.MAV_LANDED_STATE_ON_GROUND:
                ground_since = ground_since or now
            else:
                ground_since = None
            low_and_still = (h is not None and h < 0.30 and climb is not None and abs(climb) < 0.10)
            if low_and_still:
                still_since = still_since or now
            else:
                still_since = None

            if now - last > 1:
                print(f"  height {h:.2f} m" if h is not None else "  height ?",
                      f"| climb {climb:+.2f} m/s" if climb is not None else "",
                      "| FC: on ground" if self.landed_state == M.MAV_LANDED_STATE_ON_GROUND else "")
                last = now

            reason = None
            if ground_since and now - ground_since >= 1.0:
                reason = "Pixhawk reports ON GROUND"
            elif still_since and now - still_since >= 2.0:
                reason = f"height {h:.2f} m and not moving for 2 s"
            if reason:
                print(f"  Touchdown detected ({reason}) -> DISARM")
                return self.disarm_on_ground()
        print(f"  No touchdown detected after {timeout:.0f} s - still armed! "
              "Land with the RC (switch to LAND / throttle down) or press Ctrl+C and disarm.")
        return False

    def disarm_on_ground(self):
        """Disarm after touchdown: normal first, then forced (21196) if the Pixhawk refuses."""
        self.command(M.MAV_CMD_COMPONENT_ARM_DISARM, 0)
        end = time.time() + 2.0
        while time.time() < end and self.armed():
            self.update()
        if self.armed():
            print("  Normal disarm refused -> FORCE disarm")
            self.command(M.MAV_CMD_COMPONENT_ARM_DISARM, 0, FORCE)
            end = time.time() + 2.0
            while time.time() < end and self.armed():
                self.update()
        print("  Disarmed - motors stopped" if not self.armed() else "  STILL ARMED - disarm with the RC!")
        return not self.armed()


# =====================================================================================
# PRECISION LANDING CONTROLLER
# =====================================================================================
class Lander:
    """Align over the marker, descend while centred, hand over to LAND at the end.
    Produces level-frame velocity commands (vx fwd, vy right, vz down); in attitude mode they
    are turned into roll/pitch angles with the Kalman velocity estimate as feedback."""

    def __init__(self, args, attitude_mode):
        self.args = args
        self.attitude_mode = attitude_mode
        self.pid_x = PIDController(**PID_XY)
        self.pid_y = PIDController(**PID_XY)
        self.pid_z = PIDController(**PID_Z)
        self.max_xy = args.max_xy_speed
        self.trim = np.zeros(2)          # attitude mode: learned (pitch, roll) wind trim [rad]

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

    def compute(self, s, dt):
        """s = vision snapshot (fresh). Returns phase, (vx, vy, vz), offset, height."""
        fwd, right, h = s["p"]
        offset = math.hypot(fwd, right)
        scale = 0.5 if h < 1.0 else 1.0
        vx = max(-self.max_xy, min(self.max_xy, self.pid_x.update(fwd, dt) * scale))
        vy = max(-self.max_xy, min(self.max_xy, self.pid_y.update(right, dt) * scale))
        frac = CENTER_FRAC_FINAL if h < 1.0 else CENTER_FRAC
        tol = max(CENTER_MIN, frac * h)
        if offset < tol or (h < 1.2 and offset < 1.5 * tol):
            ez = h - LAND_TARGET_H
            vz = max(0.0, min(MAX_Z_SPEED, self.pid_z.update(ez, dt) * (0.6 if h < 1.0 else 1.0)))
            if ez > PID_Z["deadband"]:
                vz = max(vz, DESCEND_SPEED_FINAL if h < 1.0 else DESCEND_SPEED)
            return "DESCEND", (vx, vy, vz), offset, h
        self.pid_z.reset()
        return "ALIGN", (vx, vy, 0.0), offset, h

    def to_attitude(self, cmd, s, climb_speed, descent_speed, dt=CONTROL_DT):
        """velocity command -> (roll, pitch, thrust) for GUIDED_NOGPS.
        Velocity feedback comes from the Kalman filter; an integral trim learns steady wind."""
        vx, vy, vz = cmd
        v_drone = -s["v"] if s is not None else np.zeros(3)          # drone velocity, level frame
        if s is not None:                                             # learn the trim only while we can see
            self.trim[0] = max(-MAX_TILT_TRIM, min(MAX_TILT_TRIM, self.trim[0] - TILT_I * (vx - v_drone[0]) * dt))
            self.trim[1] = max(-MAX_TILT_TRIM, min(MAX_TILT_TRIM, self.trim[1] + TILT_I * (vy - v_drone[1]) * dt))
        pitch = -TILT_PER_MS * (vx - v_drone[0]) + self.trim[0]      # nose down = forward
        roll = TILT_PER_MS * (vy - v_drone[1]) + self.trim[1]        # right wing down = right
        pitch = max(-MAX_TILT, min(MAX_TILT, pitch))
        roll = max(-MAX_TILT, min(MAX_TILT, roll))
        speed = descent_speed if vz > 0 else climb_speed
        thrust = 0.5 - vz / (2.0 * max(0.1, speed))
        return roll, pitch, max(0.0, min(1.0, thrust))


def open_log(args):
    folder = os.path.expanduser(args.log_dir)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, time.strftime("pland_%Y%m%d_%H%M%S.csv"))
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "phase", "fresh", "fwd_m", "right_m", "height_m", "offset_m",
                "vx_cmd", "vy_cmd", "vz_cmd", "roll_cmd_deg", "pitch_cmd_deg", "thrust",
                "roll_deg", "pitch_deg", "rel_alt_m", "fps"])
    print(f"Logging to {path}")
    return f, w, path


def precision_land(d, vision, args, climb_to=None):
    """Main landing loop. Returns the last estimated horizontal offset (fwd, right) or None.
    climb_to (no-GPS mode): first CLIMB to this height while the camera keeps the drone over the
    marker - without GPS nothing else stops the drone drifting away during take-off."""
    attitude_mode = d.apm and d.no_gps
    print(f"PRECISION LANDING  ({'attitude / GUIDED_NOGPS' if attitude_mode else 'velocity'} mode)")
    climbing = climb_to is not None
    if climbing:
        d.yaw0 = d.att.yaw if d.att else 0.0
        d.alt0 = d.hud.alt if d.hud else None
        print(f"CLIMB to {climb_to:.1f} m holding over the marker ...")
    if not d.apm and not d.start_offboard_px4():
        return None
    lander = Lander(args, attitude_mode)
    logf, logw, _ = open_log(args)
    t_start = time.monotonic()
    last_seen = time.monotonic()
    last_h = args.alt
    last_offset = None
    last_print = 0.0
    last_t = time.monotonic()
    try:
        while True:
            loop_t = time.monotonic()
            dt = loop_t - last_t
            last_t = loop_t
            d.drain()
            if not d.armed():
                print("  Disarmed - stopping")
                return last_offset
            s = vision.snapshot()
            vision.show()
            cmd, phase, offset, h = (0.0, 0.0, 0.0), "SEARCH", NAN, NAN
            roll_c = pitch_c = 0.0
            thrust = 0.5

            height = s["p"][2] if s["fresh"] else d.baro_alt()
            if climbing and (height >= 0.9 * climb_to or loop_t - t_start > 30):
                print(f"  Climb done at {height:.2f} m")
                climbing = False
                lander.reset()
                t_start = loop_t
            if climbing:
                phase = "CLIMB"
                if s["fresh"]:
                    last_seen = loop_t
                    _ph, cmd, offset, h = lander.compute(s, dt)
                    last_offset = (float(s["p"][0]), float(s["p"][1]))
                cmd = (cmd[0], cmd[1], -0.5)                     # 0.5 m/s up
            elif s["fresh"]:
                last_seen = loop_t
                phase, cmd, offset, h = lander.compute(s, dt)
                last_h = h
                last_offset = (float(s["p"][0]), float(s["p"][1]))
                if h <= args.final_alt:
                    print(f"  Height {h:.2f} m <= {args.final_alt} m, offset {offset:.3f} m -> LAND")
                    return last_offset
            else:
                lander.reset()
                lost = loop_t - last_seen
                commit = max(COMMIT_LAND_ALT, 2.2 * args.marker_size)   # big markers fill the view earlier
                if last_offset is not None and last_h <= commit:
                    print(f"  Marker lost at {last_h:.2f} m (low) -> LAND")
                    return last_offset
                limit = SEARCH_TIMEOUT if last_offset is None else LOST_HOLD_TIMEOUT
                if lost > limit:
                    print(f"  No marker for {lost:.0f} s -> normal LAND")
                    return last_offset
                phase = "SEARCH" if last_offset is None else "LOST"

            if not climbing and loop_t - t_start > args.max_time:
                print(f"  Precision landing took longer than {args.max_time:.0f} s -> LAND")
                return last_offset

            if attitude_mode:
                roll_c, pitch_c, thrust = lander.to_attitude(cmd, s if s["fresh"] else None,
                                                             d.climb_speed, d.descent_speed, dt)
                d.send_attitude(thrust, roll_c, pitch_c)
            else:
                d.send_velocity_body(*cmd)

            att = d.att
            logw.writerow([f"{loop_t - t_start:.3f}", phase, int(s["fresh"]),
                           f"{s['p'][0]:.3f}", f"{s['p'][1]:.3f}", f"{s['p'][2]:.3f}",
                           f"{offset:.3f}", f"{cmd[0]:.3f}", f"{cmd[1]:.3f}", f"{cmd[2]:.3f}",
                           f"{math.degrees(roll_c):.2f}", f"{math.degrees(pitch_c):.2f}", f"{thrust:.3f}",
                           f"{math.degrees(att.roll) if att else 0:.2f}",
                           f"{math.degrees(att.pitch) if att else 0:.2f}",
                           f"{d.rel_alt() if d.pos else d.baro_alt():.2f}", f"{s['fps']:.1f}"])
            if loop_t - last_print > 0.5:
                if s["fresh"]:
                    print(f"  {phase:7s} fwd {s['p'][0]:+.2f} right {s['p'][1]:+.2f} h {h:.2f} m "
                          f"| v cmd fwd {cmd[0]:+.2f} right {cmd[1]:+.2f} down {cmd[2]:+.2f}"
                          + (f" | tilt R {math.degrees(roll_c):+.1f} P {math.degrees(pitch_c):+.1f} thr {thrust:.2f}"
                             if attitude_mode else "") + f" | cam {s['fps']:.0f} fps")
                else:
                    print(f"  {phase}: marker not visible ({loop_t - last_seen:.1f} s) - holding")
                last_print = loop_t
            time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - loop_t)))
    finally:
        logf.close()


# =====================================================================================
# VISION-ONLY BENCH TEST
# =====================================================================================
def fix_directions(vision, args):
    """Guided test. Marker in FRONT of the drone, then to its RIGHT. Finds the rotation (and a
    mirrored image, if any) that makes 'front' = FORWARD and 'right' = RIGHT. Saves directions.json."""
    import json

    def wait_enter(prompt):
        done = threading.Event()
        threading.Thread(target=lambda: (input(prompt), done.set()), daemon=True).start()
        last = 0.0
        while not done.is_set():
            vision.show()
            time.sleep(0.03)
            if time.monotonic() - last > 1.0:
                s = vision.snapshot()
                print(f"     camera sees marker: {'YES' if s['fresh'] else 'no'}", end="\r")
                last = time.monotonic()
        print()

    def measure(label):
        t0 = time.monotonic() + 0.3                 # only frames taken AFTER Enter
        pts, t_end = [], time.monotonic() + 4.0
        last_det = 0.0
        while time.monotonic() < t_end and len(pts) < 20:
            vision.show()
            s = vision.snapshot()
            if s["fresh"] and s["det_time"] > t0 and s["det_time"] != last_det:
                last_det = s["det_time"]
                pts.append(s["raw"][:2] / max(0.05, s["raw"][2]))      # direction only
            time.sleep(0.05)
        if len(pts) < 5:
            print(f"   !! marker not (fully) visible for '{label}' - the WHOLE marker must be in the picture;"
                  " move it a bit closer to the centre or hold the drone higher, then repeat")
            return None
        v = np.median(np.array(pts), axis=0)
        print(f"   measured '{label}': raw ({v[0]:+.2f}, {v[1]:+.2f})  [{len(pts)} frames]")
        return v

    print("\nFIX DIRECTIONS - hold the drone LEVEL about 1 m above the floor (or put it on a box).")
    print("Its FRONT is the side the arrow on the Pixhawk points to.\n")
    while True:
        wait_enter("1) Put the marker IN FRONT of the drone (about 20 cm ahead of the centre, whole marker\n"
                   "   visible in the window). Press Enter ")
        vf = measure("front")
        if vf is None or np.hypot(*vf) < 0.08:
            print("   marker too close to the centre - move it further in front and try again\n")
            continue
        wait_enter("2) Put the marker to the RIGHT of the drone (about 20 cm right of the centre). Press Enter ")
        vr = measure("right")
        if vr is None or np.hypot(*vr) < 0.08:
            print("   try again\n")
            continue
        ang = math.degrees(math.atan2(vf[0] * vr[1] - vf[1] * vr[0], vf @ vr))
        if not 50 < abs(ang) < 130:
            print(f"   the two positions are {abs(ang):.0f} deg apart (should be about 90) - repeat carefully\n")
            continue
        break
    rot = -math.degrees(math.atan2(vf[1], vf[0]))           # turns 'front' onto +forward
    a = math.radians(rot)
    right_after = vr[0] * math.sin(a) + vr[1] * math.cos(a)
    mirror = right_after < 0
    rot_snap = round(rot / 90.0) * 90 % 360 if abs(((rot + 45) % 90) - 45) < 20 else round(rot) % 360
    result = {"rotate_deg": float(rot_snap), "mirror": bool(mirror),
              "measured": {"front": vf.round(3).tolist(), "right": vr.round(3).tolist()},
              "date": time.strftime("%Y-%m-%d %H:%M")}
    path = os.path.expanduser(DIRECTIONS_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    meaning = {0: "camera is correct", 90: "camera turned 90 deg", 180: "FORWARD/BACK and RIGHT/LEFT both reversed",
               270: "camera turned 270 deg"}.get(int(rot_snap), f"camera turned {rot_snap:.0f} deg")
    print(f"\nRESULT: rotate {rot_snap:.0f} deg ({meaning})" + (", image is MIRRORED (left/right flipped)" if mirror else ""))
    print(f"Saved to {path} - landing.py uses it automatically from now on.")
    print("Check it:  python landing.py --vision-only   (marker in front -> FORWARD, right -> RIGHT)")


def print_window_status(args):
    if args.show:
        print(f"Live camera window: ON ({args.window_size[0]}x{args.window_size[1]})  "
              f"| picture also saved every 2 s to {args.save_view}")
    else:
        print("Live camera window: OFF - no desktop found (started over SSH?).\n"
              "  Run it from a terminal ON the Pi desktop, or over SSH type first:  export DISPLAY=:0\n"
              f"  Meanwhile the camera picture is saved every 2 s to {args.save_view}")


def vision_only(vision, args):
    print("\nVISION-ONLY TEST - no drone. Hold the drone (or camera) above the marker.")
    print("Move the marker toward the drone's FRONT: 'fwd' must go +, the drone would fly FORWARD.")
    print("Move the marker to the drone's RIGHT:     'right' must go +, the drone would fly RIGHT.")
    print("If a direction is wrong, rotate with --cam-yaw 90 / 180 / 270.  Ctrl+C to stop.\n")
    print_window_status(args)
    lander = Lander(args, False)
    last = time.monotonic()
    last_print = last_save = 0.0
    while True:
        vision.show()
        time.sleep(0.03)
        now = time.monotonic()
        if now - last_save > 2.0 and vision.save_view(args.save_view):
            last_save = now
        if now - last_print < 0.5:
            continue
        last_print = now
        s = vision.snapshot()
        dt, last = now - last, now
        if s["frames"] == 0:
            print("  NO FRAMES from the camera" + (f" - error: {s['error']}" if s["error"]
                  else " - check the ribbon cable / run: python camera_check.py"))
            continue
        if not s["fresh"]:
            hint = ""
            if s["brightness"] < 25:
                hint = "  <- image almost BLACK: lens cap? too dark?"
            elif s["brightness"] > 235:
                hint = "  <- image almost WHITE: too bright"
            elif s["others"]:
                hint = f"  <- sees marker IDs {s['others']}: wrong --marker-id / --dict?"
            print(f"  no marker  (camera {s['fps']:.0f} fps, {s['proc_ms']:.0f} ms/frame, {s['frames']} frames, "
                  f"brightness {s['brightness']:.0f}){hint}")
            lander.reset()
            continue
        phase, (vx, vy, vz), off, h = lander.compute(s, dt)
        words = []
        if abs(vx) > 0.03:
            words.append("FORWARD" if vx > 0 else "BACK")
        if abs(vy) > 0.03:
            words.append("RIGHT" if vy > 0 else "LEFT")
        if vz > 0.03:
            words.append("DOWN")
        print(f"  marker fwd {s['p'][0]:+.2f} right {s['p'][1]:+.2f} m, height {h:.2f} m, offset {off:.2f} m"
              f" -> {phase}: would fly {' + '.join(words) or 'HOLD'}"
              f"  (vx {vx:+.2f} vy {vy:+.2f} vz {vz:+.2f})  {s['fps']:.0f} fps, {s['proc_ms']:.0f} ms/frame")


# =====================================================================================
# MAIN
# =====================================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser(description="ArUco precision landing (Pi 5 + Camera Module 3 + Pixhawk)")
    ap.add_argument("--marker-size", type=float, default=MARKER_SIZE,
                    help=f"edge of the BLACK marker square in metres (default MARKER_SIZE = {MARKER_SIZE})")
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--dict", default="5X5_50", choices=list(ARUCO_DICTS), help="your marker is 5X5, ID 0")
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--width", type=int, help="camera width (default: calibration size)")
    ap.add_argument("--height", type=int, help="camera height (default: calibration size)")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--camera", default="pi", help="'pi' (Camera Module 3), or a USB index / video file")
    ap.add_argument("--lens-position", type=float, help="manual focus in dioptres (default continuous AF)")
    ap.add_argument("--cam-yaw", type=float, default=None,
                    help="camera rotation: image top vs drone front, clockwise degrees (0/90/180/270). "
                         "Default: value saved by camera_direction.py in ~/drone/cam_yaw.txt, else 0")
    ap.add_argument("--cam-latency", type=float, default=0.05, help="s, frame age when it reaches Python")
    ap.add_argument("--show", action="store_true", default=None,
                    help="show the live camera window (default ON in --vision-only when a desktop exists)")
    ap.add_argument("--no-show", dest="show", action="store_false", help="never open the camera window")
    ap.add_argument("--reverse", dest="reverse", action="store_true", default=None,
                    help="swap forward/back and right/left (adds 180 deg to the camera yaw)")
    ap.add_argument("--no-reverse", dest="reverse", action="store_false",
                    help="do NOT swap directions (ignore REVERSE_DIRECTIONS)")
    ap.add_argument("--window-size", type=int, nargs=2, default=list(WINDOW_SIZE), metavar=("W", "H"),
                    help=f"live window size in pixels (default {WINDOW_SIZE[0]} {WINDOW_SIZE[1]})")
    ap.add_argument("--detect-scale", type=float, default=None,
                    help="detect on a smaller image (faster). Default 0.5 above 800 px wide, 1.0 = off")
    ap.add_argument("--save-view", default="~/drone/camera_view.jpg",
                    help="--vision-only saves the annotated camera picture here every 2 s")
    ap.add_argument("--record", help="save annotated video, e.g. ~/drone/landing.mp4")
    ap.add_argument("--log-dir", default="~/drone/logs")
    ap.add_argument("--vision-only", action="store_true", help="bench test: no drone connection")
    ap.add_argument("--fix-directions", action="store_true",
                    help="guided test: put the marker in FRONT, then to the RIGHT -> saves the correct directions")

    ap.add_argument("--device", default="/dev/ttyAMA0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                    help=f"take-off height in m (default TAKEOFF_ALT = {TAKEOFF_ALT}; 0 = ground arm test)")
    ap.add_argument("--final-alt", type=float, default=FINAL_ALT_TO_LAND,
                    help="camera height above marker where LAND mode takes over")
    ap.add_argument("--max-xy-speed", type=float, default=MAX_XY_SPEED)
    ap.add_argument("--max-time", type=float, default=120, help="s, give up precision landing -> LAND")
    ap.add_argument("--min-volt", type=float, default=11.1)
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--keep-off", action="store_true")
    ap.add_argument("--no-gps", action="store_true")
    ap.add_argument("--no-rc", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    if args.no_gps or args.no_rc:
        args.no_checks = True
    args.dir_fix = None
    if args.cam_yaw is None and not args.fix_directions:
        try:
            import json
            with open(os.path.expanduser(DIRECTIONS_FILE)) as f:
                args.dir_fix = json.load(f)
            args.cam_yaw, args.reverse = 0.0, False
            args.cam_yaw_source = (f"{DIRECTIONS_FILE} (--fix-directions): rotate "
                                   f"{args.dir_fix['rotate_deg']:.0f} deg, mirror {args.dir_fix['mirror']}")
        except (OSError, ValueError, KeyError):
            args.dir_fix = None
    if args.fix_directions:                      # measure the RAW camera, no corrections
        args.cam_yaw, args.reverse, args.cam_yaw_source = 0.0, False, "raw (measuring)"
    if args.dir_fix is not None:
        pass
    elif args.cam_yaw is None:
        args.cam_yaw, src = 0.0, "default"
        try:
            with open(os.path.expanduser("~/drone/cam_yaw.txt")) as f:
                args.cam_yaw, src = float(f.readline().strip()), "~/drone/cam_yaw.txt (camera_direction.py)"
        except (OSError, ValueError):
            pass
        args.cam_yaw_source = src
    else:
        args.cam_yaw_source = "--cam-yaw"
    if args.reverse is None:
        args.reverse = REVERSE_DIRECTIONS if args.dir_fix is None else False
    if args.reverse:
        args.cam_yaw = (args.cam_yaw + 180.0) % 360.0
        args.cam_yaw_source += " + 180 (REVERSE_DIRECTIONS: forward<->back, right<->left)"
    if args.show is None:            # automatic: window in the bench test if there is a desktop
        args.show = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    folder = os.path.dirname(os.path.expanduser(args.save_view))
    if folder:
        os.makedirs(folder, exist_ok=True)
    return args


def main(argv=None):
    args = build_args(argv)
    print(f"precision_land.py {VERSION}")
    print(f"Marker: DICT_{args.dict} ID {args.marker_id}, size {args.marker_size:.3f} m"
          "  <- must equal your printed black square (edit MARKER_SIZE at the top of the file)")
    print(f"Camera yaw: {args.cam_yaw:g} deg  (from {args.cam_yaw_source})")

    K, D, size = load_calibration(args.calib, args.width, args.height)
    if args.width is None and size[0] > 1536:          # full-res calibration -> same shape, smaller
        f = 1536.0 / size[0]
        small = (int(round(size[0] * f / 2)) * 2, int(round(size[1] * f / 2)) * 2)
        print(f"  {size[0]}x{size[1]} is too big for 30 fps - using {small[0]}x{small[1]} (same shape)")
        K, D, size = load_calibration(args.calib, small[0], small[1])
    print("Starting camera ...")
    camera = make_camera(args, size)

    if args.vision_only or args.fix_directions:
        vision = Vision(camera, K, D, args)
        vision.start()
        try:
            if args.fix_directions:
                print_window_status(args)
                fix_directions(vision, args)
                return
            vision_only(vision, args)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            vision.stop()
            camera.close()
        return

    d = Drone(args.device, args.baud)
    d.connect()
    vision = Vision(camera, K, D, args, attitude_at=d.attitude_at)
    vision.start()
    d.ui = lambda: (vision.show(), vision.save_view_every(args.save_view, 2.0))
    print_window_status(args)

    if d.apm and checks is not None:
        frame, _ = checks.read_param(d.m, "FRAME_CLASS")
        if frame == 0:
            raise SystemExit("FRAME_CLASS = 0: frame not set. python set_arming_check.py --frame quad-x --reboot")
        up, _ = checks.read_param(d.m, "PILOT_SPEED_UP")
        dn, _ = checks.read_param(d.m, "PILOT_SPEED_DN")
        if up:
            d.climb_speed = up / 100.0
            d.descent_speed = (dn / 100.0) if dn else d.climb_speed
    if args.no_gps and not d.apm:
        raise SystemExit("--no-gps needs ArduPilot. PX4 cannot fly without a position estimate.")

    # camera must work before we fly
    print("Checking camera ...")
    t_end = time.time() + 10
    while time.time() < t_end and vision.snapshot()["frames"] < 5:
        d.update(0.1)
        vision.show()
    snap = vision.snapshot()
    if snap["frames"] < 5:
        raise SystemExit("Camera gives no frames - not flying." + (f"  Error: {snap['error']}" if snap["error"] else ""))
    print(f"  Camera OK ({snap['fps']:.0f} fps), marker {'VISIBLE' if snap['fresh'] else 'not visible yet'}")

    backup = {}
    if args.no_checks:
        print("WARNING: all checks skipped.")
        d.spin(2)
        if checks is not None:
            print("Turning pre-arm check parameters OFF:")
            backup = checks.checks_off(d.m, skip_reboot_params=True, no_rc=args.no_rc)
        else:
            print("  set_arming_check.py not found - relying on force-arm only")
    else:
        d.preflight(min_volt=args.min_volt, strict=args.strict)

    try:
        d.no_gps = args.no_gps or (d.apm and not d.gps_ok() and not args.strict)
        if d.no_gps:
            print("No GPS -> GUIDED_NOGPS: precision landing in ATTITUDE mode (no position hold, tune carefully)")
        if not d.apm and not d.gps_ok() and args.alt > 0:
            raise SystemExit("PX4 cannot take off without a GPS position.")
        for s in range(5, 0, -1):
            print(f"Starting in {s} ...  (Ctrl+C to abort)")
            d.spin(1)
        fly(d, vision, args)
    except KeyboardInterrupt:
        print("\nCtrl+C")
        if d.armed():
            d.land()
    finally:
        vision.stop()
        camera.close()
        if backup:
            if args.keep_off:
                checks.save_backup("apm" if d.apm else "px4", backup)
                print("Checks left OFF (--keep-off). Run 'python set_arming_check.py --on' when done!")
            else:
                print("Restoring pre-arm check parameters:")
                checks.checks_restore(d.m, backup)
    print("Finished.")


def cam_height(vision):
    """Height above the marker from the camera, or None when the marker is not visible."""
    def f():
        snap = vision.snapshot()
        return float(snap["p"][2]) if snap["fresh"] else None
    return f


def fly(d, vision, args):
    try:
        if not d.arm(force=args.no_checks, strict=args.strict):
            return
        if args.alt <= 0:
            print("Ground test: armed for 5 s, camera check only")
            end = time.time() + 5
            while time.time() < end:
                d.update()
                s = vision.snapshot()
            print(f"  marker {'VISIBLE' if s['fresh'] else 'not visible'}: "
                  f"fwd {s['p'][0]:+.2f} right {s['p'][1]:+.2f} h {s['p'][2]:.2f} m")
            if d.armed():
                d.disarm()
            return
        if d.no_gps:                     # climb with camera position hold, then land
            offset = precision_land(d, vision, args, climb_to=args.alt)
        else:
            if not d.takeoff(args.alt):
                if d.armed():
                    d.land()
                return
            offset = precision_land(d, vision, args)
        d.land(height_fn=cam_height(vision))
        if offset is not None:
            print(f"Last marker offset before LAND: fwd {offset[0]:+.3f} m, right {offset[1]:+.3f} m "
                  f"(|e| = {math.hypot(*offset):.3f} m)")
    except KeyboardInterrupt:
        print("\nCtrl+C -> LAND")
        d.land(height_fn=cam_height(vision))


if __name__ == "__main__":
    main()
