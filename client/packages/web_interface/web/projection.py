import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import yaml
except ImportError:
    yaml = None

VEHICLE_NAME = os.environ.get("VEHICLE_NAME", "bot")
CALIB_DIR = "/data/config/calibrations"

available = False
_Hinv = None
_K = None
_D = None
_W = _Hpx = 0
_sign = 1.0


def _find_yaml(kind):
    for name in (VEHICLE_NAME, "default"):
        path = os.path.join(CALIB_DIR, kind, name + ".yaml")
        if os.path.isfile(path):
            return path
    return None


def load():
    global available, _Hinv, _K, _D, _W, _Hpx, _sign
    if cv2 is None or yaml is None:
        print("[projection] cv2/yaml unavailable - camera overlay disabled")
        return False
    ext_path = _find_yaml("camera_extrinsic")
    int_path = _find_yaml("camera_intrinsic")
    if ext_path is None or int_path is None:
        print("[projection] calibration files not found under {} - camera overlay disabled".format(CALIB_DIR))
        return False
    try:
        with open(ext_path) as f:
            ext = yaml.safe_load(f)
        with open(int_path) as f:
            intr = yaml.safe_load(f)
        H = np.array(ext["homography"], dtype=float).reshape(3, 3)
        _Hinv = np.linalg.inv(H)
        _K = np.array(intr["camera_matrix"]["data"], dtype=float).reshape(3, 3)
        _D = np.array(intr["distortion_coefficients"]["data"], dtype=float).reshape(-1)
        _W = int(intr["image_width"])
        _Hpx = int(intr["image_height"])
        _sign = 1.0 if (_Hinv @ np.array([0.3, 0.0, 1.0]))[2] > 0 else -1.0
    except Exception as exc:
        print("[projection] failed to load calibration ({}) - camera overlay disabled".format(exc))
        return False
    available = True
    print("[projection] loaded extrinsic={} intrinsic={} ({}x{})".format(ext_path, int_path, _W, _Hpx))
    return True


def project(points):
    if not available or not points:
        return []
    try:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        n = pts.shape[0]
        out = [None] * n

        v = (_Hinv @ np.hstack([pts, np.ones((n, 1))]).T).T
        in_front = _sign * v[:, 2] > 1e-6
        if not in_front.any():
            return out
        idx = np.flatnonzero(in_front)
        w = v[idx, 2]
        xn = v[idx, 0] / w
        yn = v[idx, 1] / w
        uu = _K[0, 0] * xn + _K[0, 2]
        vu = _K[1, 1] * yn + _K[1, 2]
        in_frame = (uu >= -3.0 * _W) & (uu <= 4.0 * _W) & (vu >= -3.0 * _Hpx) & (vu <= 4.0 * _Hpx)
        if not in_frame.any():
            return out
        idx = idx[in_frame]
        rays = np.stack([xn[in_frame], yn[in_frame], np.ones(idx.shape[0])], axis=1).reshape(-1, 1, 3)
        img_pts, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), _K, _D)
        img_pts = img_pts.reshape(-1, 2)
        for i, (ud, vd) in zip(idx, img_pts):
            out[int(i)] = [float(ud) / _W, float(vd) / _Hpx]
        return out
    except Exception:
        return []
