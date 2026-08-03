# coding: utf-8

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from pprint import pprint

import cv2
import matplotlib
import numpy as np
import serial
from picamera2 import Picamera2, Preview

def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

HEADLESS = not has_display()

# Use a non-GUI backend. Qt5Agg can conflict with Picamera2's Qt preview and
# crash the process during histogram display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#################### Setting Start ####################

# Change this value before running:
#   "test"   : type commands directly in this terminal
#   "serial" : wait for commands from the configured serial port
RUN_MODE = "test"
# RUN_MODE = "serial"

CAMERA_TUNING_PATH = Path("/home/pi/Desktop/Settings/imx708_tuning.json")
# CAMERA_TUNING_PATH = None
CONFIG_PATH = Path("/home/pi/Desktop/Settings/picam_v3_config.json")
EXAMPLE_CONFIG_PATH = Path("/home/pi/Desktop/Settings/picam_v3_config.example.json")

DEFAULT_CONFIG = {
    "camera_index": None,
    "save_root": "/home/pi/Desktop/Default",
    # "save_root": "/home/pi/Desktop/ssd/pi_data/0424",
    "default_capture_dir": "Session_Default",
    # "bit_depth": "10",
    "bit_depth": "8",
    "preview_size": [1280, 720],
    "exposure_time": 20000,
    "analogue_gain": 1.0,
    "ae_enable": False,
    "awb_enable": False,
    "colour_gains": [1.0, 1.0],
    "enable_preview": True,
    "serial_port": "/dev/serial0",
    "serial_baudrate": 115200,
    "serial_timeout": 1.0,
}

RUNTIME_SETUP_FIELDS = {
    "exposure_time",
    "analogue_gain",
    "ae_enable",
    "awb_enable",
    "colour_gains",
    "default_capture_dir",
    "save_root",
}

RECONFIGURE_SETUP_FIELDS = {
    "bit_depth",
    "preview_size",
}

ALLOWED_SETUP_FIELDS = RUNTIME_SETUP_FIELDS | RECONFIGURE_SETUP_FIELDS


def get_connected_cameras():
    """Return available camera indices using rpicam-hello --list-cameras."""
    try:
        result = subprocess.check_output(
            ["rpicam-hello", "--list-cameras"],
            stderr=subprocess.STDOUT,
        ).decode()

        cams = []
        for line in result.split("\n"):
            if " : " in line:
                idx_str = line.split(":")[0].strip()
                if idx_str.isdigit():
                    cams.append(int(idx_str))
        return cams
    except Exception as exc:
        print(f"[ERROR] Could not list cameras: {exc}")
        return []


def sanitize_filename(name) -> str:
    if name is None:
        return ""
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def normalize_capture_mode(value) -> str:
    text = str(value).strip().lower()
    if text in {"8", "10", "all"}:
        return text
    raise ValueError("bit_depth must be one of: 8, 10, all")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def infer_bayer_pattern_from_raw_format(raw_format: str) -> str:
    s = raw_format.upper()
    for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
        if pattern in s:
            return pattern
    raise ValueError(f"Could not infer Bayer pattern from raw format: {raw_format}")


def cfa_index_map(pattern: str):
    pattern = pattern.upper()
    if pattern == "RGGB":
        return {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)}
    if pattern == "BGGR":
        return {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)}
    if pattern == "GRBG":
        return {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)}
    if pattern == "GBRG":
        return {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)}
    raise ValueError(f"Unsupported Bayer pattern: {pattern}")


def detect_raw_domain(raw: np.ndarray, bit_depth: int) -> str:
    vmax = int(raw.max())
    native_max = (1 << bit_depth) - 1
    if vmax <= native_max + 8:
        return "native"
    return "16bit"


def normalize_raw_array(raw: np.ndarray, raw_config: dict) -> np.ndarray:
    """
    Convert the raw stream buffer into a 2D sensor image.
    Picamera2 may expose SBGGR16 frames as uint8 byte buffers shaped by stride.
    """
    width, height = raw_config["size"]
    expected_shape = (height, width)

    if raw.shape == expected_shape and raw.dtype == np.uint16:
        return raw

    raw_format = str(raw_config.get("format", "")).upper()

    if raw.ndim != 2:
        raise ValueError(f"Unexpected raw array dimensions: shape={raw.shape}, dtype={raw.dtype}")

    if raw_format.endswith("16"):
        raw16 = np.ascontiguousarray(raw).view(np.uint16)
        if raw16.shape != expected_shape:
            raw16 = raw16.reshape(expected_shape)
        return raw16

    if raw.shape == expected_shape:
        return raw

    raise ValueError(
        f"Unsupported raw buffer layout: format={raw_config.get('format')} "
        f"shape={raw.shape} dtype={raw.dtype} expected={expected_shape}"
    )


def get_black_levels(metadata: dict, bit_depth: int, domain: str):
    bl = metadata.get("SensorBlackLevels")

    if bl is None:
        if domain == "16bit":
            if bit_depth == 10:
                v = 64 << (16 - bit_depth)
            elif bit_depth == 12:
                v = 256 << (16 - bit_depth)
            else:
                v = 0
        else:
            if bit_depth == 10:
                v = 64
            elif bit_depth == 12:
                v = 256
            else:
                v = 0
        return float(v), float(v), float(v), float(v)

    bl = tuple(float(x) for x in bl)
    if domain == "16bit":
        return bl

    shift = 16 - bit_depth
    return tuple(x / (1 << shift) for x in bl)


def apply_black_level_correction(
    raw: np.ndarray, metadata: dict, bit_depth: int, pattern: str
):
    domain = detect_raw_domain(raw, bit_depth)
    black_levels = get_black_levels(metadata, bit_depth, domain)

    raw_f = raw.astype(np.float32).copy()
    idx = cfa_index_map(pattern)
    channel_map = {
        "R": black_levels[0],
        "G1": black_levels[1],
        "G2": black_levels[2],
        "B": black_levels[3],
    }

    for channel, (row0, col0) in idx.items():
        raw_f[row0::2, col0::2] -= channel_map[channel]

    raw_f = np.maximum(raw_f, 0.0)

    if domain == "16bit":
        raw_f /= float(1 << (16 - bit_depth))

    return raw_f, domain, black_levels


def demosaic_raw_native_to_rgb16(raw_native: np.ndarray, bit_depth: int, pattern: str):
    raw_native = np.clip(raw_native, 0, (1 << bit_depth) - 1)
    raw16 = np.round(raw_native * (65535.0 / ((1 << bit_depth) - 1))).astype(np.uint16)

    code_map = {
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
    }
    return cv2.cvtColor(raw16, code_map[pattern.upper()])


def save_rgb16_tiff(path: Path, rgb16: np.ndarray):
    bgr16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr16)
    if not ok:
        raise RuntimeError(f"Failed to save TIFF: {path}")


def make_histogram_figure(image: np.ndarray, title="Histogram (RGB)"):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Histogram expects an RGB image, got shape={image.shape}")

    if image.dtype == np.uint16:
        bins = 512
        value_range = [0, 65536]
        x_label = "Intensity (16-bit)"
    else:
        bins = 256
        value_range = [0, 256]
        x_label = "Intensity (8-bit)"
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)
    for idx, color in enumerate(("r", "g", "b")):
        hist = cv2.calcHist([image], [idx], None, [bins], value_range)
        ax.plot(np.log1p(hist), color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("log(1+count)")
    ax.set_xlim([0, bins - 1])
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def choose_mode_full_fov(modes, bit_depth, allow_fallback=True):
    candidates = [mode for mode in modes if int(mode.get("bit_depth", -1)) == int(bit_depth)]

    if not candidates:
        available_depths = sorted(set(int(mode.get("bit_depth", -1)) for mode in modes))
        if not allow_fallback:
            raise RuntimeError(
                f"No sensor mode found for bit depth {bit_depth}. "
                f"Available bit depths: {available_depths}"
            )

        print(
            f"[WARN] No sensor mode found for bit depth {bit_depth}. "
            f"Available bit depths: {available_depths}"
        )
        candidates = modes
        bit_depth = int(max(available_depths))
        print(f"[WARN] Falling back to bit depth {bit_depth}")

    def score(mode):
        crop = mode.get("crop_limits", (0, 0, 0, 0))
        crop_area = crop[2] * crop[3]
        out_area = mode["size"][0] * mode["size"][1]
        same_depth_bonus = 1 if int(mode.get("bit_depth", -1)) == int(bit_depth) else 0
        return (same_depth_bonus, crop_area, out_area)

    return max(candidates, key=score)


def get_available_bit_depths(modes):
    depths = set()
    for mode in modes:
        bit_depth = mode.get("bit_depth")
        if bit_depth is None:
            continue
        try:
            depths.add(int(bit_depth))
        except (TypeError, ValueError):
            continue
    return sorted(depths)


def append_capture_log(log_file: Path, text: str):
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def ensure_capture_dirs(base_dir: Path):
    paths = {
        "log": base_dir / "log",
        "raw": base_dir / "raw",
        "10bit": base_dir / "10bit",
        "8bit": base_dir / "8bit",
        "txt": base_dir / "txt",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def make_output_paths(base_dir: Path, user_name: str):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = sanitize_filename(user_name)
    stem = safe_name if safe_name else timestamp
    capture_dirs = ensure_capture_dirs(base_dir)
    tiff_path = capture_dirs["10bit"] / f"{stem}.tiff"
    png_path = capture_dirs["8bit"] / f"{stem}.png"
    raw_npy_path = capture_dirs["raw"] / f"{stem}.raw.npy"
    meta_json_path = capture_dirs["log"] / f"{stem}.metadata.json"
    log_txt_path = capture_dirs["txt"] / f"{stem}.txt"
    return stem, timestamp, tiff_path, png_path, raw_npy_path, meta_json_path, log_txt_path


def ensure_example_config():
    if EXAMPLE_CONFIG_PATH.exists():
        return
    with open(EXAMPLE_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2, ensure_ascii=False)


def load_config():
    ensure_example_config()

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config = DEFAULT_CONFIG | loaded
        source = CONFIG_PATH
    else:
        config = DEFAULT_CONFIG.copy()
        source = "DEFAULT_CONFIG"
        print(f"[WARN] Config file not found: {CONFIG_PATH}")
        print(f"[INFO] Example config written to: {EXAMPLE_CONFIG_PATH}")

    config["bit_depth"] = normalize_capture_mode(config["bit_depth"])
    config["preview_size"] = tuple(config["preview_size"])
    config["colour_gains"] = tuple(config["colour_gains"])
    return config, source


def print_config_state(config, source):
    print("\nLoaded configuration:")
    print(f"  source              : {source}")
    print(f"  camera_index        : {config['camera_index']}")
    print(f"  save_root           : {config['save_root']}")
    print(f"  default_capture_dir : {config['default_capture_dir']}")
    print(f"  bit_depth           : {config['bit_depth']} (capture mode)")
    print(f"  preview_size        : {config['preview_size']}")
    print(f"  exposure_time       : {config['exposure_time']}")
    print(f"  analogue_gain       : {config['analogue_gain']}")
    print(f"  ae_enable           : {config['ae_enable']}")
    print(f"  awb_enable          : {config['awb_enable']}")
    print(f"  colour_gains        : {config['colour_gains']}")
    print(f"  enable_preview      : {config['enable_preview']}")
    print(f"  serial_port         : {config['serial_port']}")
    print(f"  serial_baudrate     : {config['serial_baudrate']}")
    print(f"  serial_timeout      : {config['serial_timeout']}")
    print(f"  headless            : {HEADLESS}")


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "on", "yes"}:
        return True
    if lowered in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_config_value(key: str, value: str):
    if key == "bit_depth":
        return normalize_capture_mode(value)
    if key in {"camera_index", "exposure_time"}:
        return int(value)
    if key in {"analogue_gain", "serial_timeout"}:
        return float(value)
    if key in {"ae_enable", "awb_enable", "enable_preview"}:
        return parse_bool(value)
    if key in {"preview_size", "colour_gains"}:
        parts = [part.strip() for part in value.split("|")]
        if len(parts) != 2:
            raise ValueError(f"{key} expects two values split by |")
        if key == "preview_size":
            return tuple(int(part) for part in parts)
        return tuple(float(part) for part in parts)
    return value.strip()


def format_response(status: str, message: str, **details):
    payload = {"status": status, "message": message, **details}
    return json.dumps(json_safe(payload), ensure_ascii=False)


class CameraController:
    def __init__(self, config):
        self.config = config.copy()
        if HEADLESS and self.config["enable_preview"]:
            print("[INFO] Headless environment detected. Disabling camera preview.")
            self.config["enable_preview"] = False
        self.save_root = Path(config["save_root"])
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.session_log_path = self.save_root / "_session_log.txt"
        self.camera_index = self.select_camera_index()
        if CAMERA_TUNING_PATH is not None:
            tuning = Picamera2.load_tuning_file(CAMERA_TUNING_PATH)
            print("Tuning configuration imported")
            self.camera = Picamera2(self.camera_index, tuning=tuning)
        else:
            self.camera = Picamera2(self.camera_index)
        self.sensor_modes = list(self.camera.sensor_modes)
        self.available_bit_depths = get_available_bit_depths(self.sensor_modes)

        if not self.available_bit_depths:
            raise RuntimeError("No valid sensor bit depths were reported by the camera.")

        self.selected_mode = None
        self.applied_cfg = None
        self.applied_raw_cfg = None
        self.applied_sensor_cfg = None
        self.applied_raw_format = None
        self.applied_bayer = None
        self.serial = None
        self.preview_started = False

        print("\nAvailable sensor modes:")
        for idx, mode in enumerate(self.sensor_modes):
            print(
                f"[{idx}] size={mode.get('size')} bit_depth={mode.get('bit_depth')} "
                f"fps={mode.get('fps')} format={mode.get('format')} "
                f"unpacked={mode.get('unpacked')} crop_limits={mode.get('crop_limits')}"
            )

    def select_camera_index(self):
        available_cams = get_connected_cameras()
        if not available_cams:
            raise RuntimeError("No cameras detected. Please check connections.")

        configured = self.config.get("camera_index")
        if configured is None:
            selected = available_cams[0]
            print(f"[INFO] camera_index not set. Using first detected camera: {selected}")
            return selected

        configured = int(configured)
        if configured not in available_cams:
            raise RuntimeError(
                f"Configured camera_index={configured} not found. "
                f"Detected cameras: {available_cams}"
            )
        return configured

    def make_preview_config(self):
        capture_mode = normalize_capture_mode(self.config["bit_depth"])
        self.selected_mode = choose_mode_full_fov(
            self.sensor_modes,
            10 if capture_mode == "8" else int(capture_mode if capture_mode != "all" else 10),
        )
        return self.camera.create_preview_configuration(
            main={"size": self.config["preview_size"]},
            raw={"format": self.selected_mode["unpacked"]},
            sensor={
                "output_size": self.selected_mode["size"],
                "bit_depth": self.selected_mode["bit_depth"],
            },
            controls=self.make_controls(),
            queue=False,
            buffer_count=4,
        )

    def capture_8bit_png(self, png_path: Path):
        self.update_current_config_controls()
        still_config = self.camera.create_still_configuration(
            main={"size": self.selected_mode["size"]},
            controls=self.make_controls(),
        )

        self.apply_controls()
        metadata = self.camera.switch_mode_and_capture_file(still_config, str(png_path))
        self.apply_controls()
        self.update_current_config_controls()
        self.applied_cfg = self.camera.camera_configuration()
        self.applied_raw_cfg = self.applied_cfg["raw"]
        self.applied_sensor_cfg = self.applied_cfg.get("sensor", {})
        self.applied_raw_format = self.applied_raw_cfg["format"]
        self.applied_bayer = infer_bayer_pattern_from_raw_format(self.applied_raw_format)

        if not png_path.exists():
            raise RuntimeError(f"Failed to save PNG: {png_path}")
        return png_path, metadata

    def make_controls(self):
        controls = {
            "AeEnable": self.config["ae_enable"],
            "AwbEnable": self.config["awb_enable"],
            "ExposureTime": self.config["exposure_time"],
            "AnalogueGain": self.config["analogue_gain"],
        }
        if not self.config["ae_enable"]:
            exposure_time = int(self.config["exposure_time"])
            controls["FrameDurationLimits"] = (exposure_time, exposure_time)
        if not self.config["awb_enable"]:
            controls["ColourGains"] = self.config["colour_gains"]
        return controls

    def apply_controls(self):
        controls = self.make_controls()
        self.camera.set_controls(controls)

    def update_current_config_controls(self):
        current_config = getattr(self.camera, "camera_config", None)
        if isinstance(current_config, dict):
            current_config["controls"] = self.make_controls()

    def configure_camera(self):
        preview_config = self.make_preview_config()
        self.camera.configure(preview_config)
        self.apply_controls()

        self.applied_cfg = self.camera.camera_configuration()
        self.applied_raw_cfg = self.applied_cfg["raw"]
        self.applied_sensor_cfg = self.applied_cfg.get("sensor", {})
        self.applied_raw_format = self.applied_raw_cfg["format"]
        self.applied_bayer = infer_bayer_pattern_from_raw_format(self.applied_raw_format)

        print("\nApplied camera configuration:")
        pprint(self.applied_cfg)
        print(f"\nApplied Bayer pattern: {self.applied_bayer}")

    def start(self):
        self.configure_camera()
        self.start_preview_safely()
        self.camera.start()
        append_capture_log(
            self.session_log_path,
            f"[SESSION START] {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"camera={self.camera_index} | bit_depth={self.config['bit_depth']} | "
            f"serial={self.config['serial_port']}:{self.config['serial_baudrate']} | "
            f"headless={HEADLESS}",
        )

    def reconfigure(self):
        self.camera.stop()

        self.configure_camera()
        self.camera.start()

    def start_preview_safely(self):
        if not self.config["enable_preview"] or self.preview_started:
            return
        try:
            self.camera.start_preview(Preview.QTGL)
            self.preview_started = True
        except RuntimeError as exc:
            if "event loop is already running" not in str(exc).lower():
                raise
            print(
                "[WARN] Qt preview could not start because an event loop is already "
                "running. Continuing with preview disabled."
            )
            self.config["enable_preview"] = False

    def capture(self, target_dir: str, target_name: str):
        capture_mode = normalize_capture_mode(self.config["bit_depth"])

        dir_name = sanitize_filename(target_dir) or sanitize_filename(
            self.config["default_capture_dir"]
        )
        folder_path = self.save_root / dir_name
        folder_path.mkdir(parents=True, exist_ok=True)

        stem, timestamp, tiff_path, png_path, raw_npy_path, meta_json_path, log_txt_path = (
            make_output_paths(folder_path, target_name)
        )

        metadata = None
        raw = None
        rgb16 = None
        domain = None
        black_levels = None

        if capture_mode in {"10", "all"}:
            with self.camera.captured_request() as request:
                raw = normalize_raw_array(request.make_array("raw"), self.applied_raw_cfg)
                metadata = request.get_metadata()

            sensor_bit_depth = int(self.applied_sensor_cfg.get("bit_depth", 10))
            corrected_native, domain, black_levels = apply_black_level_correction(
                raw=raw,
                metadata=metadata,
                bit_depth=sensor_bit_depth,
                pattern=self.applied_bayer,
            )

            rgb16 = demosaic_raw_native_to_rgb16(
                raw_native=corrected_native,
                bit_depth=sensor_bit_depth,
                pattern=self.applied_bayer,
            )

            save_rgb16_tiff(tiff_path, rgb16)
            np.save(raw_npy_path, raw)

        if capture_mode in {"8", "all"}:
            png_path, png_metadata = self.capture_8bit_png(png_path)
            if capture_mode == "8":
                metadata = png_metadata

        if metadata is None:
            metadata = self.camera.capture_metadata()

        self.apply_controls()
        self.update_current_config_controls()

        with open(meta_json_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(metadata), handle, indent=2, ensure_ascii=False)

        log_lines = [
            f"name                : {stem}",
            f"capture_time         : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"timestamp_tag        : {timestamp}",
            f"camera_index         : {self.camera_index}",
            f"folder               : {folder_path}",
            f"capture_mode         : {capture_mode}",
            f"selected_bit_depth   : {self.config['bit_depth']}",
            f"applied_bit_depth    : {self.applied_sensor_cfg.get('bit_depth')}",
            f"sensor_mode_size     : {self.selected_mode['size']}",
            f"applied_sensor_cfg   : {self.applied_sensor_cfg}",
            f"applied_raw_cfg      : {self.applied_raw_cfg}",
            f"raw_format           : {self.applied_raw_format}",
            f"bayer_pattern        : {self.applied_bayer}",
            f"raw_domain           : {domain}",
            f"black_levels_used    : {black_levels}",
            f"exposure_time_us     : {metadata.get('ExposureTime')}",
            f"analogue_gain        : {metadata.get('AnalogueGain')}",
            f"digital_gain         : {metadata.get('DigitalGain')}",
            f"sensor_timestamp_ns  : {metadata.get('SensorTimestamp')}",
            f"png_path             : {png_path if capture_mode in {'8', 'all'} else None}",
            f"tiff_path            : {tiff_path}",
            f"raw_npy_path         : {raw_npy_path}",
            f"metadata_json_path   : {meta_json_path}",
            f"raw_shape            : {tuple(raw.shape) if raw is not None else None}",
            f"raw_dtype            : {raw.dtype if raw is not None else None}",
            f"rgb16_shape          : {tuple(rgb16.shape) if rgb16 is not None else None}",
            f"rgb16_dtype          : {rgb16.dtype if rgb16 is not None else None}",
        ]
        with open(log_txt_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines) + "\n")

        append_capture_log(
            self.session_log_path,
            f"[CAPTURE] {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"mode={capture_mode} | dir={dir_name} | name={stem} | "
            f"exp_us={metadata.get('ExposureTime')} | again={metadata.get('AnalogueGain')} | "
            f"png={png_path.name if capture_mode in {'8', 'all'} else '-'} | "
            f"tiff={tiff_path.name if capture_mode in {'10', 'all'} else '-'}",
        )

        return {
            "dir": dir_name,
            "name": stem,
            "capture_mode": capture_mode,
            "png_path": str(png_path) if capture_mode in {"8", "all"} else None,
            "tiff_path": str(tiff_path) if capture_mode in {"10", "all"} else None,
            "raw_npy_path": str(raw_npy_path) if capture_mode in {"10", "all"} else None,
            "metadata_json_path": str(meta_json_path),
            "log_txt_path": str(log_txt_path),
            "exposure_time": metadata.get("ExposureTime"),
            "analogue_gain": metadata.get("AnalogueGain"),
        }

    def update_setting(self, key: str, raw_value: str):
        if key not in ALLOWED_SETUP_FIELDS:
            raise KeyError(
                f"Unsupported setup key: {key}. Allowed: {sorted(ALLOWED_SETUP_FIELDS)}"
            )

        value = parse_config_value(key, raw_value)
        self.config[key] = value

        if key == "save_root":
            self.save_root = Path(self.config["save_root"])
            self.save_root.mkdir(parents=True, exist_ok=True)
            self.session_log_path = self.save_root / "_session_log.txt"

        if key in RUNTIME_SETUP_FIELDS:
            self.apply_controls()
            self.update_current_config_controls()
        elif key in RECONFIGURE_SETUP_FIELDS:
            self.reconfigure()

        append_capture_log(
            self.session_log_path,
            f"[SETUP] {time.strftime('%Y-%m-%d %H:%M:%S')} | {key}={self.config[key]}",
        )
        return self.config[key]

    def print_runtime_state(self):
        print("\nRuntime status:")
        print(f"  camera_index      : {self.camera_index}")
        print(f"  save_root         : {self.save_root}")
        print(f"  bit_depth         : {self.config['bit_depth']} (capture mode)")
        print(f"  exposure_time     : {self.config['exposure_time']}")
        print(f"  analogue_gain     : {self.config['analogue_gain']}")
        print(f"  ae_enable         : {self.config['ae_enable']}")
        print(f"  awb_enable        : {self.config['awb_enable']}")
        print(f"  serial_port       : {self.config['serial_port']}")
        print(f"  serial_baudrate   : {self.config['serial_baudrate']}")
        print(f"  headless          : {HEADLESS}")
        print(f"  applied_raw       : {self.applied_raw_format}")
        print(f"  bayer_pattern     : {self.applied_bayer}")

    def save_histogram(self, image: np.ndarray, title="Histogram", stem="histogram"):
        fig = make_histogram_figure(image, title=title)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = self.save_root / f"{sanitize_filename(stem)}_{timestamp}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        append_capture_log(
            self.session_log_path,
            f"[HISTOGRAM] {time.strftime('%Y-%m-%d %H:%M:%S')} | file={output_path}",
        )
        return output_path

    def close(self):
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            if self.preview_started:
                self.camera.stop_preview()
                self.preview_started = False
        except Exception:
            pass
        self.camera.close()
        append_capture_log(
            self.session_log_path,
            f"[SESSION END] {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )


def parse_serial_command(line: str):
    parts = [part.strip() for part in line.split(",")]
    if not parts or not parts[0]:
        raise ValueError("Empty command")

    command = parts[0].lower()
    if command in {"capture", "p"}:
        command = "capture"
        if len(parts) == 2:
            return command, None, parts[1]
        if len(parts) >= 3:
            target_dir = parts[1] or None
            target_name = ",".join(parts[2:]).strip()
            return command, target_dir, target_name
        return command, None, ""
    if command == "setup":
        if len(parts) < 3:
            raise ValueError("setup command format: setup,config,value")
        return command, parts[1], parts[2]
    if command in {"status", "histogram", "quit"}:
        return command, None, None
    raise ValueError(f"Unknown command: {command}")


def print_command_help(mode: str):
    print(f"\n{mode.title()} mode started")
    print("Commands:")
    print("  capture,dir,name")
    print("  capture,name")
    print("  p")
    print("  setup,config,value")
    print("  status")
    print("  histogram")
    print("  quit")


def execute_command(controller: CameraController, line: str):
    command, arg1, arg2 = parse_serial_command(line)

    if command == "capture":
        result = controller.capture(arg1, arg2)
        return format_response(
            "OK",
            "capture completed",
            command=command,
            result=result,
        ), False

    if command == "setup":
        value = controller.update_setting(arg1, arg2)
        return format_response(
            "OK",
            "setup applied",
            command=command,
            config=arg1,
            value=value,
        ), False

    if command == "status":
        return format_response(
            "OK",
            "status",
            command=command,
            config=controller.config,
            applied_raw_format=controller.applied_raw_format,
            applied_bayer=controller.applied_bayer,
        ), False

    if command == "histogram":
        img = controller.camera.capture_array("main")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif not (img.ndim == 3 and img.shape[2] == 3):
            raise RuntimeError(f"Unexpected preview image shape: {img.shape}")

        histogram_path = controller.save_histogram(
            img, title="Preview Histogram", stem="preview_histogram"
        )
        return format_response(
            "OK",
            "histogram saved",
            command=command,
            histogram_path=str(histogram_path),
        ), False

    return format_response("OK", "shutdown requested", command=command), True


def run_serial_mode(controller: CameraController, config: dict):
    ser = serial.Serial(
        config["serial_port"],
        config["serial_baudrate"],
        timeout=config["serial_timeout"],
    )
    print(
        f"\nSerial waiting mode started on "
        f"{config['serial_port']} @ {config['serial_baudrate']} baud"
    )
    print_command_help("serial")

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            print(f"\n[RX] {line}")

            try:
                response, should_quit = execute_command(controller, line)
            except Exception as exc:
                response = format_response("ERROR", str(exc), command=line)
                should_quit = False

            ser.write((response + "\n").encode("utf-8"))
            ser.flush()
            print(f"[TX] {response}")

            if should_quit:
                break
    finally:
        if ser.is_open:
            ser.close()


def run_test_mode(controller: CameraController):
    print_command_help("test")
    print("\nExamples:")
    print("  capture,test_folder,image_001")
    print("  capture,image_001")
    print("  p")

    while True:
        line = input("\nCommand >> ").strip()
        if not line:
            continue

        try:
            response, should_quit = execute_command(controller, line)
        except Exception as exc:
            response = format_response("ERROR", str(exc), command=line)
            should_quit = False

        print(f"[RESULT] {response}")
        if should_quit:
            break


def main():
    run_mode = RUN_MODE.strip().lower()
    if run_mode not in {"test", "serial"}:
        raise ValueError('RUN_MODE must be either "test" or "serial"')

    config, source = load_config()
    print_config_state(config, source)
    print(f"  run_mode           : {run_mode}")

    controller = None

    try:
        controller = CameraController(config)
        controller.start()
        controller.print_runtime_state()

        if run_mode == "test":
            run_test_mode(controller)
        else:
            run_serial_mode(controller, config)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
    finally:
        if controller is not None:
            controller.close()
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
