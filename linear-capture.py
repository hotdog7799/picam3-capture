# coding: utf-8
"""
Picamera2 linear raw capture helper for PSF and calibration captures.

This script saves a reconstruction-friendly linear master next to a convenient
PNG preview:

  raw/<name>.raw_uint.npy          native raw Bayer array before correction
  linear/<name>.linear_float32.npy black-level corrected native-scale Bayer
  planes/<name>.planes_float32.npz R/G1/G2/B half-resolution CFA planes
  preview/<name>.preview.png       contrast-stretched visual check
  tiff/<name>.linear_float32.tiff  best-effort float32 TIFF, if OpenCV supports it
  log/<name>.metadata.json         capture metadata and camera configuration

The PNG preview is not meant for reconstruction. Use the .npy/.npz files for
PSF preprocessing and forward models.

Interactive commands include `hhh`, which searches for the exposure time that
fills a target fraction of the sensor range without clipping, then captures and
logs with it. It replaces the manual "change exposure, press h, look at the
histogram, repeat" loop.
"""

from __future__ import annotations

import argparse
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
from picamera2 import Picamera2, Preview

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TUNING_PATH = Path("/home/pi/Desktop/Settings/imx708_tuning.json")
CONFIG_PATH = Path("/home/pi/Desktop/Settings/picam_v3_config.json")
EXAMPLE_CONFIG_PATH = Path("/home/pi/Desktop/Settings/picam_linear_capture.example.json")
RUN_MODE = "test"

DEFAULT_CONFIG = {
    "camera_index": None,
    "save_root": "/home/pi/Desktop/Default",
    "default_capture_dir": "Session_Default",
    "bit_depth": "10",
    "preview_size": [1280, 720],
    "exposure_time": 200000,
    "analogue_gain": 1.0,
    "ae_enable": False,
    "awb_enable": False,
    "colour_gains": [1.0, 1.0],
    "enable_preview": True,
    "serial_port": "/dev/serial0",
    "serial_baudrate": 115200,
    "serial_timeout": 1.0,
    "warmup": 0.2,
    "tuning_path": str(DEFAULT_TUNING_PATH),
    # Auto exposure search ("hhh") parameters.
    "auto_target_fill": 0.85,
    "auto_fill_low": 0.70,
    "auto_fill_high": 0.92,
    "auto_max_saturated_pixels": 0,
    "auto_max_iterations": 12,
    "auto_exposure_min": 100,
    "auto_exposure_max": 2000000,
    "auto_peak_pixels": 64,
    "auto_save_histogram": True,
}

AUTO_SETUP_FIELDS = {
    "auto_target_fill",
    "auto_fill_low",
    "auto_fill_high",
    "auto_max_saturated_pixels",
    "auto_max_iterations",
    "auto_exposure_min",
    "auto_exposure_max",
    "auto_peak_pixels",
    "auto_save_histogram",
}

RUNTIME_SETUP_FIELDS = {
    "exposure_time",
    "analogue_gain",
    "ae_enable",
    "awb_enable",
    "colour_gains",
    "default_capture_dir",
    "save_root",
    "warmup",
}

RECONFIGURE_SETUP_FIELDS = {
    "bit_depth",
    "preview_size",
    "enable_preview",
}

ALLOWED_SETUP_FIELDS = RUNTIME_SETUP_FIELDS | RECONFIGURE_SETUP_FIELDS | AUTO_SETUP_FIELDS


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


HEADLESS = not has_display()


def sanitize_filename(name: str | None) -> str:
    if not name:
        return time.strftime("%Y%m%d-%H%M%S")
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or time.strftime("%Y%m%d-%H%M%S")


def infer_bayer_pattern(raw_format: str) -> str:
    text = raw_format.upper()
    for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
        if pattern in text:
            return pattern
    raise ValueError(f"Could not infer Bayer pattern from raw format: {raw_format}")


def cfa_index_map(pattern: str) -> dict[str, tuple[int, int]]:
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


def choose_full_fov_mode(sensor_modes: list[dict], bit_depth: int) -> dict:
    candidates = [
        mode for mode in sensor_modes if int(mode.get("bit_depth", -1)) == int(bit_depth)
    ]
    if not candidates:
        available = sorted({int(mode.get("bit_depth", -1)) for mode in sensor_modes})
        raise RuntimeError(f"No {bit_depth}-bit sensor mode found. Available: {available}")

    def score(mode: dict):
        crop = mode.get("crop_limits", (0, 0, 0, 0))
        crop_area = crop[2] * crop[3]
        out_area = mode["size"][0] * mode["size"][1]
        return crop_area, out_area

    return max(candidates, key=score)


def normalize_raw_array(raw: np.ndarray, raw_config: dict) -> np.ndarray:
    width, height = raw_config["size"]
    expected_shape = (height, width)
    raw_format = str(raw_config.get("format", "")).upper()

    if raw.shape == expected_shape and raw.dtype == np.uint16:
        return raw

    if raw.ndim == 2 and raw_format.endswith("16"):
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


def detect_raw_domain(raw: np.ndarray, bit_depth: int) -> str:
    native_max = (1 << bit_depth) - 1
    return "native" if int(raw.max()) <= native_max + 8 else "16bit"


def get_black_levels(metadata: dict, bit_depth: int, domain: str) -> tuple[float, ...]:
    black_levels = metadata.get("SensorBlackLevels")
    if black_levels is None:
        value = 64 if bit_depth == 10 else 256 if bit_depth == 12 else 0
        if domain == "16bit":
            value <<= 16 - bit_depth
        return float(value), float(value), float(value), float(value)

    black_levels = tuple(float(x) for x in black_levels)
    if domain == "16bit":
        return black_levels

    shift = 16 - bit_depth
    return tuple(x / (1 << shift) for x in black_levels)


def apply_black_level(
    raw: np.ndarray, metadata: dict, bit_depth: int, pattern: str
) -> tuple[np.ndarray, str, tuple[float, ...]]:
    domain = detect_raw_domain(raw, bit_depth)
    black_levels = get_black_levels(metadata, bit_depth, domain)
    corrected = raw.astype(np.float32, copy=True)

    level_by_channel = {
        "R": black_levels[0],
        "G1": black_levels[1],
        "G2": black_levels[2],
        "B": black_levels[3],
    }
    for channel, (row0, col0) in cfa_index_map(pattern).items():
        corrected[row0::2, col0::2] -= level_by_channel[channel]

    corrected = np.maximum(corrected, 0.0)
    if domain == "16bit":
        corrected /= float(1 << (16 - bit_depth))
    return corrected, domain, black_levels


def split_cfa_planes(raw_float: np.ndarray, pattern: str) -> dict[str, np.ndarray]:
    idx = cfa_index_map(pattern)
    return {
        channel: raw_float[row0::2, col0::2].astype(np.float32, copy=False)
        for channel, (row0, col0) in idx.items()
    }


def demosaic_linear_preview(
    raw_float: np.ndarray,
    bit_depth: int,
    pattern: str
) -> np.ndarray:

    max_value = float((1 << bit_depth) - 1)

    raw16 = np.clip(raw_float, 0, max_value)
    raw16 = np.round(
        raw16 * (65535.0 / max_value)
    ).astype(np.uint16)

    code_map = {
        "RGGB": cv2.COLOR_BayerRGGB2RGB,
        "BGGR": cv2.COLOR_BayerBGGR2RGB,
        "GRBG": cv2.COLOR_BayerGRBG2RGB,
        "GBRG": cv2.COLOR_BayerGBRG2RGB,
    }

    return cv2.cvtColor(
        raw16,
        code_map[pattern.upper()]
    )


def robust_preview_png(rgb16: np.ndarray) -> np.ndarray:
    rgb = rgb16.astype(np.float32)
    hi = float(np.percentile(rgb, 99.8))
    if hi <= 0:
        hi = float(rgb.max()) if rgb.max() > 0 else 1.0
    png = np.clip(rgb / hi * 255.0, 0, 255).astype(np.uint8)
    return png



def make_rgb_histogram_figure(rgb16: np.ndarray, title: str = "Linear RGB Histogram"):
    if rgb16.ndim != 3 or rgb16.shape[2] != 3:
        raise ValueError(f"RGB histogram expects HxWx3 image, got {rgb16.shape}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, color in enumerate(("r", "g", "b")):
        data = rgb16[:, :, idx].astype(np.float32)
        hist, edges = np.histogram(data, bins=512, range=(0, 65535))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, np.log1p(hist), color=color, label=color.upper())
    ax.set_title(title)
    ax.set_xlabel("Intensity (16-bit linear preview)")
    ax.set_ylabel("log(1 + count)")
    ax.set_xlim([0, 65535])
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def make_cfa_histogram_figure(
    planes: dict[str, np.ndarray],
    bit_depth: int,
    title: str = "Linear CFA Plane Histogram",
    saturation_limits: dict[str, float] | None = None,
):
    max_value = float((1 << bit_depth) - 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {"R": "r", "G1": "g", "G2": "lime", "B": "b"}
    plot_max = max_value
    if saturation_limits:
        plot_max = max(float(v) for v in saturation_limits.values())
    for name in ("R", "G1", "G2", "B"):
        if name not in planes:
            continue
        data = np.clip(planes[name], 0, plot_max).astype(np.float32)
        hist, edges = np.histogram(data, bins=512, range=(0, plot_max))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, np.log1p(hist), color=color_map[name], label=name)
        if saturation_limits and name in saturation_limits:
            ax.axvline(
                saturation_limits[name],
                color=color_map[name],
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
            )
    ax.set_title(title)
    ax.set_xlabel(f"Black-subtracted native signal ({bit_depth}-bit scale)")
    ax.set_ylabel("log(1 + count)")
    ax.set_xlim([0, plot_max])
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)



def raw_scale_for_domain(domain: str, bit_depth: int) -> int:
    return 1 << (16 - bit_depth) if domain == "16bit" else 1


def compute_saturation_stats(
    raw: np.ndarray,
    linear: np.ndarray,
    bit_depth: int,
    domain: str,
    black_levels: tuple[float, ...],
    pattern: str,
) -> dict:
    scale = raw_scale_for_domain(domain, bit_depth)
    sensor_max_raw = ((1 << bit_depth) - 1) * scale
    idx = cfa_index_map(pattern)
    black_by_channel = {
        "R": black_levels[0],
        "G1": black_levels[1],
        "G2": black_levels[2],
        "B": black_levels[3],
    }
    stats = {
        "raw_scale_for_native": scale,
        "sensor_max_raw": int(sensor_max_raw),
        "sensor_max_native": int((1 << bit_depth) - 1),
        "channels": {},
    }
    for channel, (row0, col0) in idx.items():
        raw_plane = raw[row0::2, col0::2]
        linear_plane = linear[row0::2, col0::2]
        black_raw = float(black_by_channel[channel])
        black_native = black_raw / scale
        signal_sat_limit = ((1 << bit_depth) - 1) - black_native
        saturated = raw_plane >= sensor_max_raw
        near_saturated = raw_plane >= sensor_max_raw - scale
        stats["channels"][channel] = {
            "black_level_raw": black_raw,
            "black_level_native": black_native,
            "signal_saturation_limit": signal_sat_limit,
            "raw_max": int(raw_plane.max()),
            "raw_max_native": float(raw_plane.max() / scale),
            "linear_max": float(linear_plane.max()),
            "saturated_pixel_count": int(saturated.sum()),
            "near_saturated_pixel_count": int(near_saturated.sum()),
            "pixel_count": int(raw_plane.size),
            "saturated_fraction": float(saturated.mean()),
            "near_saturated_fraction": float(near_saturated.mean()),
        }
    return stats


def print_saturation_summary(stats: dict) -> None:
    if not stats:
        print("Saturation summary: no stats available")
        return

    print("\nSaturation summary:")
    for channel in ("R", "G1", "G2", "B"):
        channel_stats = stats.get("channels", {}).get(channel)
        if channel_stats is None:
            continue
        saturated_pixels = channel_stats["saturated_pixel_count"]
        near_saturated_pixels = channel_stats["near_saturated_pixel_count"]
        saturated_fraction = channel_stats["saturated_fraction"]
        near_saturated_fraction = channel_stats["near_saturated_fraction"]
        print(
            f"  {channel}: saturated_pixels={saturated_pixels}, near_saturated_pixels={near_saturated_pixels}, "
            f"saturated_fraction={saturated_fraction:.6f}, near_saturated_fraction={near_saturated_fraction:.6f}"
        )
    print()


def saturation_limits_from_stats(stats: dict) -> dict[str, float]:
    return {
        channel: float(values["signal_saturation_limit"])
        for channel, values in stats.get("channels", {}).items()
    }

def robust_plane_peak(plane: np.ndarray, top_k: int) -> float:
    """Median of the top-k pixels: tracks a small bright spot without following hot pixels."""
    flat = plane.reshape(-1)
    k = int(max(1, min(int(top_k), flat.size)))
    if k == 1:
        return float(flat.max())
    top = np.partition(flat, flat.size - k)[flat.size - k :]
    return float(np.median(top))


def measure_exposure_metrics(
    raw: np.ndarray,
    metadata: dict,
    bit_depth: int,
    pattern: str,
    top_k: int,
) -> dict:
    """Per-channel headroom of a probe frame, without writing anything to disk."""
    domain = detect_raw_domain(raw, bit_depth)
    black_levels = get_black_levels(metadata, bit_depth, domain)
    scale = raw_scale_for_domain(domain, bit_depth)
    native_max = float((1 << bit_depth) - 1)
    sensor_max_raw = native_max * scale
    black_by_channel = {
        "R": black_levels[0],
        "G1": black_levels[1],
        "G2": black_levels[2],
        "B": black_levels[3],
    }

    channels = {}
    fill = 0.0
    peak_channel = None
    saturated_pixels = 0
    total_pixels = 0
    for channel, (row0, col0) in cfa_index_map(pattern).items():
        plane = raw[row0::2, col0::2]
        black_native = float(black_by_channel[channel]) / scale
        limit = max(native_max - black_native, 1.0)
        peak_signal = max(robust_plane_peak(plane, top_k) / scale - black_native, 0.0)
        max_signal = max(float(plane.max()) / scale - black_native, 0.0)
        channel_saturated = int((plane >= sensor_max_raw).sum())
        channel_fill = peak_signal / limit
        channels[channel] = {
            "peak_signal": peak_signal,
            "max_signal": max_signal,
            "signal_saturation_limit": limit,
            "fill": channel_fill,
            "max_fill": max_signal / limit,
            "saturated_pixel_count": channel_saturated,
        }
        saturated_pixels += channel_saturated
        total_pixels += int(plane.size)
        if channel_fill > fill:
            fill = channel_fill
            peak_channel = channel

    return {
        "fill": fill,
        "peak_channel": peak_channel,
        "saturated_pixels": saturated_pixels,
        "saturated_fraction": saturated_pixels / total_pixels if total_pixels else 0.0,
        "raw_max": int(raw.max()),
        "bit_depth": bit_depth,
        "raw_domain": domain,
        "channels": channels,
    }


def predict_exposure(
    exposure_us: float,
    metrics: dict,
    target_fill: float,
    max_saturated_pixels: int,
    clipped_streak: int = 0,
) -> float:
    """Linear-sensor prediction of the exposure that lands `fill` on `target_fill`."""
    if metrics["saturated_pixels"] > max_saturated_pixels:
        # Clipped: the measured fill only says "at least the ceiling", so it cannot
        # predict how far down to go. Shrink by how much of the frame is pinned, and
        # escalate while consecutive probes keep clipping so a wildly over-exposed
        # start does not crawl down one small step per frame.
        saturated_fraction = metrics["saturated_fraction"]
        if saturated_fraction > 1e-2:
            ratio = 0.35
        elif saturated_fraction > 1e-3:
            ratio = 0.60
        else:
            ratio = min(0.80, target_fill)
        ratio *= 0.5 ** max(0, clipped_streak - 1)
    else:
        ratio = target_fill / max(metrics["fill"], 1e-4)
    ratio = float(np.clip(ratio, 1.0 / 8.0, 8.0))
    return exposure_us * ratio


def write_float32_tiff_best_effort(path: Path, rgb16: np.ndarray) -> bool:
    rgb_float = rgb16.astype(np.float32) / 65535.0
    bgr_float = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2BGR)
    try:
        return bool(cv2.imwrite(str(path), bgr_float))
    except Exception as exc:
        print(f"[WARN] Could not write float32 TIFF: {exc}")
        return False


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw": root / "raw",
        # "linear": root / "linear",
        "planes": root / "planes",
        "preview": root / "preview",
        # "tiff": root / "tiff",
        "log": root / "log",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def capture_once(args) -> dict:
    tuning_path = Path(args.tuning) if args.tuning else None
    if tuning_path and tuning_path.exists():
        camera = Picamera2(args.camera_index, tuning=Picamera2.load_tuning_file(tuning_path))
        tuning_used = str(tuning_path)
    else:
        camera = Picamera2(args.camera_index)
        tuning_used = None

    try:
        selected_mode = choose_full_fov_mode(list(camera.sensor_modes), args.bit_depth)
        controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(args.exposure_time),
            "AnalogueGain": float(args.analogue_gain),
            "ColourGains": (float(args.colour_gain_r), float(args.colour_gain_b)),
            "FrameDurationLimits": (int(args.exposure_time), int(args.exposure_time)),
        }
        config = camera.create_still_configuration(
            raw={"format": selected_mode["unpacked"]},
            sensor={
                "output_size": selected_mode["size"],
                "bit_depth": selected_mode["bit_depth"],
            },
            controls=controls,
            queue=False,
            buffer_count=2,
        )
        camera.configure(config)
        camera.set_controls(controls)
        camera.start()
        time.sleep(float(args.warmup))

        with camera.captured_request() as request:
            applied_config = camera.camera_configuration()
            raw_config = applied_config["raw"]
            raw = normalize_raw_array(request.make_array("raw"), raw_config)
            metadata = request.get_metadata()

        bit_depth = int(applied_config.get("sensor", {}).get("bit_depth", args.bit_depth))
        pattern = infer_bayer_pattern(raw_config["format"])
        linear, raw_domain, black_levels = apply_black_level(raw, metadata, bit_depth, pattern)
        saturation_stats = compute_saturation_stats(raw, linear, bit_depth, raw_domain, black_levels, pattern)
        planes = split_cfa_planes(linear, pattern)
        rgb16 = demosaic_linear_preview(linear, bit_depth, pattern)
        preview = robust_preview_png(rgb16)

        root = Path(args.output).expanduser()
        dirs = ensure_dirs(root)
        stem = sanitize_filename(args.name)

        raw_path = dirs["raw"] / f"{stem}.raw_uint.npy"
        # linear_path = dirs["linear"] / f"{stem}.linear_float32.npy"
        planes_path = dirs["planes"] / f"{stem}.planes_float32.npz"
        preview_path = dirs["preview"] / f"{stem}.preview.png"
        # tiff_path = dirs["tiff"] / f"{stem}.linear_float32.tiff"
        meta_path = dirs["log"] / f"{stem}.metadata.json"

        np.save(raw_path, raw)
        # np.save(linear_path, linear.astype(np.float32, copy=False))
        np.savez_compressed(planes_path, **planes)
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        # tiff_ok = write_float32_tiff_best_effort(tiff_path, rgb16)

        payload = {
            "name": stem,
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tuning_used": tuning_used,
            "selected_mode": selected_mode,
            "applied_config": applied_config,
            "metadata": metadata,
            "bit_depth": bit_depth,
            "raw_format": raw_config["format"],
            "bayer_pattern": pattern,
            "raw_domain": raw_domain,
            "black_levels_used": black_levels,
            "raw_min_max": [int(raw.min()), int(raw.max())],
            "linear_min_max": [float(linear.min()), float(linear.max())],
            "linear_mean": float(linear.mean()),
            "saturation": saturation_stats,
            "paths": {
                "raw_uint_npy": str(raw_path),
                # "linear_float32_npy": str(linear_path),
                "planes_float32_npz": str(planes_path),
                "preview_png": str(preview_path),
                # "linear_float32_tiff": str(tiff_path) if tiff_ok else None,
                "metadata_json": str(meta_path),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)

        return payload
    finally:
        try:
            camera.stop()
        except Exception:
            pass
        camera.close()



def get_connected_cameras() -> list[int]:
    try:
        result = subprocess.check_output(
            ["rpicam-hello", "--list-cameras"],
            stderr=subprocess.STDOUT,
        ).decode()
        cameras = []
        for line in result.splitlines():
            if " : " in line:
                idx = line.split(":")[0].strip()
                if idx.isdigit():
                    cameras.append(int(idx))
        return cameras
    except Exception as exc:
        print(f"[WARN] Could not list cameras with rpicam-hello: {exc}")
        return [0]


def normalize_bit_depth(value) -> int:
    text = str(value).strip().lower()
    if text == "8":
        print("[INFO] Legacy bit_depth=8 requested; using 10-bit raw for linear capture.")
        return 10
    if text in {"10", "12"}:
        return int(text)
    raise ValueError("bit_depth must be one of: 8, 10, 12. For this script, 8 maps to 10-bit raw.")


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "on", "yes"}:
        return True
    if text in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_config_value(key: str, value: str):
    if key == "bit_depth":
        return str(normalize_bit_depth(value))
    if key in {
        "camera_index",
        "exposure_time",
        "auto_max_saturated_pixels",
        "auto_max_iterations",
        "auto_exposure_min",
        "auto_exposure_max",
        "auto_peak_pixels",
    }:
        return int(value)
    if key in {
        "analogue_gain",
        "serial_timeout",
        "warmup",
        "auto_target_fill",
        "auto_fill_low",
        "auto_fill_high",
    }:
        return float(value)
    if key in {"ae_enable", "awb_enable", "enable_preview", "auto_save_histogram"}:
        return parse_bool(value)
    if key in {"preview_size", "colour_gains"}:
        parts = [part.strip() for part in value.split("|")]
        if len(parts) != 2:
            raise ValueError(f"{key} expects two values split by |")
        if key == "preview_size":
            return [int(part) for part in parts]
        return [float(part) for part in parts]
    return value.strip()


def ensure_example_config():
    if EXAMPLE_CONFIG_PATH.exists():
        return
    EXAMPLE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXAMPLE_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2, ensure_ascii=False)


def load_config() -> tuple[dict, str]:
    ensure_example_config()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config = DEFAULT_CONFIG | loaded
        source = str(CONFIG_PATH)
    else:
        config = DEFAULT_CONFIG.copy()
        source = "DEFAULT_CONFIG"
        print(f"[WARN] Config file not found: {CONFIG_PATH}")
        print(f"[INFO] Example config written to: {EXAMPLE_CONFIG_PATH}")

    config["bit_depth"] = str(normalize_bit_depth(config["bit_depth"]))
    config["preview_size"] = tuple(config["preview_size"])
    config["colour_gains"] = tuple(config["colour_gains"])
    return config, source


def print_config_state(config: dict, source: str):
    print("\nLoaded configuration:")
    print(f"  source              : {source}")
    print(f"  camera_index        : {config['camera_index']}")
    print(f"  save_root           : {config['save_root']}")
    print(f"  default_capture_dir : {config['default_capture_dir']}")
    print(f"  bit_depth           : {config['bit_depth']} raw")
    print(f"  preview_size        : {config['preview_size']}")
    print(f"  exposure_time       : {config['exposure_time']}")
    print(f"  analogue_gain       : {config['analogue_gain']}")
    print(f"  ae_enable           : {config['ae_enable']}")
    print(f"  awb_enable          : {config['awb_enable']}")
    print(f"  colour_gains        : {config['colour_gains']}")
    print(f"  enable_preview      : {config['enable_preview']}")
    print(f"  warmup              : {config['warmup']}")
    print(f"  tuning_path         : {config['tuning_path']}")
    print(f"  headless            : {HEADLESS}")


def append_capture_log(log_file: Path, text: str):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def format_response(status: str, message: str, **details) -> str:
    return json.dumps(json_safe({"status": status, "message": message, **details}), ensure_ascii=False)


class InteractiveCameraController:
    def __init__(self, config: dict):
        self.config = config.copy()
        if HEADLESS and self.config["enable_preview"]:
            print("[INFO] Headless environment detected. Disabling preview.")
            self.config["enable_preview"] = False
        self.save_root = Path(self.config["save_root"]).expanduser()
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.session_log_path = self.save_root / "_linear_session_log.txt"
        self.camera_index = self.select_camera_index()
        self.camera = self.create_camera()
        self.sensor_modes = list(self.camera.sensor_modes)
        self.selected_mode = None
        self.applied_config = None
        self.raw_config = None
        self.bayer_pattern = None
        self.preview_started = False
        self.last_frame_exposure_us = None

        print("\nAvailable sensor modes:")
        for idx, mode in enumerate(self.sensor_modes):
            print(
                f"[{idx}] size={mode.get('size')} bit_depth={mode.get('bit_depth')} "
                f"fps={mode.get('fps')} format={mode.get('format')} "
                f"unpacked={mode.get('unpacked')} crop_limits={mode.get('crop_limits')}"
            )

    def select_camera_index(self) -> int:
        available = get_connected_cameras()
        configured = self.config.get("camera_index")
        if configured is None:
            print(f"[INFO] camera_index not set. Using first detected camera: {available[0]}")
            return int(available[0])
        configured = int(configured)
        if configured not in available:
            raise RuntimeError(f"Configured camera_index={configured} not found. Detected: {available}")
        return configured

    def create_camera(self):
        tuning_path = Path(str(self.config.get("tuning_path") or ""))
        if tuning_path.exists():
            print(f"[INFO] Loading tuning file: {tuning_path}")
            return Picamera2(self.camera_index, tuning=Picamera2.load_tuning_file(tuning_path))
        print("[WARN] Tuning file not found. Using default camera tuning.")
        return Picamera2(self.camera_index)

    def make_controls(self) -> dict:
        controls = {
            "AeEnable": bool(self.config["ae_enable"]),
            "AwbEnable": bool(self.config["awb_enable"]),
            "ExposureTime": int(self.config["exposure_time"]),
            "AnalogueGain": float(self.config["analogue_gain"]),
        }
        if not self.config["ae_enable"]:
            exposure = int(self.config["exposure_time"])
            controls["FrameDurationLimits"] = (exposure, exposure)
        if not self.config["awb_enable"]:
            controls["ColourGains"] = tuple(float(x) for x in self.config["colour_gains"])
        return controls

    def configure_camera(self):
        bit_depth = normalize_bit_depth(self.config["bit_depth"])
        self.selected_mode = choose_full_fov_mode(self.sensor_modes, bit_depth)
        preview_config = self.camera.create_preview_configuration(
            main={"size": tuple(self.config["preview_size"])},
            raw={"format": self.selected_mode["unpacked"]},
            sensor={
                "output_size": self.selected_mode["size"],
                "bit_depth": self.selected_mode["bit_depth"],
            },
            controls=self.make_controls(),
            queue=False,
            buffer_count=4,
        )
        self.camera.configure(preview_config)
        self.camera.set_controls(self.make_controls())
        self.applied_config = self.camera.camera_configuration()
        self.raw_config = self.applied_config["raw"]
        self.bayer_pattern = infer_bayer_pattern(self.raw_config["format"])
        print("\nApplied camera configuration:")
        pprint(self.applied_config)
        print(f"\nApplied Bayer pattern: {self.bayer_pattern}")

    def start(self):
        self.configure_camera()
        if self.config["enable_preview"]:
            self.start_preview_safely()
        self.camera.start()
        # The sensor starts on the configured exposure, so the first settle check
        # already knows which value a stale frame would be carrying.
        self.last_frame_exposure_us = int(self.config["exposure_time"])
        append_capture_log(
            self.session_log_path,
            f"[SESSION START] {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"camera={self.camera_index} | bit_depth={self.config['bit_depth']} | headless={HEADLESS}",
        )

    def start_preview_safely(self):
        if self.preview_started:
            return
        try:
            self.camera.start_preview(Preview.QTGL)
            self.preview_started = True
        except RuntimeError as exc:
            print(f"[WARN] Preview disabled: {exc}")
            self.config["enable_preview"] = False

    def reconfigure(self):
        self.camera.stop()
        self.configure_camera()
        self.camera.start()

    def exposure_limits(self) -> tuple[int, int]:
        """Sensor exposure range, intersected with the configured auto search range."""
        low = int(self.config.get("auto_exposure_min", 100))
        high = int(self.config.get("auto_exposure_max", 2000000))
        try:
            control_min, control_max, _ = self.camera.camera_controls["ExposureTime"]
            low = max(low, int(control_min))
            high = min(high, int(control_max))
        except Exception as exc:
            print(f"[WARN] Could not read ExposureTime limits from camera: {exc}")
        if high < low:
            high = low
        return low, high

    def grab_settled_raw(self, max_frames: int = 10, stall_frames: int = 4) -> tuple[np.ndarray, dict]:
        """Return the first frame actually exposed with the currently configured time.

        Control changes reach the sensor a few frames later, so a frame captured
        right after set_controls() still carries the previous exposure. Frames are
        discarded until the reported ExposureTime matches the request.

        A request the sensor refuses (e.g. below its minimum line time) would never
        match, so a repeated reading is also accepted as the real exposure - but only
        after `stall_frames` identical readings of a value that differs from the one
        in flight before the change. A shorter stall check cannot tell a clamp apart
        from the normal control latency and would accept a stale frame.
        """
        requested = int(self.config["exposure_time"])
        check_exposure = not bool(self.config["ae_enable"])
        tolerance = max(2.0, 0.02 * requested)
        pre_change_exposure = self.last_frame_exposure_us
        previous_actual = None
        stall = 0
        raw = None
        metadata = None
        for _ in range(max(1, int(max_frames))):
            with self.camera.captured_request() as request:
                self.applied_config = self.camera.camera_configuration()
                self.raw_config = self.applied_config["raw"]
                raw = normalize_raw_array(request.make_array("raw"), self.raw_config)
                metadata = request.get_metadata()
            if not check_exposure:
                break
            actual = metadata.get("ExposureTime")
            if actual is None:
                break
            self.last_frame_exposure_us = int(actual)
            if abs(float(actual) - requested) <= tolerance:
                break
            stall = stall + 1 if actual == previous_actual else 1
            previous_actual = actual
            # Without a known pre-change exposure a repeated reading is ambiguous:
            # it may be the clamp, or the old value still in flight. Keep discarding.
            if pre_change_exposure is not None and stall >= int(stall_frames) and actual != pre_change_exposure:
                print(f"[WARN] Sensor clamped exposure to {actual} us for request {requested} us.")
                break
        else:
            if check_exposure and previous_actual is not None:
                print(
                    f"[WARN] Exposure did not settle after {max_frames} frames: "
                    f"requested {requested} us, last frame reported {previous_actual} us."
                )
        return raw, metadata

    def probe_exposure(self, exposure_us: int) -> dict:
        """Apply an exposure, grab one settled frame and measure its headroom."""
        low, high = self.exposure_limits()
        exposure_us = int(np.clip(int(round(exposure_us)), low, high))
        self.config["exposure_time"] = exposure_us
        self.camera.set_controls(self.make_controls())
        time.sleep(float(self.config.get("warmup", 0.2)))
        raw, metadata = self.grab_settled_raw()

        bit_depth = int(
            self.applied_config.get("sensor", {}).get("bit_depth", normalize_bit_depth(self.config["bit_depth"]))
        )
        pattern = infer_bayer_pattern(self.raw_config["format"])
        metrics = measure_exposure_metrics(
            raw, metadata, bit_depth, pattern, int(self.config.get("auto_peak_pixels", 64))
        )
        metrics["requested_exposure_us"] = exposure_us
        metrics["exposure_us"] = int(metadata.get("ExposureTime") or exposure_us)
        metrics["analogue_gain"] = float(metadata.get("AnalogueGain") or self.config["analogue_gain"])
        return metrics

    def auto_exposure_search(self) -> dict:
        """Search the exposure that fills the requested fraction of the sensor range.

        The sensor is linear, so one measurement predicts the next exposure directly.
        A bracket of known under/over exposures is kept alongside the prediction so a
        clipped frame (where the measured fill is meaningless) cannot make it diverge.
        """
        target = float(self.config["auto_target_fill"])
        fill_low = float(self.config["auto_fill_low"])
        fill_high = float(self.config["auto_fill_high"])
        max_saturated = int(self.config["auto_max_saturated_pixels"])
        max_iterations = int(self.config["auto_max_iterations"])
        limit_low, limit_high = self.exposure_limits()

        exposure = int(np.clip(int(self.config["exposure_time"]), limit_low, limit_high))
        under_bound = None  # highest exposure known to be too dim
        over_bound = None  # lowest exposure known to be too bright / clipped
        trace = []
        best = None
        status = "max_iterations"
        clipped_streak = 0

        for iteration in range(1, max_iterations + 1):
            metrics = self.probe_exposure(exposure)
            actual = metrics["exposure_us"]
            fill = metrics["fill"]
            saturated = metrics["saturated_pixels"]
            clipped = saturated > max_saturated
            step = {
                "iteration": iteration,
                "requested_exposure_us": metrics["requested_exposure_us"],
                "exposure_us": actual,
                "fill": fill,
                "peak_channel": metrics["peak_channel"],
                "saturated_pixels": saturated,
                "saturated_fraction": metrics["saturated_fraction"],
                "raw_max": metrics["raw_max"],
            }
            trace.append(step)
            print(
                f"  [auto {iteration}/{max_iterations}] exp={actual} us  fill={fill:.3f}"
                f" ({metrics['peak_channel']})  saturated={saturated}"
            )

            if not clipped and (best is None or fill > best["fill"]) and fill <= fill_high:
                best = {"exposure_us": actual, "fill": fill, "metrics": metrics}

            if not clipped and fill_low <= fill <= fill_high:
                status = "converged"
                best = {"exposure_us": actual, "fill": fill, "metrics": metrics}
                break

            clipped_streak = clipped_streak + 1 if clipped else 0
            if clipped or fill > fill_high:
                over_bound = actual if over_bound is None else min(over_bound, actual)
            else:
                under_bound = actual if under_bound is None else max(under_bound, actual)

            predicted = predict_exposure(actual, metrics, target, max_saturated, clipped_streak)

            # Keep the prediction inside the bracket; fall back to bisection otherwise.
            if under_bound is not None and over_bound is not None and under_bound < over_bound:
                if not (under_bound < predicted < over_bound):
                    predicted = float(np.sqrt(under_bound * over_bound))
            elif over_bound is not None and predicted >= over_bound:
                predicted = over_bound * 0.7
            elif under_bound is not None and predicted <= under_bound:
                predicted = under_bound * 1.4

            next_exposure = int(np.clip(int(round(predicted)), limit_low, limit_high))

            if next_exposure == exposure:
                if fill < fill_low and next_exposure >= limit_high:
                    status = "underexposed_at_exposure_limit"
                elif clipped and next_exposure <= limit_low:
                    status = "saturated_at_exposure_limit"
                else:
                    status = "exposure_step_quantized"
                break
            exposure = next_exposure

        if best is None:
            # Every probe clipped: fall back to the dimmest exposure that was tried.
            fallback = min((step["exposure_us"] for step in trace), default=exposure)
            best = {"exposure_us": int(fallback), "fill": float("nan"), "metrics": None}
            if status == "converged":
                status = "no_unsaturated_exposure_found"

        chosen = int(best["exposure_us"])
        chosen_fill = best["fill"]
        known_fill = chosen_fill == chosen_fill  # False only for NaN
        fill_text = f"{chosen_fill:.4f}" if known_fill else "n/a"
        self.config["exposure_time"] = chosen
        self.camera.set_controls(self.make_controls())

        result = {
            "status": status,
            "chosen_exposure_us": chosen,
            "chosen_fill": chosen_fill if known_fill else None,
            "target_fill": target,
            "accept_band": [fill_low, fill_high],
            "max_saturated_pixels": max_saturated,
            "iterations": len(trace),
            "exposure_search_limits_us": [limit_low, limit_high],
            "analogue_gain": float(self.config["analogue_gain"]),
            "trace": trace,
        }
        append_capture_log(
            self.session_log_path,
            f"[AUTOEXP] {time.strftime('%Y-%m-%d %H:%M:%S')} | status={status} | "
            f"exp_us={chosen} | fill={fill_text} | iterations={len(trace)}",
        )
        return result

    def auto_capture(self, target_dir: str | None, target_name: str | None) -> dict:
        print("\nAuto exposure search started "
              f"(target fill={self.config['auto_target_fill']}, "
              f"band=[{self.config['auto_fill_low']}, {self.config['auto_fill_high']}])")
        started = time.time()
        search = self.auto_exposure_search()
        elapsed = time.time() - started
        search["elapsed_seconds"] = round(elapsed, 3)

        if search["status"] != "converged":
            print(f"[WARN] Auto exposure did not converge: {search['status']}. "
                  f"Capturing with the best exposure found ({search['chosen_exposure_us']} us).")

        stem = target_name or f"auto_{time.strftime('%Y%m%d-%H%M%S')}"
        payload = self.capture(
            target_dir,
            stem,
            with_histogram=bool(self.config.get("auto_save_histogram", True)),
            extra={"auto_exposure": search},
        )
        fill = search["chosen_fill"]
        print(
            f"\nAuto exposure done in {elapsed:.1f}s: exposure={search['chosen_exposure_us']} us, "
            f"fill={f'{fill:.3f}' if fill is not None else 'n/a'}, status={search['status']}"
        )
        print_saturation_summary(payload.get("saturation", {}))
        return payload

    def capture(
        self,
        target_dir: str | None,
        target_name: str | None,
        with_histogram: bool = False,
        extra: dict | None = None,
    ) -> dict:
        dir_name = sanitize_filename(target_dir) if target_dir else sanitize_filename(self.config["default_capture_dir"])
        stem = sanitize_filename(target_name)
        folder = self.save_root / dir_name
        folder.mkdir(parents=True, exist_ok=True)
        dirs = ensure_dirs(folder)

        self.camera.set_controls(self.make_controls())
        time.sleep(float(self.config.get("warmup", 0.2)))
        raw, metadata = self.grab_settled_raw()

        bit_depth = int(self.applied_config.get("sensor", {}).get("bit_depth", normalize_bit_depth(self.config["bit_depth"])))
        pattern = infer_bayer_pattern(self.raw_config["format"])
        linear, raw_domain, black_levels = apply_black_level(raw, metadata, bit_depth, pattern)
        saturation_stats = compute_saturation_stats(raw, linear, bit_depth, raw_domain, black_levels, pattern)
        planes = split_cfa_planes(linear, pattern)
        rgb16 = demosaic_linear_preview(linear, bit_depth, pattern)
        preview = robust_preview_png(rgb16)

        raw_path = dirs["raw"] / f"{stem}.raw_uint.npy"
        # linear_path = dirs["linear"] / f"{stem}.linear_float32.npy"
        planes_path = dirs["planes"] / f"{stem}.planes_float32.npz"
        preview_path = dirs["preview"] / f"{stem}.preview.png"
        # tiff_path = dirs["tiff"] / f"{stem}.linear_float32.tiff"
        meta_path = dirs["log"] / f"{stem}.metadata.json"

        np.save(raw_path, raw)
        # np.save(linear_path, linear.astype(np.float32, copy=False))
        np.savez_compressed(planes_path, **planes)
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        # tiff_ok = write_float32_tiff_best_effort(tiff_path, rgb16)

        rgb_hist_path = None
        cfa_hist_path = None
        if with_histogram:
            # Histograms come from the frame that was just saved, so no extra capture.
            hist_dir = folder / "histogram"
            hist_dir.mkdir(parents=True, exist_ok=True)
            rgb_hist_path = hist_dir / f"{stem}.rgb_histogram.png"
            cfa_hist_path = hist_dir / f"{stem}.cfa_histogram.png"
            save_figure(
                make_rgb_histogram_figure(rgb16, title=f"Linear RGB Histogram: {stem}"),
                rgb_hist_path,
            )
            save_figure(
                make_cfa_histogram_figure(
                    planes,
                    bit_depth,
                    title=f"Linear CFA Histogram: {stem}",
                    saturation_limits=saturation_limits_from_stats(saturation_stats),
                ),
                cfa_hist_path,
            )

        payload = {
            "dir": dir_name,
            "name": stem,
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_index": self.camera_index,
            "selected_mode": self.selected_mode,
            "applied_config": self.applied_config,
            "metadata": metadata,
            "bit_depth": bit_depth,
            "raw_format": self.raw_config["format"],
            "bayer_pattern": pattern,
            "raw_domain": raw_domain,
            "black_levels_used": black_levels,
            "raw_min_max": [int(raw.min()), int(raw.max())],
            "linear_min_max": [float(linear.min()), float(linear.max())],
            "linear_mean": float(linear.mean()),
            "saturation": saturation_stats,
            "paths": {
                "raw_uint_npy": str(raw_path),
                # "linear_float32_npy": str(linear_path),
                "planes_float32_npz": str(planes_path),
                "preview_png": str(preview_path),
                # "linear_float32_tiff": str(tiff_path) if tiff_ok else None,
                "metadata_json": str(meta_path),
                "rgb_histogram_png": str(rgb_hist_path) if rgb_hist_path else None,
                "cfa_histogram_png": str(cfa_hist_path) if cfa_hist_path else None,
            },
        }
        if extra:
            payload.update(extra)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        append_capture_log(
            self.session_log_path,
            f"[CAPTURE] {time.strftime('%Y-%m-%d %H:%M:%S')} | dir={dir_name} | name={stem} | "
            f"exp_us={metadata.get('ExposureTime')} | again={metadata.get('AnalogueGain')} | raw_max={int(raw.max())}",
        )
        return payload

    def save_histogram(self, stem: str | None = None) -> dict:
        stem = sanitize_filename(stem) if stem else f"linear_histogram_{time.strftime('%Y%m%d-%H%M%S')}"
        folder = self.save_root / sanitize_filename(self.config["default_capture_dir"])
        hist_dir = folder / "histogram"
        hist_dir.mkdir(parents=True, exist_ok=True)

        self.camera.set_controls(self.make_controls())
        time.sleep(float(self.config.get("warmup", 0.2)))
        raw, metadata = self.grab_settled_raw()

        bit_depth = int(self.applied_config.get("sensor", {}).get("bit_depth", normalize_bit_depth(self.config["bit_depth"])))
        pattern = infer_bayer_pattern(self.raw_config["format"])
        linear, raw_domain, black_levels = apply_black_level(raw, metadata, bit_depth, pattern)
        saturation_stats = compute_saturation_stats(raw, linear, bit_depth, raw_domain, black_levels, pattern)
        saturation_limits = saturation_limits_from_stats(saturation_stats)
        planes = split_cfa_planes(linear, pattern)
        rgb16 = demosaic_linear_preview(linear, bit_depth, pattern)

        rgb_hist_path = hist_dir / f"{stem}.rgb_histogram.png"
        cfa_hist_path = hist_dir / f"{stem}.cfa_histogram.png"
        meta_path = hist_dir / f"{stem}.histogram.json"

        save_figure(
            make_rgb_histogram_figure(rgb16, title=f"Linear RGB Histogram: {stem}"),
            rgb_hist_path,
        )
        save_figure(
            make_cfa_histogram_figure(
                planes,
                bit_depth,
                title=f"Linear CFA Histogram: {stem}",
                saturation_limits=saturation_limits,
            ),
            cfa_hist_path,
        )

        payload = {
            "name": stem,
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata,
            "bit_depth": bit_depth,
            "raw_format": self.raw_config["format"],
            "bayer_pattern": pattern,
            "raw_domain": raw_domain,
            "black_levels_used": black_levels,
            "raw_min_max": [int(raw.min()), int(raw.max())],
            "linear_min_max": [float(linear.min()), float(linear.max())],
            "linear_mean": float(linear.mean()),
            "saturation": saturation_stats,
            "paths": {
                "rgb_histogram_png": str(rgb_hist_path),
                "cfa_histogram_png": str(cfa_hist_path),
                "metadata_json": str(meta_path),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        append_capture_log(
            self.session_log_path,
            f"[HISTOGRAM] {time.strftime('%Y-%m-%d %H:%M:%S')} | name={stem} | raw_max={int(raw.max())} | rgb={rgb_hist_path}",
        )
        print_saturation_summary(saturation_stats)
        return payload

    def update_setting(self, key: str, raw_value: str):
        if key not in ALLOWED_SETUP_FIELDS:
            raise KeyError(f"Unsupported setup key: {key}. Allowed: {sorted(ALLOWED_SETUP_FIELDS)}")
        value = parse_config_value(key, raw_value)
        self.config[key] = value
        if key == "save_root":
            self.save_root = Path(self.config["save_root"]).expanduser()
            self.save_root.mkdir(parents=True, exist_ok=True)
            self.session_log_path = self.save_root / "_linear_session_log.txt"
        if key in RUNTIME_SETUP_FIELDS:
            self.camera.set_controls(self.make_controls())
        elif key in RECONFIGURE_SETUP_FIELDS:
            self.reconfigure()
        append_capture_log(self.session_log_path, f"[SETUP] {time.strftime('%Y-%m-%d %H:%M:%S')} | {key}={self.config[key]}")
        return self.config[key]

    def print_runtime_state(self):
        print("\nRuntime status:")
        print(f"  camera_index    : {self.camera_index}")
        print(f"  save_root       : {self.save_root}")
        print(f"  default_dir     : {self.config['default_capture_dir']}")
        print(f"  bit_depth       : {self.config['bit_depth']} raw")
        print(f"  exposure_time   : {self.config['exposure_time']}")
        print(f"  analogue_gain   : {self.config['analogue_gain']}")
        print(f"  ae_enable       : {self.config['ae_enable']}")
        print(f"  awb_enable      : {self.config['awb_enable']}")
        print(f"  colour_gains    : {self.config['colour_gains']}")
        print(f"  applied_raw     : {self.raw_config['format'] if self.raw_config else None}")
        print(f"  bayer_pattern   : {self.bayer_pattern}")
        print(
            f"  auto_exposure   : target_fill={self.config['auto_target_fill']} "
            f"band=[{self.config['auto_fill_low']}, {self.config['auto_fill_high']}] "
            f"max_sat_px={self.config['auto_max_saturated_pixels']} "
            f"max_iter={self.config['auto_max_iterations']} "
            f"range_us=[{self.config['auto_exposure_min']}, {self.config['auto_exposure_max']}] "
            f"peak_px={self.config['auto_peak_pixels']} "
            f"histogram={self.config['auto_save_histogram']}"
        )

    def close(self):
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            if self.preview_started:
                self.camera.stop_preview()
        except Exception:
            pass
        self.camera.close()
        append_capture_log(self.session_log_path, f"[SESSION END] {time.strftime('%Y-%m-%d %H:%M:%S')}")


def parse_interactive_command(line: str):
    parts = [part.strip() for part in line.split(",")]
    if not parts or not parts[0]:
        raise ValueError("Empty command")
    command = parts[0].lower()
    if command in {"capture", "p"}:
        if len(parts) == 1:
            return "capture", None, ""
        if len(parts) == 2:
            return "capture", None, parts[1]
        return "capture", parts[1] or None, ",".join(parts[2:]).strip()
    if command == "setup":
        if len(parts) < 3:
            raise ValueError("setup command format: setup,config,value")
        return "setup", parts[1], ",".join(parts[2:]).strip()
    if command == "h":
        if len(parts) >= 2 and parts[1]:
            return command, None, parts[1]
        return command, None, None
    if command in {"hhh", "auto"}:
        if len(parts) == 1:
            return "hhh", None, None
        if len(parts) == 2:
            return "hhh", None, parts[1] or None
        return "hhh", parts[1] or None, ",".join(parts[2:]).strip() or None
    if command in {"status", "quit"}:
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
    print("  h              (histogram only, current exposure)")
    print("  h,name")
    print("  hhh            (auto exposure search + capture + histogram)")
    print("  hhh,name")
    print("  hhh,dir,name")
    print("  quit")


def execute_interactive_command(controller: InteractiveCameraController, line: str):
    command, arg1, arg2 = parse_interactive_command(line)
    if command == "capture":
        result = controller.capture(arg1, arg2)
        return format_response("OK", "capture completed", command=command, result=result), False
    if command == "setup":
        value = controller.update_setting(arg1, arg2)
        return format_response("OK", "setup applied", command=command, config=arg1, value=value), False
    if command == "status":
        controller.print_runtime_state()
        return format_response("OK", "status", command=command, config=controller.config), False
    if command == "h":
        result = controller.save_histogram(arg2)
        return format_response("OK", "histogram saved", command=command, result=result), False
    if command == "hhh":
        result = controller.auto_capture(arg1, arg2)
        return (
            format_response(
                "OK",
                "auto exposure capture completed",
                command=command,
                auto_exposure=result.get("auto_exposure"),
                result=result,
            ),
            False,
        )
    return format_response("OK", "shutdown requested", command=command), True


def run_test_mode(controller: InteractiveCameraController):
    print_command_help("test")
    print("\nExamples:")
    print("  capture,red-led,04p0")
    print("  capture,04p0")
    print("  setup,exposure_time,200000")
    print("  h")
    print("  hhh,red-led,04p0")
    print("  setup,auto_target_fill,0.9")
    while True:
        line = input("\nCommand >> ").strip()
        if not line:
            continue
        try:
            response, should_quit = execute_interactive_command(controller, line)
        except Exception as exc:
            response = format_response("ERROR", str(exc), command=line)
            should_quit = False
        print(f"[RESULT] {response}")
        if should_quit:
            break


def run_interactive_main():
    config, source = load_config()
    print_config_state(config, source)
    controller = None
    try:
        controller = InteractiveCameraController(config)
        controller.start()
        controller.print_runtime_state()
        run_test_mode(controller)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
    finally:
        if controller is not None:
            controller.close()
        print("Camera released. Goodbye.")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Linear raw Picamera2 capture. Run with no arguments for the "
            "interactive mode compatible with picam_v4.py, or pass --output "
            "and --name for a single capture."
        )
    )
    parser.add_argument("--output", help="Output session directory for one-shot mode")
    parser.add_argument("--name", help="Capture name for one-shot mode, e.g. 06p5")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--bit-depth", type=int, default=10, choices=(10, 12))
    parser.add_argument("--exposure-time", type=int, default=200000)
    parser.add_argument("--analogue-gain", type=float, default=1.0)
    parser.add_argument("--colour-gain-r", type=float, default=1.0)
    parser.add_argument("--colour-gain-b", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument(
        "--tuning",
        default=str(DEFAULT_TUNING_PATH) if DEFAULT_TUNING_PATH.exists() else "",
        help="Optional Picamera2 tuning JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None and args.name is None:
        run_interactive_main()
        return
    if args.output is None or args.name is None:
        raise SystemExit("--output and --name must be provided together for one-shot mode. Omit both for interactive mode.")
    result = capture_once(args)
    print(json.dumps(json_safe(result["paths"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
