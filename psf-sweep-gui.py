# coding: utf-8
"""
psf-sweep-gui.py - fullscreen GUI for PSF captures with the KDC101 / Z825B LED stage.

Layout (designed for the 1280x720 headless desktop, scales up)
  top-left     live camera preview, optional 4x4 green alignment grid
  bottom-left  live CFA histogram (R, G1, G2, B) with saturation markers
  right        stage control | camera control | save location + capture | sweep / log tabs

Everything that touches the camera or the stage for more than a few milliseconds runs in a
single worker thread, so the preview and the GUI never freeze. STOP (stage) and Abort (sweep)
stay active while a job runs.

Capture processing, auto exposure ("hhh") and file formats come from burst-linear-capture.py;
stage control from kdc101.py; sweep helpers from psf-sweep-capture.py.

Run:   uv run psf-sweep-gui.py [--windowed] [--max-mm 16] [--save-root DIR]
Keys:  F11 toggles fullscreen, Esc leaves fullscreen.
Test:  uv run psf-sweep-gui.py --selftest   (capture, jog, 3-point sweep, then exits)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    # The picamera2 GL preview widget needs an X11 connection (Xwayland is fine).
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt5.QtCore import Qt, pyqtSignal  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kdc101 import KDC101, KDC101Error  # noqa: E402

GUI_CONFIG_PATH = Path("/home/pi/Desktop/Settings/psf_gui_config.json")

FORMAT_OPTIONS = [
    ("raw_npy", "raw .npy", "native uint16 Bayer array (raw/<name>.raw_uint.npy)"),
    ("planes_npz", "planes .npz", "black-level corrected R/G1/G2/B float32 planes (planes/<name>.planes_float32.npz)"),
    ("preview_png", "preview .png", "contrast-stretched 8-bit preview (preview/<name>.preview.png)"),
    ("linear_npy", "linear .npy", "black-level corrected float32 Bayer array (linear/<name>.linear_float32.npy)"),
    ("tiff_float32", "tiff f32", "float32 RGB TIFF (tiff/<name>.linear_float32.tiff)"),
    ("histogram_png", "hist .png", "RGB + CFA histogram figures (histogram/<name>.*_histogram.png)"),
]

GUI_DEFAULTS = {
    "save_root": "",
    "capture_dir": "session",
    "capture_name": "psf",
    "max_mm": 16.0,
    "fine_step_mm": 0.1,
    "coarse_step_mm": 1.0,
    "wd_memo": "",
    "grid": True,
    "live_hist": True,
    "formats": {
        "raw_npy": True,
        "planes_npz": True,
        "preview_png": True,
        "linear_npy": False,
        "tiff_float32": False,
        "histogram_png": False,
    },
    "sweep": {
        "start": 16.0,
        "stop": 0.0,
        "step": 1.0,
        "settle": 0.5,
        "burst": 3,
        "auto_exposure": True,
        "dir": "",
        "prefix": "",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gui_config() -> dict:
    cfg = json.loads(json.dumps(GUI_DEFAULTS))
    if GUI_CONFIG_PATH.exists():
        try:
            loaded = json.loads(GUI_CONFIG_PATH.read_text(encoding="utf-8"))
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                    cfg[key].update(value)
                else:
                    cfg[key] = value
        except Exception as exc:
            print(f"[WARN] Could not read {GUI_CONFIG_PATH}: {exc}")
    return cfg


def save_gui_config(cfg: dict):
    GUI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUI_CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================================ worker
class Worker(QtCore.QThread):
    """Owns every slow camera / stage operation. Jobs are callables run in order."""

    log = pyqtSignal(str)
    hist = pyqtSignal(object)
    position = pyqtSignal(object)
    busy = pyqtSignal(bool)
    progress = pyqtSignal(int, int, str)
    exposure = pyqtSignal(int)
    failed = pyqtSignal(str)
    job_done = pyqtSignal(str)

    def __init__(self, cam, controller, motor, sweep_mod, hist_period: float = 0.5):
        super().__init__()
        self.cam = cam
        self.controller = controller
        self.motor = motor
        self.sweep_mod = sweep_mod
        self.jobs: queue.Queue = queue.Queue()
        self.abort = threading.Event()
        self.live_hist = True
        self.hist_period = hist_period
        self.is_busy = False
        self._stop = False
        self._last_hist = 0.0
        self._last_pos = 0.0
        self._pos_errors = 0

    # ------------------------------------------------------------ plumbing
    def submit(self, name: str, fn):
        self.jobs.put((name, fn))

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                name, fn = self.jobs.get(timeout=0.25)
            except queue.Empty:
                self._idle()
                continue
            self.is_busy = True
            self.busy.emit(True)
            self.abort.clear()
            try:
                fn()
                self.job_done.emit(name)
            except Exception as exc:
                traceback.print_exc()
                self.failed.emit(f"{name}: {exc}")
            finally:
                self.is_busy = False
                self.busy.emit(False)

    def _idle(self):
        now = time.monotonic()
        if self.motor is not None and now - self._last_pos > 0.3:
            self._last_pos = now
            try:
                self.position.emit(self.motor.status())
                self._pos_errors = 0
            except Exception as exc:
                self._pos_errors += 1
                if self._pos_errors in (1, 10, 100):
                    self.log.emit(f"[STAGE] status read failed: {exc}")
        if self.live_hist and now - self._last_hist > self.hist_period:
            try:
                raw, meta = self._grab_raw()
                self.hist.emit(self._histogram(raw, meta, "live"))
            except Exception as exc:
                self.log.emit(f"[HIST] {exc}")
            self._last_hist = time.monotonic()

    # ------------------------------------------------------------- camera
    def _grab_raw(self):
        ctrl = self.controller
        with ctrl.camera.captured_request() as request:
            ctrl.applied_config = ctrl.camera.camera_configuration()
            ctrl.raw_config = ctrl.applied_config["raw"]
            raw = self.cam.normalize_raw_array(request.make_array("raw"), ctrl.raw_config)
            meta = request.get_metadata()
        return raw, meta

    def _bit_depth_pattern(self):
        ctrl = self.controller
        bit_depth = int(ctrl.applied_config.get("sensor", {}).get("bit_depth",
                        self.cam.normalize_bit_depth(ctrl.config["bit_depth"])))
        pattern = self.cam.infer_bayer_pattern(ctrl.raw_config["format"])
        return bit_depth, pattern

    def _histogram(self, raw, meta, label: str) -> dict:
        cam = self.cam
        bit_depth, pattern = self._bit_depth_pattern()
        domain = cam.detect_raw_domain(raw, bit_depth)
        full = ((1 << bit_depth) - 1) * cam.raw_scale_for_domain(domain, bit_depth)
        bins = 128
        channels = {}
        for ch, (r, c) in cam.cfa_index_map(pattern).items():
            plane = raw[r::2, c::2]
            counts, _ = np.histogram(plane[::2, ::2], bins=bins, range=(0, full + 1))
            channels[ch] = {
                "counts": counts,
                "max": int(plane.max()),
                "saturated": int(np.count_nonzero(plane >= full)),
            }
        return {
            "label": label,
            "bins": bins,
            "full_scale": int(full),
            "channels": channels,
            "exposure_us": meta.get("ExposureTime"),
            "gain": meta.get("AnalogueGain"),
        }

    def auto_exposure(self) -> dict:
        search = self.controller.run_auto_exposure()
        self.exposure.emit(int(search["chosen_exposure_us"]))
        self.log.emit("[AE] " + self.controller.auto_exposure_summary(search))
        return search

    def save_frame(self, folder: Path, stem: str, formats: dict, extra: dict) -> dict:
        cam = self.cam
        ctrl = self.controller
        ctrl.camera.set_controls(ctrl.make_controls())
        time.sleep(float(ctrl.config.get("warmup", 0.2)))
        raw, metadata = ctrl.grab_settled_raw()

        bit_depth, pattern = self._bit_depth_pattern()
        linear, raw_domain, black_levels = cam.apply_black_level(raw, metadata, bit_depth, pattern)
        saturation_stats = cam.compute_saturation_stats(raw, linear, bit_depth, raw_domain, black_levels, pattern)
        planes = cam.split_cfa_planes(linear, pattern)
        need_rgb = formats.get("preview_png") or formats.get("tiff_float32") or formats.get("histogram_png")
        rgb16 = cam.demosaic_linear_preview(linear, bit_depth, pattern) if need_rgb else None

        def subdir(name: str) -> Path:
            path = folder / name
            path.mkdir(parents=True, exist_ok=True)
            return path

        paths = {}
        if formats.get("raw_npy"):
            p = subdir("raw") / f"{stem}.raw_uint.npy"
            np.save(p, raw)
            paths["raw_uint_npy"] = str(p)
        if formats.get("linear_npy"):
            p = subdir("linear") / f"{stem}.linear_float32.npy"
            np.save(p, linear.astype(np.float32, copy=False))
            paths["linear_float32_npy"] = str(p)
        if formats.get("planes_npz"):
            p = subdir("planes") / f"{stem}.planes_float32.npz"
            np.savez_compressed(p, **planes)
            paths["planes_float32_npz"] = str(p)
        if formats.get("preview_png"):
            p = subdir("preview") / f"{stem}.preview.png"
            cv2.imwrite(str(p), cv2.cvtColor(cam.robust_preview_png(rgb16), cv2.COLOR_RGB2BGR))
            paths["preview_png"] = str(p)
        if formats.get("tiff_float32"):
            p = subdir("tiff") / f"{stem}.linear_float32.tiff"
            paths["linear_float32_tiff"] = str(p) if cam.write_float32_tiff_best_effort(p, rgb16) else None
        if formats.get("histogram_png"):
            hist_dir = subdir("histogram")
            p_rgb = hist_dir / f"{stem}.rgb_histogram.png"
            p_cfa = hist_dir / f"{stem}.cfa_histogram.png"
            cam.save_figure(cam.make_rgb_histogram_figure(rgb16, title=f"Linear RGB Histogram: {stem}"), p_rgb)
            cam.save_figure(
                cam.make_cfa_histogram_figure(
                    planes, bit_depth, title=f"Linear CFA Histogram: {stem}",
                    saturation_limits=cam.saturation_limits_from_stats(saturation_stats),
                ),
                p_cfa,
            )
            paths["rgb_histogram_png"] = str(p_rgb)
            paths["cfa_histogram_png"] = str(p_cfa)
        meta_path = subdir("log") / f"{stem}.metadata.json"
        paths["metadata_json"] = str(meta_path)

        payload = {
            "dir": folder.name,
            "name": stem,
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_index": ctrl.camera_index,
            "selected_mode": ctrl.selected_mode,
            "applied_config": ctrl.applied_config,
            "metadata": metadata,
            "bit_depth": bit_depth,
            "raw_format": ctrl.raw_config["format"],
            "bayer_pattern": pattern,
            "raw_domain": raw_domain,
            "black_levels_used": black_levels,
            "raw_min_max": [int(raw.min()), int(raw.max())],
            "linear_min_max": [float(linear.min()), float(linear.max())],
            "linear_mean": float(linear.mean()),
            "saturation": saturation_stats,
            "formats": formats,
            "paths": paths,
            "source": "psf-sweep-gui",
        }
        payload.update(extra)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(cam.json_safe(payload), handle, indent=2, ensure_ascii=False)
        burst = extra.get("burst") or {}
        burst_text = f"burst={burst['index']}/{burst['count']} | " if int(burst.get("count", 1)) > 1 else ""
        cam.append_capture_log(
            ctrl.session_log_path,
            f"[CAPTURE] {time.strftime('%Y-%m-%d %H:%M:%S')} | dir={folder.name} | name={stem} | {burst_text}"
            f"exp_us={metadata.get('ExposureTime')} | again={metadata.get('AnalogueGain')} | raw_max={int(raw.max())}",
        )
        self.hist.emit(self._histogram(raw, metadata, stem))
        return payload

    def capture_burst(self, dir_name: str, base_name: str, formats: dict, extra: dict | None = None) -> dict:
        cam = self.cam
        ctrl = self.controller
        dir_name = cam.sanitize_filename(dir_name or ctrl.config["default_capture_dir"])
        base = cam.sanitize_filename(base_name)
        folder = ctrl.save_root / dir_name
        folder.mkdir(parents=True, exist_ok=True)
        count, delay_min, delay_max = cam.burst_settings(ctrl.config)
        frames = []
        delays = []
        for index in range(1, count + 1):
            if self.abort.is_set():
                self.log.emit("[CAPTURE] aborted")
                break
            if index > 1:
                delays.append(round(cam.burst_pause(delay_min, delay_max), 3))
                time.sleep(delays[-1])
            stem = cam.burst_frame_name(base, index, count)
            frame_extra = dict(extra or {})
            frame_extra["burst"] = {
                "base_name": base,
                "index": index,
                "count": count,
                "delay_range_seconds": [delay_min, delay_max],
                "delay_before_seconds": delays[-1] if index > 1 else 0.0,
            }
            frame = self.save_frame(folder, stem, formats, frame_extra)
            frames.append(frame)
            self.log.emit(
                f"[CAPTURE] {dir_name}/{stem}  exp={frame['metadata'].get('ExposureTime')} us  "
                f"raw_max={frame['raw_min_max'][1]}"
            )
        if not frames:
            raise RuntimeError("no frame captured")
        summary = dict(frames[0])
        summary["name"] = base
        summary["frames"] = frames
        summary["burst"] = {"base_name": base, "count": count, "delays_seconds": delays,
                            "names": [f["name"] for f in frames]}
        return summary

    # -------------------------------------------------------------- stage
    def stage_block(self, target_mm, move_seconds, settle_s) -> dict:
        if self.motor is None:
            return {"connected": False}
        st = self.motor.status()
        return {
            "connected": True,
            "controller": self.motor.info,
            "port": self.motor.port,
            "enc_per_mm": self.motor.enc_per_mm,
            "soft_limits_mm": [self.motor.soft_min_mm, self.motor.soft_max_mm],
            "target_mm": target_mm,
            "measured_mm": round(st.position_mm, 5),
            "measured_counts": st.position_counts,
            "flags": st.flags,
            "homed": st.homed,
            "move_seconds": move_seconds,
            "settle_seconds": settle_s,
        }

    def move(self, mm: float) -> float:
        if self.motor is None:
            raise KDC101Error("stage not connected")
        t0 = time.monotonic()
        st = self.motor.move_to(mm, on_status=self.position.emit)
        dt = time.monotonic() - t0
        self.position.emit(st)
        self.log.emit(f"[STAGE] {st.position_mm:.4f} mm  ({dt:.1f}s, err {self.motor.last_move_error_mm*1000:+.1f} um)")
        return dt

    def home(self):
        if self.motor is None:
            raise KDC101Error("stage not connected")
        self.log.emit("[STAGE] homing (moves to the 0 mm end) ...")
        st = self.motor.home(on_status=self.position.emit)
        self.position.emit(st)
        self.log.emit(f"[STAGE] homed at {st.position_mm:.4f} mm")

    # -------------------------------------------------------------- jobs
    def capture_job(self, dir_name: str, name: str, formats: dict, auto_exposure: bool, wd_memo: str):
        extra = {"wd_memo": wd_memo}
        if auto_exposure:
            extra["auto_exposure"] = self.auto_exposure()
        extra["stage"] = self.stage_block(None, None, None)
        payload = self.capture_burst(dir_name, name, formats, extra)
        self.log.emit(f"[DONE] {payload['name']}: {len(payload['frames'])} frame(s) -> {self.controller.save_root / payload['dir']}")

    def sweep_job(self, p: dict, formats: dict):
        cam = self.cam
        ctrl = self.controller
        if self.motor is None:
            raise KDC101Error("stage not connected")
        positions = self.sweep_mod.sweep_positions(p["start"], p["stop"], p["step"])
        for mm in positions:
            self.motor.check_soft_limits(mm)
        dir_name = cam.sanitize_filename(p["dir"] or ctrl.config["default_capture_dir"])
        folder = ctrl.save_root / dir_name
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = folder / f"_sweep_{stamp}.csv"
        json_path = folder / f"_sweep_{stamp}.json"
        use_index = len(set(positions)) != len(positions)
        n = len(positions)
        saved_burst = ctrl.config["burst_count"]
        ctrl.config["burst_count"] = int(p["burst"])
        self.log.emit(f"[SWEEP] {n} positions {positions[0]:.3f} -> {positions[-1]:.3f} mm, "
                      f"step {abs(p['step']):.3f}, burst {p['burst']}, AE {'on' if p['auto_exposure'] else 'off'} -> {folder}")
        cam.append_capture_log(ctrl.session_log_path,
                               f"[SWEEP START] {time.strftime('%Y-%m-%d %H:%M:%S')} | dir={dir_name} | "
                               f"wd_memo={p.get('wd_memo','')!r} | positions={positions}")
        rows = []
        results = []
        started = time.monotonic()
        aborted = False
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "name", "target_mm", "measured_mm", "move_s", "exposure_us", "fill",
                                 "ae_status", "ae_iterations", "raw_max", "frames", "elapsed_s", "wd_memo"])
                for idx, mm in enumerate(positions, start=1):
                    if self.abort.is_set():
                        aborted = True
                        self.log.emit(f"[SWEEP] aborted before position {idx}/{n}")
                        break
                    self.progress.emit(idx - 1, n, f"{idx}/{n}: moving to {mm:.3f} mm")
                    t0 = time.monotonic()
                    move_s = self.move(mm)
                    if p["settle"] > 0:
                        time.sleep(float(p["settle"]))
                    name = self.sweep_mod.position_name(mm, 2, p.get("prefix", ""))
                    if use_index:
                        name = f"{idx:02d}-{name}"
                    extra = {
                        "wd_memo": p.get("wd_memo", ""),
                        "sweep": {"index": idx, "count": n, "positions_mm": positions,
                                  "settle_seconds": p["settle"], "csv": str(csv_path)},
                    }
                    ae = None
                    if p["auto_exposure"]:
                        self.progress.emit(idx - 1, n, f"{idx}/{n}: auto exposure at {mm:.3f} mm")
                        ae = self.auto_exposure()
                        extra["auto_exposure"] = ae
                    extra["stage"] = self.stage_block(mm, move_s, p["settle"])
                    self.progress.emit(idx - 1, n, f"{idx}/{n}: capturing {name}")
                    payload = self.capture_burst(dir_name, name, formats, extra)
                    frames = payload["frames"]
                    raw_max = max(int(f["raw_min_max"][1]) for f in frames)
                    row = {
                        "index": idx,
                        "name": name,
                        "target_mm": mm,
                        "measured_mm": extra["stage"]["measured_mm"],
                        "move_s": round(move_s, 2),
                        "exposure_us": ae["chosen_exposure_us"] if ae else frames[0]["metadata"].get("ExposureTime"),
                        "fill": None if not ae or ae.get("chosen_fill") is None else round(ae["chosen_fill"], 4),
                        "ae_status": ae["status"] if ae else "manual",
                        "ae_iterations": ae.get("iterations") if ae else None,
                        "raw_max": raw_max,
                        "frames": " ".join(f["name"] for f in frames),
                        "elapsed_s": round(time.monotonic() - t0, 1),
                        "wd_memo": p.get("wd_memo", ""),
                    }
                    writer.writerow([row[k] for k in row])
                    handle.flush()
                    rows.append(row)
                    results.append({"row": row, "paths": [f["paths"] for f in frames]})
                    self.progress.emit(idx, n, f"{idx}/{n} done: {name} exp={row['exposure_us']} us raw_max={raw_max}")
        finally:
            ctrl.config["burst_count"] = saved_burst
        total = time.monotonic() - started
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(cam.json_safe({
                "started": stamp,
                "aborted": aborted,
                "total_seconds": round(total, 1),
                "params": p,
                "positions_mm": positions,
                "formats": formats,
                "controller": self.motor.info,
                "camera_config": ctrl.config,
                "results": results,
            }), handle, indent=2, ensure_ascii=False)
        cam.append_capture_log(ctrl.session_log_path,
                               f"[SWEEP END] {time.strftime('%Y-%m-%d %H:%M:%S')} | dir={dir_name} | "
                               f"{len(rows)}/{n} positions in {total:.0f}s | aborted={aborted} | csv={csv_path.name}")
        self.log.emit(f"[SWEEP] {'aborted after' if aborted else 'finished'} {len(rows)}/{n} positions "
                      f"in {total/60:.1f} min -> {csv_path.name}")
        self.progress.emit(len(rows), n, f"{'aborted' if aborted else 'finished'}: {len(rows)}/{n} in {total/60:.1f} min")


class ElideLabel(QtWidgets.QLabel):
    """Single-line label that never forces a minimum width; long text is elided."""

    def __init__(self, text: str = ""):
        super().__init__()
        self._full = text
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self._refresh()

    def setText(self, text: str):  # noqa: N802 (Qt naming)
        self._full = text
        self.setToolTip(text)
        self._refresh()

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)

    def _refresh(self):
        width = max(20, self.width() - 4)
        QtWidgets.QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, width))


# ========================================================================= histogram
class HistogramCanvas(FigureCanvasQTAgg):
    COLORS = {"R": "#e53935", "G1": "#2e7d32", "G2": "#7cb342", "B": "#1e88e5"}

    def __init__(self):
        self.fig = Figure(figsize=(8, 2.2), dpi=90)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.2)
        self.ax.set_yscale("log")
        self.ax.set_xlabel("raw value", fontsize=8)
        self.ax.set_ylabel("pixels", fontsize=8)
        self.ax.tick_params(labelsize=7)
        self.ax.grid(True, alpha=0.25)
        self.lines = {}
        self.full_line = None
        self.title = self.ax.set_title("waiting for frames ...", fontsize=8, loc="left")
        self.setMinimumHeight(150)

    def update_hist(self, data: dict):
        full = data["full_scale"]
        edges = np.linspace(0, full + 1, data["bins"] + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        ymax = 1.0
        for ch, info in data["channels"].items():
            y = np.maximum(info["counts"], 0.5)
            ymax = max(ymax, float(y.max()))
            if ch not in self.lines:
                (self.lines[ch],) = self.ax.plot(centers, y, color=self.COLORS.get(ch, "k"), lw=1.1, label=ch)
            else:
                self.lines[ch].set_data(centers, y)
        if self.full_line is None:
            self.full_line = self.ax.axvline(full, color="k", ls="--", lw=0.8)
            self.ax.legend(loc="upper right", fontsize=7, ncol=4, framealpha=0.6)
        else:
            self.full_line.set_xdata([full, full])
        self.ax.set_xlim(0, full * 1.03)
        self.ax.set_ylim(0.5, ymax * 3)
        chans = data["channels"]
        sat_total = sum(c["saturated"] for c in chans.values())
        exp = data.get("exposure_us")
        gain = data.get("gain")
        maxes = "  ".join(f"{ch} {chans[ch]['max']}" for ch in ("R", "G1", "G2", "B") if ch in chans)
        sats = "  ".join(f"{ch} {chans[ch]['saturated']}" for ch in ("R", "G1", "G2", "B") if ch in chans)
        self.title.set_text(
            f"{data['label']}   exp {exp} us   gain {gain:.2f}   full {full}\n"
            f"max: {maxes}     saturated px: {sats}"
            if gain is not None else f"{data['label']}  max: {maxes}  saturated: {sats}"
        )
        self.title.set_color("#c62828" if sat_total > 0 else "#1b5e20")
        self.draw_idle()


# ======================================================================== main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cam, controller, motor, sweep_mod, gui_cfg: dict, args):
        super().__init__()
        self.cam = cam
        self.controller = controller
        self.motor = motor
        self.sweep_mod = sweep_mod
        self.gui_cfg = gui_cfg
        self.args = args
        self.current_mm = None
        self.grid_overlay = None
        self._target_initialized = False
        self.setWindowTitle("PSF sweep - KDC101 / Z825B + Pi Camera")

        self.worker = Worker(cam, controller, motor, sweep_mod, hist_period=args.hist_period)
        self.worker.log.connect(self.log_msg)
        self.worker.hist.connect(self.on_hist)
        self.worker.position.connect(self.on_position)
        self.worker.busy.connect(self.on_busy)
        self.worker.progress.connect(self.on_progress)
        self.worker.exposure.connect(self.on_exposure_from_worker)
        self.worker.failed.connect(self.on_failed)

        self._build_ui()
        self._load_from_config()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- left: preview + histogram
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self.preview = self._make_preview()
        left_layout.addWidget(self.preview, stretch=5)
        self.histogram = HistogramCanvas()
        left_layout.addWidget(self.histogram, stretch=2)

        # ---- right: control column (scrolls if the screen is small)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self._build_stage_group())
        right_layout.addWidget(self._build_camera_group())
        right_layout.addWidget(self._build_save_group())
        right_layout.addWidget(self._build_tabs(), stretch=1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        right_w = int(min(max(screen.width() * 0.375, 480), 640))
        scroll.setFixedWidth(right_w)

        root.addWidget(left, stretch=1)
        root.addWidget(scroll, stretch=0)

        # ---- status bar
        status = self.statusBar()
        self.status_label = ElideLabel("ready")
        status.addWidget(self.status_label, 1)
        self.fs_button = QtWidgets.QPushButton("Window / Fullscreen (F11)")
        self.fs_button.clicked.connect(self.toggle_fullscreen)
        status.addPermanentWidget(self.fs_button)
        exit_button = QtWidgets.QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        status.addPermanentWidget(exit_button)

        self.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
            "QPushButton#stop { background: #c62828; color: white; font-weight: bold; }"
            "QPushButton#abort { background: #ef6c00; color: white; font-weight: bold; }"
            "QPushButton#primary { background: #1565c0; color: white; font-weight: bold; }"
        )

    def _make_preview(self):
        from picamera2.previews.qt import QGlPicamera2, QPicamera2
        w, h = self.controller.config["preview_size"]
        try:
            widget = QGlPicamera2(self.controller.camera, width=w, height=h, keep_ar=True)
            self.preview_kind = "QGlPicamera2"
        except Exception as exc:
            print(f"[WARN] QGlPicamera2 unavailable ({exc}); using software preview")
            widget = QPicamera2(self.controller.camera, width=w, height=h, keep_ar=True)
            self.preview_kind = "QPicamera2"
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        widget.setMinimumSize(320, 180)
        return widget

    @staticmethod
    def _form_row(*widgets):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        for item in widgets:
            if isinstance(item, str):
                row.addWidget(QtWidgets.QLabel(item))
            elif isinstance(item, int):
                row.addStretch(item)
            else:
                row.addWidget(item)
        return row

    def _build_stage_group(self):
        group = QtWidgets.QGroupBox("KDC101 stage (Z825B) - larger = LED closer to camera")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(3)

        self.pos_label = QtWidgets.QLabel("--.---- mm")
        font = QtGui.QFont("DejaVu Sans Mono", 18, QtGui.QFont.Bold)
        self.pos_label.setFont(font)
        self.pos_label.setAlignment(Qt.AlignCenter)
        self.pos_label.setStyleSheet("color: #1565c0; background: #eef3fb; border-radius: 4px; padding: 2px;")
        self.flags_label = ElideLabel("stage: connecting ...")
        self.flags_label.setStyleSheet("color: #555;")
        self.flags_label.setMinimumWidth(150)
        self.wd_edit = QtWidgets.QLineEdit()
        self.wd_edit.setPlaceholderText("WD memo, e.g. 'WD 48 mm, mask #3' (saved with every capture)")
        layout.addWidget(self.pos_label)
        layout.addLayout(self._form_row(self.flags_label, 1, "WD memo:", self.wd_edit))

        self.target_spin = QtWidgets.QDoubleSpinBox()
        self.target_spin.setDecimals(3)
        self.target_spin.setSuffix(" mm")
        self.target_spin.setKeyboardTracking(False)
        self.go_button = QtWidgets.QPushButton("Go")
        self.go_button.setObjectName("primary")
        self.go_button.setFixedWidth(52)
        self.go_button.clicked.connect(self.on_go)
        self.minus_button = QtWidgets.QPushButton("-")
        self.plus_button = QtWidgets.QPushButton("+")
        self.minus_button.setFixedWidth(48)
        self.plus_button.setFixedWidth(48)
        self.minus_button.clicked.connect(lambda: self.on_jog(-1))
        self.plus_button.clicked.connect(lambda: self.on_jog(+1))
        self.fine_toggle = QtWidgets.QPushButton("Fine 0.1")
        self.fine_toggle.setFixedWidth(70)
        self.fine_toggle.setToolTip("Toggle: jog and Target step 0.1 mm (pressed) or 1.0 mm (released)")
        self.fine_toggle.setCheckable(True)
        self.fine_toggle.toggled.connect(self._update_jog_labels)
        layout.addLayout(self._form_row("Target:", self.target_spin, self.go_button, 1,
                                        self.minus_button, self.plus_button, self.fine_toggle))

        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setDecimals(2)
        self.max_spin.setRange(0.1, 25.0)
        self.max_spin.setSuffix(" mm")
        self.max_spin.setKeyboardTracking(False)
        self.max_spin.valueChanged.connect(self.on_max_changed)
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.setFixedWidth(60)
        self.home_button.clicked.connect(self.on_home)
        self.stop_button = QtWidgets.QPushButton("STOP")
        self.stop_button.setObjectName("stop")
        self.stop_button.setFixedWidth(64)
        self.stop_button.clicked.connect(self.on_stop)
        self.max_spin.setToolTip("Software upper limit: the LED position closest to the camera that is still safe")
        layout.addLayout(self._form_row("Max (near cam):", self.max_spin, 1, self.home_button, self.stop_button))
        return group

    def _build_camera_group(self):
        group = QtWidgets.QGroupBox("Pi Camera (linear raw, AE/AWB off)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(3)

        lo, hi, _default = self.controller.camera.camera_controls["ExposureTime"]
        self.exp_lo = max(int(lo), 10)
        self.exp_hi = int(min(int(hi), 10_000_000))
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(self.exp_lo, int(hi))
        self.exp_spin.setSuffix(" us")
        self.exp_spin.setKeyboardTracking(False)
        self.exp_spin.setValue(int(self.controller.config["exposure_time"]))
        self.exp_spin.valueChanged.connect(self.on_exposure_spin)
        self.exp_half = QtWidgets.QPushButton("/2")
        self.exp_double = QtWidgets.QPushButton("x2")
        self.exp_half.setFixedWidth(36)
        self.exp_double.setFixedWidth(36)
        self.exp_half.clicked.connect(lambda: self.exp_spin.setValue(max(self.exp_lo, self.exp_spin.value() // 2)))
        self.exp_double.clicked.connect(lambda: self.exp_spin.setValue(min(self.exp_spin.maximum(), self.exp_spin.value() * 2)))
        self.ae_button = QtWidgets.QPushButton("AE (hhh)")
        self.ae_button.setToolTip("Auto exposure search (same as hhh): fills the sensor range without clipping, then applies it")
        self.ae_button.setFixedWidth(72)
        self.ae_button.clicked.connect(self.on_auto_exposure)
        layout.addLayout(self._form_row("Exposure:", self.exp_spin, self.exp_half, self.exp_double, 1, self.ae_button))

        self.exp_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.exp_slider.setRange(0, 1000)
        self.exp_slider.valueChanged.connect(self.on_exposure_slider)
        self._slider_timer = QtCore.QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(150)
        self._slider_timer.timeout.connect(self._apply_slider_exposure)
        layout.addWidget(self.exp_slider)

        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setDecimals(2)
        self.gain_spin.setRange(1.0, 16.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setKeyboardTracking(False)
        self.gain_spin.setValue(float(self.controller.config["analogue_gain"]))
        self.gain_spin.valueChanged.connect(self.on_gain_changed)
        self.burst_spin = QtWidgets.QSpinBox()
        self.burst_spin.setRange(1, 50)
        self.burst_spin.setValue(int(self.controller.config["burst_count"]))
        self.burst_spin.valueChanged.connect(lambda v: self.controller.config.__setitem__("burst_count", int(v)))
        self.delay_min = QtWidgets.QDoubleSpinBox()
        self.delay_min.setRange(0.0, 30.0)
        self.delay_min.setDecimals(1)
        self.delay_min.setSingleStep(0.1)
        self.delay_min.setValue(float(self.controller.config["burst_delay_min"]))
        self.delay_min.valueChanged.connect(lambda v: self.controller.config.__setitem__("burst_delay_min", float(v)))
        self.delay_max = QtWidgets.QDoubleSpinBox()
        self.delay_max.setRange(0.0, 30.0)
        self.delay_max.setDecimals(1)
        self.delay_max.setSingleStep(0.1)
        self.delay_max.setValue(float(self.controller.config["burst_delay_max"]))
        self.delay_max.valueChanged.connect(lambda v: self.controller.config.__setitem__("burst_delay_max", float(v)))
        layout.addLayout(self._form_row("Gain:", self.gain_spin, 1, "Burst:", self.burst_spin,
                                        "delay", self.delay_min, "-", self.delay_max, "s"))

        bit_depth = self.controller.config["bit_depth"]
        gains = self.controller.config["colour_gains"]
        info = ElideLabel(f"AWB off (colour gains {gains[0]:.1f}/{gains[1]:.1f}), AE off, {bit_depth}-bit raw, "
                          f"mode {self.controller.selected_mode.get('size')}")
        info.setStyleSheet("color: #555;")
        layout.addWidget(info)

        fmt_grid = QtWidgets.QGridLayout()
        fmt_grid.setSpacing(2)
        self.format_checks = {}
        for i, (key, label, tip) in enumerate(FORMAT_OPTIONS):
            cb = QtWidgets.QCheckBox(label)
            cb.setToolTip(tip)
            self.format_checks[key] = cb
            fmt_grid.addWidget(cb, i // 3, i % 3)
        fmt_label = QtWidgets.QLabel("Save formats (metadata .json always):")
        layout.addWidget(fmt_label)
        layout.addLayout(fmt_grid)

        self.grid_check = QtWidgets.QCheckBox("4x4 alignment grid")
        self.grid_check.toggled.connect(self.on_grid_toggled)
        self.live_hist_check = QtWidgets.QCheckBox("live histogram")
        self.live_hist_check.toggled.connect(self.on_live_hist_toggled)
        layout.addLayout(self._form_row(self.grid_check, self.live_hist_check, 1))
        return group

    def _build_save_group(self):
        group = QtWidgets.QGroupBox("Save location + capture")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(3)
        self.root_edit = QtWidgets.QLineEdit(str(self.controller.save_root))
        self.root_edit.editingFinished.connect(self.on_root_edited)
        browse = QtWidgets.QPushButton("Browse")
        browse.setFixedWidth(64)
        browse.clicked.connect(self.on_browse)
        layout.addLayout(self._form_row("Save root:", self.root_edit, browse))
        self.dir_edit = QtWidgets.QLineEdit()
        self.name_edit = QtWidgets.QLineEdit()
        layout.addLayout(self._form_row("Dir:", self.dir_edit, "Name:", self.name_edit))
        self.capture_button = QtWidgets.QPushButton("Capture burst")
        self.capture_button.setObjectName("primary")
        self.capture_button.setFixedWidth(110)
        self.capture_button.clicked.connect(lambda: self.on_capture(False))
        self.ae_capture_button = QtWidgets.QPushButton("AE + capture")
        self.ae_capture_button.setFixedWidth(100)
        self.ae_capture_button.clicked.connect(lambda: self.on_capture(True))
        self.path_hint = ElideLabel("")
        self.path_hint.setStyleSheet("color: #555;")
        layout.addLayout(self._form_row(self.capture_button, self.ae_capture_button, 1))
        layout.addWidget(self.path_hint)
        self.dir_edit.textChanged.connect(self._update_path_hint)
        self.name_edit.textChanged.connect(self._update_path_hint)
        return group

    def _build_tabs(self):
        tabs = QtWidgets.QTabWidget()
        # ---- sweep
        sweep = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(sweep)
        layout.setSpacing(3)
        layout.setContentsMargins(6, 6, 6, 6)

        def dspin(lo, hi, dec, step, suffix=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            return s

        self.sw_start = dspin(0.0, 25.0, 3, 0.5)
        self.sw_stop = dspin(0.0, 25.0, 3, 0.5)
        self.sw_step = dspin(0.001, 25.0, 3, 0.1)
        layout.addLayout(self._form_row("Start", self.sw_start, "Stop", self.sw_stop, "Step", self.sw_step, "mm"))
        self.sw_settle = dspin(0.0, 60.0, 1, 0.1, " s")
        self.sw_settle.setToolTip("Pause after each move before auto exposure / capture")
        self.sw_burst = QtWidgets.QSpinBox()
        self.sw_burst.setRange(1, 50)
        self.sw_ae = QtWidgets.QCheckBox("auto exposure at each position")
        layout.addLayout(self._form_row("Settle:", self.sw_settle, "Burst:", self.sw_burst, self.sw_ae, 1))
        self.sw_dir = QtWidgets.QLineEdit()
        self.sw_dir.setPlaceholderText("(uses Dir above when empty)")
        self.sw_prefix = QtWidgets.QLineEdit()
        self.sw_prefix.setPlaceholderText("name prefix, e.g. red-")
        layout.addLayout(self._form_row("Sweep dir:", self.sw_dir, "Prefix:", self.sw_prefix))
        self.sw_info = ElideLabel("")
        self.sw_info.setStyleSheet("color: #555;")
        self.sw_info.setMinimumWidth(120)
        for w in (self.sw_start, self.sw_stop, self.sw_step, self.sw_settle, self.sw_burst):
            w.valueChanged.connect(self._update_sweep_info)
        self.sw_ae.toggled.connect(self._update_sweep_info)
        self.sweep_button = QtWidgets.QPushButton("Start sweep")
        self.sweep_button.setObjectName("primary")
        self.sweep_button.setFixedWidth(96)
        self.sweep_button.clicked.connect(self.on_sweep)
        self.abort_button = QtWidgets.QPushButton("Abort")
        self.abort_button.setObjectName("abort")
        self.abort_button.setFixedWidth(60)
        self.abort_button.clicked.connect(self.on_abort)
        self.abort_button.setEnabled(False)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        layout.addLayout(self._form_row(self.sw_info, 1, self.sweep_button, self.abort_button))
        layout.addWidget(self.progress)
        layout.addStretch(1)
        tabs.addTab(sweep, "Sweep")
        # ---- log
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QtGui.QFont("DejaVu Sans Mono", 8))
        tabs.addTab(self.log_view, "Log")
        self.tabs = tabs
        return tabs

    # ------------------------------------------------------------ config
    def _load_from_config(self):
        cfg = self.gui_cfg
        self.max_spin.setValue(float(cfg["max_mm"]))
        self.target_spin.setRange(0.0, float(cfg["max_mm"]))
        self.wd_edit.setText(cfg.get("wd_memo", ""))
        self.dir_edit.setText(cfg.get("capture_dir", "session"))
        self.name_edit.setText(cfg.get("capture_name", "psf"))
        for key, cb in self.format_checks.items():
            cb.setChecked(bool(cfg["formats"].get(key, False)))
        self.live_hist_check.setChecked(bool(cfg.get("live_hist", True)))
        sw = cfg["sweep"]
        self.sw_start.setValue(float(sw["start"]))
        self.sw_stop.setValue(float(sw["stop"]))
        self.sw_step.setValue(float(sw["step"]))
        self.sw_settle.setValue(float(sw["settle"]))
        self.sw_burst.setValue(int(sw["burst"]))
        self.sw_ae.setChecked(bool(sw["auto_exposure"]))
        self.sw_dir.setText(sw.get("dir", ""))
        self.sw_prefix.setText(sw.get("prefix", ""))
        self._update_jog_labels()
        self._update_sweep_info()
        self._update_path_hint()
        self._sync_slider_from_spin()

    def _collect_config(self) -> dict:
        cfg = self.gui_cfg
        cfg["save_root"] = self.root_edit.text().strip()
        cfg["capture_dir"] = self.dir_edit.text().strip()
        cfg["capture_name"] = self.name_edit.text().strip()
        cfg["max_mm"] = self.max_spin.value()
        cfg["wd_memo"] = self.wd_edit.text()
        cfg["grid"] = self.grid_check.isChecked()
        cfg["live_hist"] = self.live_hist_check.isChecked()
        cfg["formats"] = {key: cb.isChecked() for key, cb in self.format_checks.items()}
        cfg["sweep"] = {
            "start": self.sw_start.value(),
            "stop": self.sw_stop.value(),
            "step": self.sw_step.value(),
            "settle": self.sw_settle.value(),
            "burst": self.sw_burst.value(),
            "auto_exposure": self.sw_ae.isChecked(),
            "dir": self.sw_dir.text().strip(),
            "prefix": self.sw_prefix.text().strip(),
        }
        return cfg

    def formats(self) -> dict:
        return {key: cb.isChecked() for key, cb in self.format_checks.items()}

    # ------------------------------------------------------------- start
    def start(self):
        self.controller.camera.start()
        self.controller.last_frame_exposure_us = int(self.controller.config["exposure_time"])
        self.cam.append_capture_log(
            self.controller.session_log_path,
            f"[SESSION START] {time.strftime('%Y-%m-%d %H:%M:%S')} | camera={self.controller.camera_index} | "
            f"bit_depth={self.controller.config['bit_depth']} | gui=psf-sweep-gui | preview={self.preview_kind}",
        )
        self.grid_check.setChecked(bool(self.gui_cfg.get("grid", True)))
        self.on_grid_toggled(self.grid_check.isChecked())
        if self.motor is None:
            self.flags_label.setText("stage: NOT CONNECTED")
            self.flags_label.setStyleSheet("color: #c62828; font-weight: bold;")
            for w in (self.go_button, self.minus_button, self.plus_button, self.home_button, self.sweep_button):
                w.setEnabled(False)
        else:
            m = self.motor.info
            self.log_msg(f"[STAGE] {m['model']} s/n {m['serial']} fw {m['firmware']} on {self.motor.port}, "
                         f"limits [{self.motor.soft_min_mm}, {self.motor.soft_max_mm}] mm")
        self.log_msg(f"[CAMERA] preview widget {self.preview_kind}, save root {self.controller.save_root}")
        self.worker.start()

    # ----------------------------------------------------------- helpers
    def log_msg(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{stamp} {text}")
        self.status_label.setText(text[:120])
        print(text)

    def jog_step(self) -> float:
        return float(self.gui_cfg.get("fine_step_mm", 0.1)) if self.fine_toggle.isChecked() \
            else float(self.gui_cfg.get("coarse_step_mm", 1.0))

    def _update_jog_labels(self):
        step = self.jog_step()
        self.minus_button.setText(f"-{step:g}")
        self.plus_button.setText(f"+{step:g}")
        self.target_spin.setSingleStep(step)

    def _update_path_hint(self):
        root = self.root_edit.text().strip()
        d = self.cam.sanitize_filename(self.dir_edit.text().strip() or "session")
        n = self.cam.sanitize_filename(self.name_edit.text().strip() or "capture")
        count = self.burst_spin.value()
        suffix = f"-1..{count}" if count > 1 else ""
        self.path_hint.setText(f"-> {root}/{d}/raw/{n}{suffix}.raw_uint.npy, planes/, preview/, log/")

    def _update_sweep_info(self):
        try:
            positions = self.sweep_mod.sweep_positions(self.sw_start.value(), self.sw_stop.value(), self.sw_step.value())
        except Exception:
            self.sw_info.setText("invalid range")
            return
        n = len(positions)
        burst = self.sw_burst.value()
        per_pos = (self.sw_settle.value() + abs(self.sw_step.value()) / 2.0 + 1.5
                   + (8.0 if self.sw_ae.isChecked() else 0.0)
                   + burst * (2.5 + 0.5 * (self.delay_min.value() + self.delay_max.value())))
        self.sw_info.setText(f"{n} positions, {n*burst} frames, ~{n*per_pos/60:.1f} min")

    def _sync_slider_from_spin(self):
        v = self.exp_spin.value()
        frac = (np.log(max(v, self.exp_lo)) - np.log(self.exp_lo)) / (np.log(self.exp_hi) - np.log(self.exp_lo))
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(int(round(float(np.clip(frac, 0, 1)) * 1000)))
        self.exp_slider.blockSignals(False)

    def apply_camera_controls(self):
        self.controller.camera.set_controls(self.controller.make_controls())

    # ------------------------------------------------------------- slots
    def on_exposure_spin(self, value: int):
        self.controller.config["exposure_time"] = int(value)
        self.apply_camera_controls()
        self._sync_slider_from_spin()
        self.status_label.setText(f"exposure {value} us")

    def on_exposure_slider(self, value: int):
        self._slider_timer.start()

    def _apply_slider_exposure(self):
        frac = self.exp_slider.value() / 1000.0
        us = int(round(self.exp_lo * (self.exp_hi / self.exp_lo) ** frac))
        self.exp_spin.blockSignals(True)
        self.exp_spin.setValue(us)
        self.exp_spin.blockSignals(False)
        self.controller.config["exposure_time"] = us
        self.apply_camera_controls()
        self.status_label.setText(f"exposure {us} us")

    def on_exposure_from_worker(self, us: int):
        self.exp_spin.blockSignals(True)
        self.exp_spin.setValue(int(us))
        self.exp_spin.blockSignals(False)
        self._sync_slider_from_spin()

    def on_gain_changed(self, value: float):
        self.controller.config["analogue_gain"] = float(value)
        self.apply_camera_controls()

    def on_grid_toggled(self, on: bool):
        if on:
            if self.grid_overlay is None:
                w, h = self.controller.config["preview_size"]
                ov = np.zeros((h, w, 4), dtype=np.uint8)
                colour = (0, 255, 0, 230)
                for k in range(1, 4):
                    x = int(round(w * k / 4))
                    y = int(round(h * k / 4))
                    ov[max(0, y - 1):y + 1, :] = colour
                    ov[:, max(0, x - 1):x + 1] = colour
                self.grid_overlay = ov
            self.preview.set_overlay(self.grid_overlay)
        else:
            self.preview.set_overlay(None)

    def on_live_hist_toggled(self, on: bool):
        self.worker.live_hist = bool(on)

    def on_root_edited(self):
        text = self.root_edit.text().strip()
        if not text:
            return
        try:
            self.controller.update_setting("save_root", text)
            self.log_msg(f"[SAVE] root = {self.controller.save_root}")
        except Exception as exc:
            self.on_failed(f"save root: {exc}")
        self._update_path_hint()

    def on_browse(self):
        start = self.root_edit.text().strip() or str(Path.home() / "Desktop")
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose save root", start)
        if chosen:
            self.root_edit.setText(chosen)
            self.on_root_edited()

    def on_max_changed(self, value: float):
        if self.motor is not None:
            self.motor.soft_max_mm = float(value)
        self.target_spin.setRange(0.0, float(value))
        self.sw_start.setMaximum(float(value))
        self.sw_stop.setMaximum(float(value))
        self.status_label.setText(f"software limit {value:.2f} mm (closest to camera)")

    def on_position(self, st):
        self.current_mm = st.position_mm
        if not self._target_initialized:
            # Start with Target = where the stage is, so a stray "Go" moves nowhere.
            self._target_initialized = True
            self.target_spin.blockSignals(True)
            self.target_spin.setValue(min(max(st.position_mm, self.target_spin.minimum()), self.target_spin.maximum()))
            self.target_spin.blockSignals(False)
        self.pos_label.setText(f"{st.position_mm:.4f} mm")
        moving = st.moving
        self.pos_label.setStyleSheet(
            "color: #c62828; background: #fdecea; border-radius: 4px; padding: 2px;" if moving else
            "color: #1565c0; background: #eef3fb; border-radius: 4px; padding: 2px;")
        text = ("MOVING  " if moving else "") + ("homed" if st.homed else "NOT HOMED") + \
               ("" if st.enabled else "  DISABLED")
        self.flags_label.setText(f"stage: {text}")
        self.flags_label.setStyleSheet("color: #c62828;" if (moving or not st.homed) else "color: #2e7d32;")

    def on_hist(self, data):
        self.histogram.update_hist(data)

    def on_busy(self, busy: bool):
        for w in (self.go_button, self.minus_button, self.plus_button, self.home_button, self.capture_button,
                  self.ae_capture_button, self.ae_button, self.sweep_button, self.exp_spin, self.exp_slider,
                  self.exp_half, self.exp_double, self.gain_spin, self.max_spin):
            w.setEnabled(not busy and (self.motor is not None or w not in (
                self.go_button, self.minus_button, self.plus_button, self.home_button, self.sweep_button)))
        self.abort_button.setEnabled(busy)
        if not busy:
            self.status_label.setText("ready")

    def on_progress(self, done: int, total: int, text: str):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.progress.setFormat(f"%v / %m   {text}")
        self.status_label.setText(text)

    def on_failed(self, text: str):
        self.log_msg(f"[ERROR] {text}")
        self.status_label.setStyleSheet("color: #c62828; font-weight: bold;")
        QtCore.QTimer.singleShot(4000, lambda: self.status_label.setStyleSheet(""))

    # ------------------------------------------------------------ actions
    def on_go(self):
        mm = self.target_spin.value()
        self.worker.submit("move", lambda: self.worker.move(mm))

    def on_jog(self, sign: int):
        if self.current_mm is None:
            return
        target = round(self.current_mm + sign * self.jog_step(), 4)
        target = min(max(target, self.motor.soft_min_mm), self.motor.soft_max_mm)
        self.target_spin.blockSignals(True)
        self.target_spin.setValue(target)
        self.target_spin.blockSignals(False)
        self.worker.submit("jog", lambda: self.worker.move(target))

    def on_home(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Home stage", "Home the stage? It moves to the 0 mm end (away from the camera).",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.Yes:
            self.worker.submit("home", self.worker.home)

    def on_stop(self):
        self.worker.abort.set()
        if self.motor is not None:
            try:
                self.motor.stop()
                self.log_msg("[STAGE] STOP sent")
            except Exception as exc:
                self.on_failed(f"stop: {exc}")

    def on_abort(self):
        self.worker.abort.set()
        self.log_msg("[SWEEP] abort requested (finishes the current frame)")

    def on_auto_exposure(self):
        self.worker.submit("auto exposure", self.worker.auto_exposure)

    def on_capture(self, with_ae: bool):
        dir_name = self.dir_edit.text().strip()
        name = self.name_edit.text().strip()
        if not name:
            self.on_failed("capture: name is empty")
            return
        formats = self.formats()
        wd = self.wd_edit.text()
        self.worker.submit("capture", lambda: self.worker.capture_job(dir_name, name, formats, with_ae, wd))

    def sweep_params(self) -> dict:
        return {
            "start": self.sw_start.value(),
            "stop": self.sw_stop.value(),
            "step": self.sw_step.value(),
            "settle": self.sw_settle.value(),
            "burst": self.sw_burst.value(),
            "auto_exposure": self.sw_ae.isChecked(),
            "dir": self.sw_dir.text().strip() or self.dir_edit.text().strip(),
            "prefix": self.sw_prefix.text().strip(),
            "wd_memo": self.wd_edit.text(),
        }

    def on_sweep(self):
        p = self.sweep_params()
        formats = self.formats()
        self._collect_config()
        try:
            save_gui_config(self.gui_cfg)
        except Exception as exc:
            print(f"[WARN] could not save GUI config: {exc}")
        self.tabs.setCurrentIndex(0)
        self.worker.submit("sweep", lambda: self.worker.sweep_job(p, formats))

    # ---------------------------------------------------------- window
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.worker.abort.set()
        if self.motor is not None:
            try:
                self.motor.stop()
            except Exception:
                pass
        self.worker.stop()
        self.worker.wait(8000)
        try:
            self._collect_config()
            save_gui_config(self.gui_cfg)
        except Exception as exc:
            print(f"[WARN] could not save GUI config: {exc}")
        try:
            self.preview.cleanup()
        except Exception:
            pass
        try:
            self.controller.close()
        except Exception as exc:
            print(f"[WARN] camera close: {exc}")
        if self.motor is not None:
            try:
                print(f"[STAGE] final position {self.motor.position_mm():.4f} mm")
            except Exception:
                pass
            self.motor.close()
        print("Camera and stage released.")
        event.accept()

    # --------------------------------------------------------- selftest
    def run_selftest(self):
        """Scripted exercise of every job type; exits when the queue drains."""
        self.log_msg("[SELFTEST] start")
        self.grid_check.setChecked(True)
        self.exp_spin.setValue(400)
        self.dir_edit.setText("gui-selftest")
        self.name_edit.setText("selftest")
        self.burst_spin.setValue(1)
        self.wd_edit.setText("selftest WD memo")
        self.on_capture(False)
        if self.motor is not None:
            base = self.max_spin.value()
            self.worker.submit("jog", lambda: self.worker.move(round(base - 0.5, 3)))
            self.worker.submit("jog", lambda: self.worker.move(base))
            self.sw_start.setValue(base)
            self.sw_stop.setValue(round(base - 1.0, 3))
            self.sw_step.setValue(0.5)
            self.sw_burst.setValue(1)
            self.sw_ae.setChecked(True)
            self.sw_settle.setValue(0.3)
            self.sw_dir.setText("gui-selftest-sweep")
            self.on_sweep()
            self.worker.submit("return", lambda: self.worker.move(base))
        self._selftest_timer = QtCore.QTimer(self)
        self._selftest_timer.setInterval(1000)
        self._selftest_idle = 0

        def poll():
            if self.worker.jobs.empty() and not self.worker.is_busy:
                self._selftest_idle += 1
                if self._selftest_idle >= 3:
                    self._selftest_timer.stop()
                    self.log_msg("[SELFTEST] done, closing")
                    self.close()
            else:
                self._selftest_idle = 0

        self._selftest_timer.timeout.connect(poll)
        self._selftest_timer.start()


# ============================================================================= main
def parse_args():
    ap = argparse.ArgumentParser(description="PSF sweep GUI (KDC101 / Z825B + Pi Camera)")
    ap.add_argument("--port", default="auto", help="KDC101 serial port (default: auto-detect)")
    ap.add_argument("--save-root", help="override save_root from the camera config / GUI config")
    ap.add_argument("--max-mm", type=float, help="software upper limit (closest to camera); default from GUI config")
    ap.add_argument("--windowed", action="store_true", help="start in a window instead of fullscreen")
    ap.add_argument("--hist-period", type=float, default=0.5, help="live histogram refresh period, seconds")
    ap.add_argument("--no-motor", action="store_true", help="camera only, skip the KDC101")
    ap.add_argument("--selftest", action="store_true", help="scripted capture + jog + sweep, then exit")
    return ap.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("DejaVu Sans", 9))

    cam = load_module("burst_linear_capture", HERE / "burst-linear-capture.py")
    sweep_mod = load_module("psf_sweep_capture", HERE / "psf-sweep-capture.py")
    gui_cfg = load_gui_config()
    if args.max_mm is not None:
        gui_cfg["max_mm"] = float(args.max_mm)

    config, source = cam.load_config()
    if args.save_root:
        config["save_root"] = args.save_root
    elif gui_cfg.get("save_root"):
        config["save_root"] = gui_cfg["save_root"]
    config["enable_preview"] = False  # the Qt widget is the preview
    cam.print_config_state(config, source)
    gui_cfg["save_root"] = str(Path(config["save_root"]).expanduser())

    motor = None
    if not args.no_motor:
        try:
            motor = KDC101(args.port, soft_max_mm=float(gui_cfg["max_mm"]))
            st = motor.status()
            print(f"[STAGE] {motor.info['model']} s/n {motor.info['serial']} on {motor.port}: {st}")
        except Exception as exc:
            print(f"[WARN] KDC101 not available: {exc}")
            motor = None

    controller = cam.InteractiveCameraController(config)
    controller.configure_camera()  # configure before the Qt widget attaches
    window = MainWindow(cam, controller, motor, sweep_mod, gui_cfg, args)
    if args.windowed:
        window.resize(1280, 720)
        window.show()
    else:
        window.showFullScreen()
    window.start()
    if args.selftest:
        QtCore.QTimer.singleShot(3000, window.run_selftest)
    code = app.exec_()
    sys.exit(code)


if __name__ == "__main__":
    main()
