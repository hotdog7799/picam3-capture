# coding: utf-8
"""
PSF sweep: move the LED on a KDC101 / Z825B stage and capture at each position.

Reuses the camera controller from burst-linear-capture.py (auto exposure search
+ burst capture + metadata), and drives the stage through kdc101.py over
the KDC101's USB serial port (auto-detected). Every frame's metadata JSON gains a "stage" block with the
commanded and measured position, and each sweep writes a summary CSV + JSON
into the capture directory.

Examples
  uv run psf-sweep-capture.py                       interactive
  uv run psf-sweep-capture.py --test                 capture at max, min, max
  uv run psf-sweep-capture.py --sweep 16 0 -1 --dir psf-red   16,15,...,0 mm
  uv run psf-sweep-capture.py --max-mm 15.5 --sweep 15.5 0 -0.5

Interactive commands
  pos | status              stage position and flags
  home                      home the stage (moves to 0 mm end)
  move,<mm>                 absolute move
  rel,<dmm>                 relative move
  cap,<name> | cap,<dir>,<name>
                            auto exposure + burst capture at the current position
  sweep,<start>,<stop>,<step>[,<dir>]
                            move + auto exposure + capture at every position
  test                      max -> min -> max with a capture at each
  limit,<max_mm>            change the software upper limit for this session
  setup,<key>,<value>       camera setting (same keys as burst-linear-capture.py)
  quit
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kdc101 import KDC101, KDC101Error  # noqa: E402


def load_capture_module():
    """Import burst-linear-capture.py (hyphenated name, so not a plain import)."""
    path = HERE / "burst-linear-capture.py"
    spec = importlib.util.spec_from_file_location("burst_linear_capture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def position_name(mm: float, decimals: int, prefix: str = "") -> str:
    width = 3 + decimals if decimals > 0 else 2
    text = f"{mm:0{width}.{decimals}f}".replace(".", "p").replace("-", "m")
    return f"{prefix}{text}"


def sweep_positions(start: float, stop: float, step: float) -> list[float]:
    if step == 0:
        raise ValueError("step must not be 0")
    if (stop - start) * step < 0:
        step = -step
    count = int(round(abs(stop - start) / abs(step)))
    positions = [round(start + i * step, 4) for i in range(count + 1)]
    if abs(positions[-1] - stop) > abs(step) * 0.5:
        positions.append(round(stop, 4))
    return positions


class SweepSession:
    def __init__(self, cam_module, controller, motor: KDC101, args):
        self.cam = cam_module
        self.controller = controller
        self.motor = motor
        self.settle_s = float(args.settle)
        self.decimals = int(args.decimals)
        self.prefix = args.name_prefix or ""

    # -------------------------------------------------------------- stage
    def stage_block(self, target_mm: float | None, move_seconds: float | None) -> dict:
        st = self.motor.status()
        return {
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
            "settle_seconds": self.settle_s,
        }

    def move(self, mm: float) -> float:
        t0 = time.monotonic()
        st = self.motor.move_to(mm)
        dt = time.monotonic() - t0
        print(f"[STAGE] -> {mm:.4f} mm : {st}  ({dt:.2f}s, err {self.motor.last_move_error_mm*1000:+.1f} um)")
        if self.settle_s > 0:
            time.sleep(self.settle_s)
        return dt

    # ------------------------------------------------------------ capture
    def capture_here(self, target_dir: str | None, name: str, target_mm: float | None = None,
                     move_seconds: float | None = None, sweep_info: dict | None = None) -> dict:
        search = self.controller.run_auto_exposure()
        print(f"\n{self.controller.auto_exposure_summary(search)}")
        extra = {
            "auto_exposure": search,
            "stage": self.stage_block(target_mm, move_seconds),
        }
        if sweep_info:
            extra["sweep"] = sweep_info
        payload = self.controller.capture(target_dir, name, extra=extra)
        payload["auto_exposure"] = search
        payload["stage"] = extra["stage"]
        return payload

    def run_sweep(self, positions: list[float], target_dir: str | None, index_names: bool = False,
                  label: str = "sweep") -> list[dict]:
        for mm in positions:
            self.motor.check_soft_limits(mm)
        dir_name = self.cam.sanitize_filename(target_dir) if target_dir else \
            self.cam.sanitize_filename(self.controller.config["default_capture_dir"])
        folder = self.controller.save_root / dir_name
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = folder / f"_{label}_{stamp}.csv"
        json_path = folder / f"_{label}_{stamp}.json"
        duplicates = len(set(positions)) != len(positions)
        use_index = index_names or duplicates

        print(f"\n[SWEEP] {len(positions)} positions: {positions[0]:.3f} -> {positions[-1]:.3f} mm "
              f"into {folder}")
        self.cam.append_capture_log(
            self.controller.session_log_path,
            f"[SWEEP START] {time.strftime('%Y-%m-%d %H:%M:%S')} | dir={dir_name} | "
            f"positions={positions}",
        )
        rows = []
        results = []
        sweep_started = time.monotonic()
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "index", "name", "target_mm", "measured_mm", "move_s", "exposure_us",
                "fill", "ae_status", "ae_iterations", "raw_max", "frames", "elapsed_s",
            ])
            for idx, mm in enumerate(positions, start=1):
                print(f"\n===== [{idx}/{len(positions)}] position {mm:.3f} mm =====")
                t0 = time.monotonic()
                move_s = self.move(mm)
                name = position_name(mm, self.decimals, self.prefix)
                if use_index:
                    name = f"{idx:02d}-{name}"
                payload = self.capture_here(
                    dir_name, name, target_mm=mm, move_seconds=move_s,
                    sweep_info={"label": label, "index": idx, "count": len(positions),
                                "positions_mm": positions, "csv": str(csv_path)},
                )
                ae = payload["auto_exposure"]
                frames = payload.get("frames", [payload])
                raw_max = max(int(f["raw_min_max"][1]) for f in frames)
                elapsed = time.monotonic() - t0
                row = {
                    "index": idx,
                    "name": name,
                    "target_mm": mm,
                    "measured_mm": payload["stage"]["measured_mm"],
                    "move_s": round(move_s, 2),
                    "exposure_us": ae.get("chosen_exposure_us"),
                    "fill": None if ae.get("chosen_fill") is None else round(ae["chosen_fill"], 4),
                    "ae_status": ae.get("status"),
                    "ae_iterations": ae.get("iterations"),
                    "raw_max": raw_max,
                    "frames": " ".join(f["name"] for f in frames),
                    "elapsed_s": round(elapsed, 1),
                }
                writer.writerow([row[k] for k in row])
                handle.flush()
                rows.append(row)
                results.append({"row": row, "paths": [f["paths"] for f in frames]})
                print(f"[DONE] {name}: exp={row['exposure_us']} us fill={row['fill']} "
                      f"raw_max={raw_max} frames={len(frames)} ({elapsed:.1f}s)")
        total = time.monotonic() - sweep_started
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.cam.json_safe({
                "label": label,
                "started": stamp,
                "total_seconds": round(total, 1),
                "positions_mm": positions,
                "settle_seconds": self.settle_s,
                "controller": self.motor.info,
                "camera_config": self.controller.config,
                "results": results,
            }), handle, indent=2, ensure_ascii=False)
        self.cam.append_capture_log(
            self.controller.session_log_path,
            f"[SWEEP END] {time.strftime('%Y-%m-%d %H:%M:%S')} | dir={dir_name} | "
            f"{len(positions)} positions in {total:.0f}s | csv={csv_path.name}",
        )
        print(f"\n[SWEEP] finished {len(positions)} positions in {total/60:.1f} min")
        print(f"  summary csv : {csv_path}")
        print(f"  summary json: {json_path}")
        print(f"  {'idx':>3} {'name':<14} {'target':>8} {'measured':>9} {'exp_us':>8} {'fill':>6} {'raw_max':>7} {'s':>6}")
        for r in rows:
            print(f"  {r['index']:>3} {r['name']:<14} {r['target_mm']:>8.3f} {r['measured_mm']:>9.4f} "
                  f"{str(r['exposure_us']):>8} {str(r['fill']):>6} {r['raw_max']:>7} {r['elapsed_s']:>6}")
        return rows

    # ---------------------------------------------------------- interactive
    def handle(self, line: str) -> bool:
        parts = [p.strip() for p in line.split(",")]
        cmd = parts[0].lower()
        if cmd in {"quit", "q", "exit"}:
            return True
        if cmd in {"help", "?"}:
            print(__doc__)
        elif cmd in {"pos", "status"}:
            print(f"[STAGE] {self.motor.status()}  limits=[{self.motor.soft_min_mm}, {self.motor.soft_max_mm}] mm")
            if cmd == "status":
                self.controller.print_runtime_state()
        elif cmd == "home":
            print("[STAGE] homing (moves to the 0 mm end) ...")
            print(f"[STAGE] {self.motor.home()}")
        elif cmd == "move":
            self.move(float(parts[1]))
        elif cmd == "rel":
            self.move(self.motor.position_mm() + float(parts[1]))
        elif cmd == "limit":
            new_max = float(parts[1])
            if not 0 < new_max <= self.motor.travel_mm:
                raise ValueError(f"limit must be within 0..{self.motor.travel_mm} mm")
            self.motor.soft_max_mm = new_max
            print(f"[STAGE] software upper limit = {new_max} mm")
        elif cmd == "cap":
            if len(parts) < 2 or not parts[1]:
                raise ValueError("cap,<name>  or  cap,<dir>,<name>")
            target_dir, name = (None, parts[1]) if len(parts) == 2 else (parts[1] or None, parts[2])
            here = self.motor.position_mm()
            payload = self.capture_here(target_dir, name, target_mm=here)
            print(f"[DONE] {payload['name']} at {payload['stage']['measured_mm']:.4f} mm, "
                  f"exp={payload['auto_exposure'].get('chosen_exposure_us')} us")
        elif cmd == "sweep":
            if len(parts) < 4:
                raise ValueError("sweep,<start>,<stop>,<step>[,<dir>]")
            positions = sweep_positions(float(parts[1]), float(parts[2]), float(parts[3]))
            target_dir = parts[4] if len(parts) > 4 and parts[4] else None
            self.run_sweep(positions, target_dir)
        elif cmd == "test":
            hi, lo = self.motor.soft_max_mm, self.motor.soft_min_mm
            self.run_sweep([hi, lo, hi], parts[1] if len(parts) > 1 and parts[1] else None,
                           index_names=True, label="test")
        elif cmd == "setup":
            if len(parts) < 3:
                raise ValueError("setup,<key>,<value>")
            value = self.controller.update_setting(parts[1], ",".join(parts[2:]))
            print(f"[SETUP] {parts[1]} = {value}")
        else:
            raise ValueError(f"Unknown command: {cmd}")
        return False

    def interactive(self):
        print(__doc__)
        print(f"[STAGE] {self.motor.status()}  limits=[{self.motor.soft_min_mm}, {self.motor.soft_max_mm}] mm")
        while True:
            try:
                line = input("\nSweep >> ").strip()
            except EOFError:
                break
            if not line:
                continue
            try:
                if self.handle(line):
                    break
            except KeyboardInterrupt:
                self.motor.stop()
                print("\n[STAGE] stop requested")
            except (KDC101Error, ValueError) as exc:
                print(f"[ERROR] {exc}")
            except Exception as exc:
                print(f"[ERROR] {type(exc).__name__}: {exc}")


def parse_args():
    ap = argparse.ArgumentParser(description="KDC101 / Z825B PSF sweep capture")
    ap.add_argument("--port", default="auto", help="KDC101 serial port (default: auto-detect Thorlabs by-id)")
    ap.add_argument("--save-root", help="override save_root from the camera config")
    ap.add_argument("--dir", help="capture directory under save_root (default: config default_capture_dir)")
    ap.add_argument("--min-mm", type=float, default=0.0, help="software lower limit (default 0)")
    ap.add_argument("--max-mm", type=float, default=16.0,
                    help="software upper limit = closest safe position to the camera (default 16.0)")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds to wait after each move (default 0.5)")
    ap.add_argument("--decimals", type=int, default=2, help="decimals in position names, 16p00 (default 2)")
    ap.add_argument("--name-prefix", default="", help="prefix for position names, e.g. red-")
    ap.add_argument("--no-home-check", action="store_true", help="do not refuse to run when the stage is not homed")
    ap.add_argument("--sweep", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                    help="run one sweep and exit")
    ap.add_argument("--test", action="store_true", help="capture at max, min, max and exit")
    ap.add_argument("--index-names", action="store_true", help="prefix sweep names with 01-, 02-, ...")
    return ap.parse_args()


def main():
    args = parse_args()
    cam = load_capture_module()
    config, source = cam.load_config()
    if args.save_root:
        config["save_root"] = args.save_root
    if args.dir:
        config["default_capture_dir"] = args.dir
    cam.print_config_state(config, source)

    motor = KDC101(args.port, soft_min_mm=args.min_mm, soft_max_mm=args.max_mm)
    print(f"\n[STAGE] {motor.info['model']} s/n {motor.info['serial']} fw {motor.info['firmware']} on {motor.port}")
    st = motor.status()
    print(f"[STAGE] {st}  limits=[{args.min_mm}, {args.max_mm}] mm")
    if not st.homed and not args.no_home_check:
        print("[STAGE] stage is not homed; homing now (moves to the 0 mm end) ...")
        st = motor.home()
        print(f"[STAGE] {st}")

    controller = None
    try:
        controller = cam.InteractiveCameraController(config)
        controller.start()
        controller.print_runtime_state()
        session = SweepSession(cam, controller, motor, args)
        if args.test:
            session.run_sweep([args.max_mm, args.min_mm, args.max_mm], args.dir, index_names=True, label="test")
        elif args.sweep:
            session.run_sweep(sweep_positions(*args.sweep), args.dir, index_names=args.index_names)
        else:
            session.interactive()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        try:
            motor.stop()
        except Exception:
            pass
    finally:
        if controller is not None:
            controller.close()
        try:
            print(f"[STAGE] final position {motor.position_mm():.4f} mm")
        except Exception:
            pass
        motor.close()
        print("Camera and stage released.")


if __name__ == "__main__":
    main()
