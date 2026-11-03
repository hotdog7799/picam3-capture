# coding: utf-8
"""
Minimal Thorlabs APT protocol driver for a KDC101 brushed DC controller.

Written for the Z825B 25 mm stage on /dev/ttyUSB0 (Linux ftdi_sio). It only
implements what a PSF sweep needs: identify, home, absolute / relative moves,
position and status readback, stop. Units in the public API are millimetres.

Protocol reference: Thorlabs "APT Communications Protocol" (issue 37).
Header 6 bytes, little endian:
  short  : msg_id u16 | param1 u8 | param2 u8 | dest u8 | source u8
  long   : msg_id u16 | data_len u16 | dest|0x80 u8 | source u8 | data...
"""

from __future__ import annotations

import glob
import struct
import threading
import time
from dataclasses import dataclass

import serial

THORLABS_BY_ID_GLOB = "/dev/serial/by-id/usb-Thorlabs_*"


def find_port(port: str = "auto") -> str:
    """Resolve 'auto' to the Thorlabs by-id symlink (stable across replugs) or /dev/ttyUSB0."""
    if port and port != "auto":
        return port
    matches = sorted(glob.glob(THORLABS_BY_ID_GLOB))
    if matches:
        return matches[0]
    return "/dev/ttyUSB0"

HOST = 0x01
DEVICE = 0x50  # generic USB device (KDC101 is a single-channel unit)
CHAN = 1

# Z8xx series on a KDC101 (TDC001 compatible scaling).
Z8_ENC_PER_MM = 34554.96
Z8_VEL_PER_MM_S = 772981.3
Z8_ACC_PER_MM_S2 = 263.8
Z825B_TRAVEL_MM = 25.0

MGMSG_MOD_IDENTIFY = 0x0223
MGMSG_MOD_SET_CHANENABLESTATE = 0x0210
MGMSG_MOD_REQ_CHANENABLESTATE = 0x0211
MGMSG_MOD_GET_CHANENABLESTATE = 0x0212
MGMSG_HW_REQ_INFO = 0x0005
MGMSG_HW_GET_INFO = 0x0006
MGMSG_HW_STOP_UPDATEMSGS = 0x0012
MGMSG_HW_NO_FLASH_PROGRAMMING = 0x0018
MGMSG_MOT_REQ_POSCOUNTER = 0x0411
MGMSG_MOT_GET_POSCOUNTER = 0x0412
MGMSG_MOT_SET_VELPARAMS = 0x0413
MGMSG_MOT_REQ_VELPARAMS = 0x0414
MGMSG_MOT_GET_VELPARAMS = 0x0415
MGMSG_MOT_REQ_HOMEPARAMS = 0x0441
MGMSG_MOT_GET_HOMEPARAMS = 0x0442
MGMSG_MOT_MOVE_HOME = 0x0443
MGMSG_MOT_MOVE_HOMED = 0x0444
MGMSG_MOT_MOVE_RELATIVE = 0x0448
MGMSG_MOT_MOVE_ABSOLUTE = 0x0453
MGMSG_MOT_MOVE_COMPLETED = 0x0464
MGMSG_MOT_MOVE_STOP = 0x0465
MGMSG_MOT_MOVE_STOPPED = 0x0466
MGMSG_MOT_REQ_STATUSBITS = 0x0429
MGMSG_MOT_GET_STATUSBITS = 0x042A
MGMSG_MOT_REQ_DCSTATUSUPDATE = 0x0490
MGMSG_MOT_GET_DCSTATUSUPDATE = 0x0491
MGMSG_MOT_ACK_DCSTATUSUPDATE = 0x0492

STATUS_BITS = {
    0x00000001: "fwd_limit",
    0x00000002: "rev_limit",
    0x00000010: "moving_fwd",
    0x00000020: "moving_rev",
    0x00000040: "jogging_fwd",
    0x00000080: "jogging_rev",
    0x00000200: "homing",
    0x00000400: "homed",
    0x00001000: "tracking",
    0x00002000: "settled",
    0x00004000: "motion_error",
    0x01000000: "current_limit",
    0x80000000: "enabled",
}
MOVING_MASK = 0x00000010 | 0x00000020 | 0x00000040 | 0x00000080 | 0x00000200


class KDC101Error(RuntimeError):
    pass


@dataclass
class Status:
    position_mm: float
    position_counts: int
    velocity_raw: int
    bits: int

    @property
    def flags(self) -> list[str]:
        return [name for bit, name in STATUS_BITS.items() if self.bits & bit]

    @property
    def moving(self) -> bool:
        return bool(self.bits & MOVING_MASK)

    @property
    def homed(self) -> bool:
        return bool(self.bits & 0x00000400)

    @property
    def enabled(self) -> bool:
        return bool(self.bits & 0x80000000)

    def __str__(self) -> str:
        return f"{self.position_mm:.4f} mm ({self.position_counts} cts) flags={'|'.join(self.flags) or '-'}"


class KDC101:
    def __init__(
        self,
        port: str = "auto",
        *,
        enc_per_mm: float = Z8_ENC_PER_MM,
        travel_mm: float = Z825B_TRAVEL_MM,
        soft_min_mm: float = 0.0,
        soft_max_mm: float | None = None,
        verbose: bool = False,
    ):
        self.port = find_port(port)
        self.enc_per_mm = float(enc_per_mm)
        self.travel_mm = float(travel_mm)
        self.soft_min_mm = float(soft_min_mm)
        self.soft_max_mm = float(travel_mm if soft_max_mm is None else soft_max_mm)
        self.verbose = verbose
        self._unsolicited: list[int] = []
        self._write_lock = threading.Lock()
        self.last_move_error_mm = 0.0
        self.ser = serial.Serial(
            self.port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5,
            rtscts=True,
        )
        try:
            self.ser.set_low_latency_mode(True)
        except Exception:
            pass
        # APT doc: raise RTS, purge, give the FTDI a moment.
        self.ser.rts = True
        time.sleep(0.05)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self._send_short(MGMSG_HW_STOP_UPDATEMSGS)
        self._send_short(MGMSG_HW_NO_FLASH_PROGRAMMING)
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.info = self.get_info()

    # ------------------------------------------------------------------ wire
    def _send_short(self, msg_id: int, p1: int = 0, p2: int = 0):
        frame = struct.pack("<HBBBB", msg_id, p1 & 0xFF, p2 & 0xFF, DEVICE, HOST)
        if self.verbose:
            print(f"  -> {frame.hex(' ')}")
        with self._write_lock:
            self.ser.write(frame)
            self.ser.flush()

    def _send_long(self, msg_id: int, data: bytes):
        frame = struct.pack("<HHBB", msg_id, len(data), DEVICE | 0x80, HOST) + data
        if self.verbose:
            print(f"  -> {frame.hex(' ')}")
        with self._write_lock:
            self.ser.write(frame)
            self.ser.flush()

    def _read_message(self, timeout: float) -> tuple[int, int, int, bytes] | None:
        """Return (msg_id, param1, param2, data) or None on timeout."""
        self.ser.timeout = timeout
        header = self.ser.read(6)
        if len(header) < 6:
            if header:
                raise KDC101Error(f"Truncated header: {header.hex(' ')}")
            return None
        msg_id, = struct.unpack("<H", header[:2])
        data = b""
        if header[4] & 0x80:
            length, = struct.unpack("<H", header[2:4])
            self.ser.timeout = max(timeout, 0.5)
            data = self.ser.read(length)
            if len(data) < length:
                raise KDC101Error(f"Truncated payload for 0x{msg_id:04X}: {len(data)}/{length} bytes")
        if self.verbose:
            print(f"  <- 0x{msg_id:04X} {header.hex(' ')} {data.hex(' ')}")
        return msg_id, header[2], header[3], data

    def _wait_for(self, wanted: int, timeout: float = 2.0) -> tuple[int, int, bytes]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise KDC101Error(f"Timeout waiting for message 0x{wanted:04X}")
            msg = self._read_message(min(remaining, 0.5))
            if msg is None:
                continue
            msg_id, p1, p2, data = msg
            if msg_id == wanted:
                return p1, p2, data
            self._handle_unsolicited(msg_id, p1, p2, data)

    def _handle_unsolicited(self, msg_id: int, p1: int, p2: int, data: bytes):
        """Messages the controller sends on its own (move completed, homed, ...)."""
        self._unsolicited.append(msg_id)
        if msg_id == MGMSG_MOT_GET_DCSTATUSUPDATE:
            # Only streamed after HW_START_UPDATEMSGS; keep the controller happy anyway.
            self._send_short(MGMSG_MOT_ACK_DCSTATUSUPDATE)
        if self.verbose:
            print(f"  (unsolicited 0x{msg_id:04X})")

    # --------------------------------------------------------------- helpers
    def mm_to_counts(self, mm: float) -> int:
        return int(round(mm * self.enc_per_mm))

    def counts_to_mm(self, counts: int) -> float:
        return counts / self.enc_per_mm

    def check_soft_limits(self, mm: float):
        if not (self.soft_min_mm - 1e-9 <= mm <= self.soft_max_mm + 1e-9):
            raise KDC101Error(
                f"Target {mm:.4f} mm outside software limits "
                f"[{self.soft_min_mm:.3f}, {self.soft_max_mm:.3f}] mm"
            )
        if not (0.0 <= mm <= self.travel_mm):
            raise KDC101Error(f"Target {mm:.4f} mm outside stage travel 0..{self.travel_mm} mm")

    # --------------------------------------------------------------- queries
    def get_info(self) -> dict:
        self._send_short(MGMSG_HW_REQ_INFO)
        _, _, data = self._wait_for(MGMSG_HW_GET_INFO)
        if len(data) < 84:
            raise KDC101Error(f"Short HW_GET_INFO payload: {len(data)} bytes")
        serial_no, model, hw_type, fw_minor, fw_interim, fw_major, _unused = struct.unpack("<I8sH4B", data[:18])
        notes = data[18:66]
        hw_version, mod_state, n_channels = struct.unpack("<HHH", data[78:84])
        return {
            "serial": serial_no,
            "model": model.decode("ascii", "replace").strip("\x00 "),
            "type": hw_type,
            "firmware": f"{fw_major}.{fw_interim}.{fw_minor}",
            "notes": notes.decode("ascii", "replace").strip("\x00 "),
            "hw_version": hw_version,
            "mod_state": mod_state,
            "channels": n_channels,
        }

    def identify(self):
        """Flash the front-panel LEDs."""
        self._send_short(MGMSG_MOD_IDENTIFY, CHAN)

    def status(self) -> Status:
        self._send_short(MGMSG_MOT_REQ_DCSTATUSUPDATE, CHAN)
        _, _, data = self._wait_for(MGMSG_MOT_GET_DCSTATUSUPDATE)
        if len(data) < 14:
            raise KDC101Error(f"Short DCSTATUSUPDATE payload: {len(data)} bytes")
        _chan, pos, vel, _res, bits = struct.unpack("<HihHI", data[:14])
        return Status(self.counts_to_mm(pos), pos, vel, bits)

    def position_mm(self) -> float:
        self._send_short(MGMSG_MOT_REQ_POSCOUNTER, CHAN)
        _, _, data = self._wait_for(MGMSG_MOT_GET_POSCOUNTER)
        _chan, pos = struct.unpack("<Hi", data[:6])
        return self.counts_to_mm(pos)

    def velocity_params(self) -> dict:
        self._send_short(MGMSG_MOT_REQ_VELPARAMS, CHAN)
        _, _, data = self._wait_for(MGMSG_MOT_GET_VELPARAMS)
        _chan, vmin, acc, vmax = struct.unpack("<Hiii", data[:14])
        return {
            "min_mm_s": vmin / Z8_VEL_PER_MM_S,
            "max_mm_s": vmax / Z8_VEL_PER_MM_S,
            "accel_mm_s2": acc / Z8_ACC_PER_MM_S2,
            "raw": (vmin, acc, vmax),
        }

    def set_velocity(self, max_mm_s: float, accel_mm_s2: float):
        data = struct.pack(
            "<Hiii",
            CHAN,
            0,
            int(round(accel_mm_s2 * Z8_ACC_PER_MM_S2)),
            int(round(max_mm_s * Z8_VEL_PER_MM_S)),
        )
        self._send_long(MGMSG_MOT_SET_VELPARAMS, data)

    def channel_enabled(self) -> bool:
        self._send_short(MGMSG_MOD_REQ_CHANENABLESTATE, CHAN)
        _, p2, _ = self._wait_for(MGMSG_MOD_GET_CHANENABLESTATE)
        return p2 == 0x01

    def enable(self, on: bool = True):
        self._send_short(MGMSG_MOD_SET_CHANENABLESTATE, CHAN, 0x01 if on else 0x02)
        time.sleep(0.1)

    # ---------------------------------------------------------------- motion
    def _wait_motion_done(self, timeout: float, completion_msg: int, settle_poll: float = 0.1,
                          on_status=None) -> Status:
        """Poll status until the move finished or a completion message arrived.

        `on_status(Status)` is called on every poll so a GUI can show the live position.
        """
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        self._unsolicited.clear()
        seen_moving = False
        last = None
        while time.monotonic() < deadline:
            # Drain anything the controller sent on its own (MOVE_COMPLETED etc.).
            while self.ser.in_waiting >= 6:
                msg = self._read_message(0.2)
                if msg is None:
                    break
                self._handle_unsolicited(*msg)
            if MGMSG_MOT_MOVE_STOPPED in self._unsolicited:
                raise KDC101Error("Move was stopped")
            last = self.status()
            if on_status is not None:
                on_status(last)
            if last.bits & 0x00004000:
                raise KDC101Error(f"Motion error reported: {last}")
            seen_moving = seen_moving or last.moving
            completed = completion_msg in self._unsolicited
            elapsed = time.monotonic() - started
            if not last.moving and (completed or seen_moving or elapsed > 1.0):
                # Either the controller told us, or we watched it move and stop,
                # or nothing happened for a second (zero-length move).
                time.sleep(settle_poll)
                again = self.status()
                if not again.moving:
                    return again
            time.sleep(settle_poll)
        raise KDC101Error(f"Timeout after {timeout:.1f}s waiting for motion to finish (last: {last})")

    def home(self, timeout: float = 90.0, on_status=None) -> Status:
        self._send_short(MGMSG_MOT_MOVE_HOME, CHAN)
        st = self._wait_motion_done(timeout, MGMSG_MOT_MOVE_HOMED, on_status=on_status)
        if not st.homed:
            raise KDC101Error(f"Homing finished but homed flag is not set: {st}")
        return st

    def move_to(self, mm: float, timeout: float | None = None, on_status=None) -> Status:
        self.check_soft_limits(mm)
        here = self.position_mm()
        if timeout is None:
            timeout = abs(mm - here) / 0.3 + 10.0
        self._send_long(MGMSG_MOT_MOVE_ABSOLUTE, struct.pack("<Hi", CHAN, self.mm_to_counts(mm)))
        st = self._wait_motion_done(timeout, MGMSG_MOT_MOVE_COMPLETED, on_status=on_status)
        self.last_move_error_mm = st.position_mm - mm
        if abs(self.last_move_error_mm) > 0.020:
            print(f"[WARN] Move to {mm:.4f} mm ended at {st.position_mm:.4f} mm "
                  f"(error {self.last_move_error_mm*1000:+.1f} um)")
        return st

    def move_by(self, delta_mm: float, timeout: float | None = None, on_status=None) -> Status:
        target = self.position_mm() + delta_mm
        return self.move_to(target, timeout=timeout, on_status=on_status)

    def stop(self, immediate: bool = False):
        self._send_short(MGMSG_MOT_MOVE_STOP, CHAN, 0x01 if immediate else 0x02)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="KDC101 / Z825B quick check")
    ap.add_argument("--port", default="auto", help="serial port, default: auto-detect Thorlabs by-id")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--identify", action="store_true", help="flash the front panel LEDs")
    args = ap.parse_args()
    with KDC101(args.port, verbose=args.verbose) as m:
        print("port      :", m.port)
        print("info      :", m.info)
        print("enabled   :", m.channel_enabled())
        print("velocity  :", m.velocity_params())
        print("status    :", m.status())
        print("poscounter:", f"{m.position_mm():.4f} mm")
        if args.identify:
            m.identify()
