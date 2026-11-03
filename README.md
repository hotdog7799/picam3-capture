# picam3-capture

Raspberry Pi 5 + Pi Camera v3 (IMX708) 선형 raw 촬영 코드와, Thorlabs KDC101 / Z825B LED 스테이지로
PSF sweep을 하는 도구 모음. 실제 경로는 `/home/pi/Desktop/new-capturecode`.

## 파일

| 파일 | 역할 |
|---|---|
| `burst-linear-capture.py` | 메인 촬영 스크립트(대화형). 선형 raw를 burst(여러 장)로 저장, `hhh` 자동 노출 탐색, 히스토그램 |
| `psf-sweep-gui.py` | 전체화면 PyQt5 GUI. 라이브 프리뷰 + 4x4 정렬 격자, 라이브 CFA 히스토그램, KDC101 스테이지 제어, burst 캡처, sweep |
| `psf-sweep-capture.py` | sweep CLI (`--test`, `--sweep START STOP STEP`, 대화형). GUI가 위치 계산 함수를 import함 |
| `kdc101.py` | KDC101 APT 프로토콜 드라이버 (pyserial, mm 단위, 소프트 리밋). `python3 kdc101.py` 로 읽기 전용 점검 |
| `settings/` | `/home/pi/Desktop/Settings/` 스냅샷: IMX708 튜닝 파일, 카메라 설정, GUI 설정 |
| `linear-capture.py`, `picam_v4.py`, `main.py` | 이전 버전 (정리 예정) |

스크립트는 설정을 절대 경로 `/home/pi/Desktop/Settings/` 에서 읽는다. 새 Pi에 복구할 때는
`cp settings/*.json /home/pi/Desktop/Settings/`.

## 실행

```bash
cd ~/Desktop/new-capturecode
uv venv --system-site-packages .venv && uv sync   # 처음 한 번 (picamera2/cv2/PyQt5는 apt 패키지)
uv run burst-linear-capture.py                     # 대화형 촬영
uv run psf-sweep-gui.py                            # GUI (F11 전체화면 전환, --windowed, --selftest)
uv run psf-sweep-capture.py --sweep 16 0 -1 --dir psf --max-mm 16
```

필요한 apt 패키지: `python3-picamera2 python3-opencv python3-pyqt5 python3-matplotlib`.
KDC101은 USB에 꽂으면 커널 ftdi_sio가 `/dev/ttyUSB0` 로 잡고, 코드는
`/dev/serial/by-id/usb-Thorlabs_*` 를 자동 탐지한다.

스테이지 값이 클수록 LED가 카메라에 가깝다 (working distance와 반대). 카메라에 가장 가까운
허용 위치는 캘리브레이션마다 달라지므로 `--max-mm` / GUI의 Max 값으로 매번 지정한다.

## 데이터

촬영 결과(npy/npz/png, 로그, sweep CSV)는 레포에 넣지 않는다(`.gitignore`).
연구실 NAS(`/home/pi/Desktop/share`)로 rsync 한다. `command.md` 참고.
