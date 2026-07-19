# Deployment Guide — Renesas R-Car (4 GB)

## 1. Setup board

- Flash Yocto or Ubuntu with `aarch64` support for R-Car
- Connect via SSH or serial
- Update: `sudo apt update && sudo apt install -y python3 python3-pip python3-venv git cmake build-essential`

## 2. Transfer project

```bash
# From your Mac
scp -r /Users/rv/AI\ Automation\ Projects/AI_COE user@renesas-ip:~/dms
```

Or clone from git if you have one.

## 3. Create virtual env & install deps

```bash
cd ~/dms
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Note:** On R-Car ARM64, install `opencv-python` from apt or build from source:
```bash
sudo apt install -y python3-opencv
# OR
pip install opencv-python --no-binary opencv-python
```

## 4. Transfer demo video

```bash
scp /Users/rv/AI\ Automation\ Projects/AI_COE/videos/dataset.mp4 user@renesas-ip:~/dms/videos/
```

## 5. Run

```bash
source .venv/bin/activate

# Headless (no display)
python main.py --video videos/dataset.mp4 --no-display

# With display if HDMI connected
python main.py --video videos/dataset.mp4
```

## 6. Access UI from another machine

The Flask SSE server runs on port 5050 by default.
From your browser on the same network:

```
http://renesas-ip:5050/
```

## Performance expectations

- YOLO int8 ONNX (3.3 MB): ~30–50 FPS on R-Car (vs 94 FPS on M1)
- MediaPipe: ~15–30 FPS
- Combined pipeline: ~10–25 FPS depending on CPU governor

## Troubleshooting

| Issue | Fix |
|---|---|
| `No module named 'cv2'` | `pip install opencv-python` or `apt install python3-opencv` |
| `Illegal instruction` | Build ONNX Runtime from source for ARM: `pip install onnxruntime --no-binary onnxruntime` |
| MediaPipe slow | Try `pip install mediapipe --no-binary mediapipe` for ARM-optimized build |
| Low FPS | Set `YOLO_FRAME_SKIP = 4` or `PROCESS_WIDTH = 320` in `src/config.py` |
