# Edge AI Driver Monitoring System — Phase 1 Demo

**AI Center of Excellence · Renesas R-Car / 4 GB RAM prototype**

A fully on-device Driver Monitoring System that runs on a Linux edge board,
detects phone usage, drowsiness, yawning, distraction, and continuous driving,
and streams structured events to a Flask + SSE UI. All alerts and Q&A use
deterministic templates — no LLM on-device.

---

## What was built

| Component | Status | Notes |
|---|---|---|
| YOLOv8n object detection | ✅ | `.pt`, `fp32.onnx`, `fp16.onnx`, `int8.onnx` benchmarked. int8 is the on-device default. |
| MediaPipe face mesh → 468 landmarks | ✅ | 0.10.14 (legacy `solutions` API). |
| Time-based event windows | ✅ | Frame-rate independent; tuned for demo impact. |
| Structured `DMSEvent` log (driver / trip / admin audiences) | ✅ | JSONL to `logs/events.jsonl`, SSE stream, file. |
| Flask + SSE bridge | ✅ | Endpoints: `/`, `/events`, `/ask`, `/stats`. Stub HTML UI. |
| Driver Q&A (template) | ✅ | Answers to "how many times was I distracted?" etc. without an LLM. |
| Offline trip report | ✅ | `scripts/report.py` — template-based; LLM removed. |
| int8 calibration (State Farm stratified) | ✅ | 2600 images (250/class × 10 classes + 100 video frames). |
| Live camera input | ⏸️ Phase 2 | Flag present in `main.py`, marked Phase 2 in `--help`. |
| TTS voice output | ⏸️ Phase 2 | `speak()` is a print stub; swap for pyttsx3 / espeak. |
| LLM on-device | ❌ Not used | `LLM_ENABLED=False`; GGUF model file removed. |

## Architecture

```
                ┌──────────────────────────────────┐
                │  pre-recorded video (Phase 1)     │
                │  live camera (Phase 2)            │
                └────────────────┬─────────────────┘
                                 │  BGR frame
                                 ▼
                ┌──────────────────────────────────┐
                │  OpenCV                          │
                │  ├─ VideoCapture (read source)    │
                │  ├─ cv2.resize → 640×360         │
                │  ├─ solvePnP → yaw, pitch        │
                │  ├─ draw HUD + bounding boxes     │
                │  ├─ VideoWriter (if --save)       │
                │  └─ imshow (display window)       │
                └────────────────┬─────────────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               ▼                                     ▼
    ┌──────────────────────┐    ┌──────────────────────────┐
    │ YOLOv8n int8 ONNX    │    │ MediaPipe Face Mesh      │
    │ every 3rd frame      │    │ every frame              │
    │ @ 320×320            │    │ → 468 face landmarks     │
    │ → cell phone boxes   │    └────────────┬─────────────┘
    │   (COCO class 67)    │                 │
    └─────────┬────────────┘                 │
               │              ┌────────────────┴────────────────┐
              │              ▼                                 ▼
              │   ┌──────────────────────┐   ┌──────────────────────┐
              │   │ _ear (numpy)         │   │ OpenCV solvePnP      │
              │   │ → EAR                │   │ → yaw, pitch         │
              │   ├──────────────────────┤   └──────────────────────┘
              │   │ _mar (numpy)         │
              │   │ → MAR                │
              │   └──────────┬───────────┘
              └──────────────┼───────────────────────────────────────┘
                             ▼
                ┌──────────────────────────────┐
                │  Event Aggregator             │
                │  - time-based windows         │
                │  - per-event cooldowns        │
                │  - audience routing           │
                └──────────────┬───────────────┘
                               │ DMSEvent
                               ▼
                ┌──────────────────────────────┐
                │  EventSink (multi)            │
                │  ├─ logs/events.jsonl         │
                │  ├─ Flask SSE server          │
                │  └─ driver TTS stub           │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Stub UI (browser)            │
                │  http://host:5050             │
                └──────────────────────────────┘
```

## Quick start

```bash
# 1. activate the virtualenv
source .venv/bin/activate

# 2. one-command demo (opens UI, processes videos/dataset.mp4)
bash scripts/run_demo.sh --save
```

Then open **http://127.0.0.1:5050** in a browser.

## Endpoints

| URL | Purpose |
|---|---|
| `GET  /`         | Stub HTML UI (alerts + log + ask box) |
| `GET  /events`   | SSE stream of all `DMSEvent` JSON |
| `POST /ask`      | `{ "question": "..." }` → `{ "answer": "..." }` |
| `GET  /stats`    | Current trip-stats snapshot |

Example:

```bash
curl -X POST http://127.0.0.1:5050/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"how many times was I distracted?"}'
```

## What we measured

See **`output/benchmarks/yolo_report.md`** for the full table. Headline:

- **int8 ONNX: 1.15× faster than .pt, 48% smaller** on the same CPU.
- fp16 ONNX is *slower* than fp32 on CPU — no fp16 acceleration without VNNI-AVX512 or a GPU.
- Demo video FPS (M1, with the full pipeline): **~80 FPS** end-to-end (headless).

## Phase 2 — what is deferred

1. **Live camera integration** — USB, MIPI CSI, or RTSP. Will need a new config block and possibly libcamera/V4L2 on the Renesas board.
2. **Renesas-board benchmarking** — the M1 numbers don't transfer 1:1 to the R-Car. Real FPS will be measured on the actual board.
3. **TTS voice output** — replace `speak()` in `src/alert_templates.py` with a call to `pyttsx3` or `espeak`. The function signature stays the same.
4. **Custom YOLO fine-tuning** — to add `cigarette`, `food/drink`, `seatbelt` (COCO does not include these as distinct classes).
5. **Per-driver profile / personalization** — EAR/MAR thresholds vary by individual; a 30-second calibration step per driver would cut false positives substantially.

## File map

```
AI_COE/
├── main.py                      # entry point
├── requirements.txt
├── src/
│   ├── config.py                # all thresholds, paths, ports
│   ├── events.py                # DMSEvent dataclass
│   ├── alert_templates.py       # TTS-ready strings + speak() stub
│   ├── dms.py                   # core pipeline
│   └── ui_server.py             # Flask + SSE bridge + stub HTML
├── scripts/
│   ├── run_demo.sh              # one-command launcher
│   ├── bench_yolo.py            # YOLO model benchmark
│   ├── extract_frames.py        # video → frames for calibration
│   └── report.py                # JSONL → markdown trip report
├── models/
│   ├── yolov8n.pt               # PyTorch baseline
│   ├── yolov8n_fp32.onnx        # ONNX fp32
│   ├── yolov8n_fp16.onnx        # ONNX fp16
│   └── yolov8n_int8.onnx        # ONNX int8  ← default on-device
├── data/calib/                  # int8 calibration set + audit trail
├── output/
│   ├── dms_out_*.mp4            # annotated output videos
│   └── benchmarks/              # yolo_report.md, trip_report.md
├── logs/events.jsonl            # every DMSEvent, JSONL
└── videos/dataset.mp4           # demo input
```

## How the LLM is used (and how it isn't)

| Use case | Where the model runs | Status |
|---|---|---|
| Real-time driver alerts (drowsy / phone / etc.) | **On-device, templates** | Ships now |
| Driver Q&A ("how many times was I distracted?") | **On-device, templates** | Ships now |
| Trip report from `logs/events.jsonl` | **On-device, templates** | `scripts/report.py` |
| Anything per-frame at inference time | **Never** | Blocked on purpose |

Templates are deterministic, instant, and TTS-ready — suitable for real-time
alerts without loading an LLM on the 4 GB Renesas board.

## On-device threshold tuning (current values)

From `src/config.py`:

| Threshold | Value | Meaning |
|---|---:|---|
| `EAR_THRESHOLD` | 0.21 | Below this → eyes closing |
| `EAR_CONSEC_SECS` | 0.7 | Eyes closed continuously > 0.7s → DROWSINESS alert |
| `MAR_THRESHOLD` | 0.70 | Above this → mouth opening |
| `MAR_CONSEC_SECS` | 0.5 | Mouth open > 0.5s → YAWN alert |
| `HEAD_YAW_THRESHOLD` | 30° | Look left/right beyond this |
| `HEAD_PITCH_THRESHOLD` | 25° | Look up/down beyond this |
| `GAZE_AWAY_CONSEC_SECS` | 1.0 | Look away > 1.0s → DISTRACTION alert |
| `NO_FACE_CONSEC_SECS` | 2.0 | No face > 2.0s → NO_FACE alert |
| `YOLO_CONFIDENCE` | 0.45 | Min YOLO confidence for `cell phone` |
| `CONTINUOUS_DRIVE_ALERT_MINS` | 120 | First continuous-drive alert after 2 hours |
| `CONTINUOUS_DRIVE_REMIND_MINS` | 30 | Re-alert cadence after the first |

All time-based → same behavior at 15 fps, 30 fps, or 60 fps input.
