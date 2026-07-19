"""
Benchmark YOLO model variants on a fixed set of frames.

Compares: .pt baseline, fp32.onnx, fp16.onnx, int8.onnx (whichever exist).
Reports model size, mean FPS, and cell-phone detection count parity.

Usage:
  python scripts/bench_yolo.py --video videos/dataset.mp4 --frames 200
"""
import argparse
import os
import sys
import time
import json

import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import (
    YOLO_MODEL_PATH, YOLO_ONNX_FP32, YOLO_ONNX_FP16, YOLO_ONNX_INT8,
    YOLO_INPUT_SIZE, YOLO_CONFIDENCE, YOLO_TARGET_CLASSES,
    PROCESS_WIDTH, PROCESS_HEIGHT, BENCH_DIR,
)


CANDIDATES = [
    ("yolov8n.pt",     YOLO_MODEL_PATH),
    ("fp32.onnx",      YOLO_ONNX_FP32),
    ("fp16.onnx",      YOLO_ONNX_FP16),
    ("int8.onnx",      YOLO_ONNX_INT8),
]


def file_size_mb(p: str) -> float:
    if not os.path.exists(p):
        return 0.0
    return os.path.getsize(p) / (1024 * 1024)


def cell_phone_count(model: YOLO, frames, conf: float) -> int:
    total = 0
    for f in frames:
        try:
            results = model.predict(source=f, imgsz=YOLO_INPUT_SIZE, conf=conf, verbose=False)
        except Exception as e:
            print(f"  predict error: {e}")
            continue
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in YOLO_TARGET_CLASSES:
                    total += 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "videos", "dataset.mp4"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: {args.video} not found")
        return 1

    os.makedirs(BENCH_DIR, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return 1

    frames = []
    while len(frames) < args.frames and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames @ {PROCESS_WIDTH}x{PROCESS_HEIGHT}")

    results = []
    for name, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"[skip] {name}: not found at {path}")
            continue
        print(f"\n=== {name}  ({file_size_mb(path):.2f} MB) ===")
        try:
            model = YOLO(path)
        except Exception as e:
            print(f"  load error: {e}")
            continue

        for i in range(min(args.warmup, len(frames))):
            try:
                model.predict(source=frames[i], imgsz=YOLO_INPUT_SIZE,
                              conf=YOLO_CONFIDENCE, verbose=False)
            except Exception:
                pass

        t0 = time.time()
        det_count = cell_phone_count(model, frames, YOLO_CONFIDENCE)
        elapsed = time.time() - t0
        fps = len(frames) / elapsed if elapsed > 0 else 0.0
        results.append({
            "model": name,
            "path": path,
            "size_mb": round(file_size_mb(path), 2),
            "frames": len(frames),
            "elapsed_s": round(elapsed, 2),
            "fps": round(fps, 2),
            "cell_phone_detections": det_count,
        })
        print(f"  {len(frames)} frames in {elapsed:.2f}s -> {fps:.2f} FPS  | cell-phone hits: {det_count}")

    out_path = os.path.join(BENCH_DIR, "yolo_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({"video": args.video, "frames": len(frames),
                   "input_size": YOLO_INPUT_SIZE, "results": results}, f, indent=2)
    print(f"\nSaved {out_path}")

    if results:
        baseline = next((r for r in results if r["model"] == "yolov8n.pt"), results[0])
        lines = ["# YOLO Benchmark — Edge AI DMS (Phase 1)\n",
                 f"Video: `{os.path.basename(args.video)}`  |  Frames: {len(frames)}  |  YOLO imgsz: {YOLO_INPUT_SIZE}\n",
                 "| Model | Size (MB) | Mean FPS | Speedup vs .pt | cell-phone hits | Parity vs .pt |",
                 "|---|---:|---:|---:|---:|---:|"]
        for r in results:
            speedup = r["fps"] / baseline["fps"] if baseline["fps"] > 0 else 1.0
            parity = (r["cell_phone_detections"] / baseline["cell_phone_detections"]
                      if baseline["cell_phone_detections"] > 0 else 1.0)
            lines.append(f"| {r['model']} | {r['size_mb']:.2f} | {r['fps']:.2f} | "
                         f"{speedup:.2f}× | {r['cell_phone_detections']} | {parity:.0%} |")
        md_path = os.path.join(BENCH_DIR, "yolo_report.md")
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved {md_path}")

        print("\n--- Recommendation ---")
        best_onnx = max((r for r in results if r["model"].endswith(".onnx")),
                         key=lambda r: r["fps"], default=None)
        if best_onnx and baseline["cell_phone_detections"] > 0:
            parity = best_onnx["cell_phone_detections"] / baseline["cell_phone_detections"]
            if parity >= 0.85:
                print(f"Use {best_onnx['model']} on-device: "
                      f"{best_onnx['fps']:.2f} FPS ({best_onnx['fps']/baseline['fps']:.2f}×), "
                      f"{parity:.0%} of baseline detections.")
            else:
                print(f"Best ONNX ({best_onnx['model']}) only retains {parity:.0%} detections. "
                      f"Stick with .pt baseline or improve calibration set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
