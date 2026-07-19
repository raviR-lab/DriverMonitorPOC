"""
Extract evenly-spaced frames from a video into a folder of JPEGs.
Used as a fallback calibration set when no labeled dataset is available.

Usage:
  python scripts/extract_frames.py --video videos/dataset.mp4 --out data/calib/video_frames --count 300
"""
import argparse
import os
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--count", type=int, default=300, help="Number of frames to extract")
    ap.add_argument("--prefix", default="frame", help="Filename prefix")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: {args.video} not found")
        return 1

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("ERROR: video has no frames")
        return 1

    n = min(args.count, total)
    step = max(1, total // n)
    saved = 0
    i = 0
    while saved < n:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            out_path = os.path.join(args.out, f"{args.prefix}_{saved:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        i += 1

    cap.release()
    print(f"Saved {saved} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
