"""
Edge AI Driver Monitoring System — Renesas R-Car optimized.

Phase 1 changes:
  - Time-based thresholds (frame-rate independent)
  - Blink vs. drowsiness correctly separated
  - YOLO detection cache cleared on exception
  - Structured DMSEvent output via sink
  - LLM (when enabled) is used only for /ask Q&A, never per-frame
"""
from __future__ import annotations
import cv2
import numpy as np
import time
import threading
from collections import deque
from typing import Optional, Callable, List
from dataclasses import asdict

import mediapipe as mp
from ultralytics import YOLO

from src.config import *
from src.events import DMSEvent, EventType, Severity, Audience
from src.alert_templates import make_event, make_admin_event, speak


# ─────────────────────────────────────────────────────────────────────────────
# Public event sink protocol (SSE server, file logger, anything)
# ─────────────────────────────────────────────────────────────────────────────
class EventSink:
    def emit(self, event: DMSEvent) -> None: ...
    def close(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# TripStats — running counters used both for live alerts and for Q&A
# ─────────────────────────────────────────────────────────────────────────────
class TripStats:
    def __init__(self) -> None:
        self.session_start = time.time()
        self.last_break = self.session_start
        self.counts = {et.value: 0 for et in EventType}
        self.history: List[dict] = []
        self.lock = threading.Lock()

    def bump(self, event_type: EventType) -> None:
        with self.lock:
            self.counts[event_type.value] += 1
            if len(self.history) > 500:
                self.history.pop(0)

    def record(self, event: DMSEvent) -> None:
        with self.lock:
            entry = event.to_json()
            entry.pop("trip_stats", None)
            self.history.append(entry)

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.session_start
            driving_secs = time.time() - self.last_break
            return {
                "session_minutes": round(elapsed / 60, 1),
                "driving_minutes_since_break": round(driving_secs / 60, 1),
                "counts": dict(self.counts),
                "recent_events": list(self.history[-20:]),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Time-based event window: tracks when a condition started, so the duration
# can be compared against a time threshold (frame-rate independent).
# ─────────────────────────────────────────────────────────────────────────────
class _Window:
    __slots__ = ("start", "active")

    def __init__(self) -> None:
        self.start: Optional[float] = None
        self.active: bool = False

    def begin(self) -> None:
        if not self.active:
            self.start = time.time()
            self.active = True

    def end(self) -> None:
        self.active = False
        self.start = None

    def duration(self) -> float:
        if not self.active or self.start is None:
            return 0.0
        return time.time() - self.start

    def reached(self, secs: float) -> bool:
        return self.active and self.duration() >= secs


# ─────────────────────────────────────────────────────────────────────────────
# Background LLM worker — only used when LLM_ENABLED=True
# ─────────────────────────────────────────────────────────────────────────────
class LLMWorker:
    def __init__(self) -> None:
        self.llm = None
        if not LLM_ENABLED:
            return
        if not os.path.exists(LLM_MODEL_PATH):
            print(f"[LLM] model not found at {LLM_MODEL_PATH}; Q&A disabled.")
            return
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=LLM_N_CTX,
                n_threads=LLM_N_THREADS,
                verbose=False,
            )
            print("[LLM] loaded.")
        except Exception as e:
            print(f"[LLM] load failed: {e}")
            self.llm = None

    def answer(self, question: str, stats: dict) -> Optional[str]:
        if not self.llm:
            return None
        prompt = (
            "<|im_start|>system\n"
            "You are an in-vehicle safety copilot. Use only the trip stats provided. "
            "If the question is not answerable from the stats, say so briefly. "
            "Keep replies under 25 words.<|im_end|>\n"
            f"<|im_start|>user\nTrip stats: {stats}\nQuestion: {question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        try:
            r = self.llm(
                prompt,
                max_tokens=80,
                temperature=LLM_TEMPERATURE,
                stop=["<|im_end|>"],
            )
            return r["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[LLM] infer error: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Core DMS
# ─────────────────────────────────────────────────────────────────────────────
class DriverMonitoringSystem:
    def __init__(self, sink: Optional[EventSink] = None) -> None:
        self.sink = sink
        self.trip = TripStats()
        self.llm_worker = LLMWorker()

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # YOLO — prefer int8 ONNX if present, then fp16, then fp32, then .pt
        self.yolo = self._load_yolo()
        self.yolo_frame_counter = 0
        self.last_yolo_detections: List[dict] = []
        self.last_yolo_run_at: float = 0.0

        # Time-based windows
        self.ear_window = _Window()
        self.mar_window = _Window()
        self.gaze_window = _Window()
        self.no_face_window = _Window()

        # Cooldowns
        self.last_alert_time: dict = {}
        self.last_continuous_alert = 0.0
        self.last_admin_alert: dict = {}

        # Head-pose smoothing
        self.pose_buffer = deque(maxlen=5)

    # ────────────────── model loading ──────────────────
    def _load_yolo(self):
        for p in (YOLO_ONNX_INT8, YOLO_ONNX_FP16, YOLO_ONNX_FP32, YOLO_MODEL_PATH):
            if os.path.exists(p):
                try:
                    print(f"[YOLO] loading {os.path.basename(p)}")
                    return YOLO(p)
                except Exception as e:
                    print(f"[YOLO] load error for {p}: {e}")
        print(f"[YOLO] no model found; object detection disabled.")
        return None

    # ────────────────── geometric helpers ──────────────────
    @staticmethod
    def _ear(landmarks: np.ndarray) -> float:
        def ratio(eye):
            v1 = np.linalg.norm(eye[1] - eye[5])
            v2 = np.linalg.norm(eye[2] - eye[4])
            h = np.linalg.norm(eye[0] - eye[3]) + 1e-6
            return (v1 + v2) / (2.0 * h)

        left = np.array([landmarks[i] for i in [362, 385, 387, 263, 373, 380]])
        right = np.array([landmarks[i] for i in [33, 160, 158, 133, 153, 144]])
        return (ratio(left) + ratio(right)) / 2.0

    @staticmethod
    def _mar(landmarks: np.ndarray) -> float:
        v1 = np.linalg.norm(landmarks[81] - landmarks[178])
        v2 = np.linalg.norm(landmarks[13] - landmarks[14])
        v3 = np.linalg.norm(landmarks[311] - landmarks[402])
        h = np.linalg.norm(landmarks[61] - landmarks[291]) * 2.0 + 1e-6
        return (v1 + v2 + v3) / h

    @staticmethod
    def _head_pose(landmarks: np.ndarray, w: int, h: int):
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ], dtype="double")

        image_points = np.array([
            landmarks[1], landmarks[152], landmarks[33],
            landmarks[263], landmarks[61], landmarks[291],
        ], dtype="double")

        focal = w
        camera_matrix = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1],
        ], dtype="double")
        dist = np.zeros((4, 1))

        ok, rvec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None

        rmat, _ = cv2.Rodrigues(rvec)
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
            np.hstack((rmat, np.zeros((3, 1)))),
            camera_matrix,
        )
        pitch = float(euler[0][0]) if hasattr(euler[0], "__len__") else float(euler[0])
        yaw = float(euler[1][0]) if hasattr(euler[1], "__len__") else float(euler[1])
        return yaw, pitch

    # ────────────────── alert dispatch ──────────────────
    def _cooldown_ok(self, key: str, secs: float) -> bool:
        now = time.time()
        last = self.last_alert_time.get(key, 0.0)
        if now - last < secs:
            return False
        self.last_alert_time[key] = now
        return True

    def _emit(self, event: DMSEvent, admin_mirror: bool = False) -> None:
        self.trip.bump(event.event_type)
        self.trip.record(event)
        if self.sink:
            self.sink.emit(event)
        if event.audience == Audience.DRIVER:
            speak(event)
        if admin_mirror:
            adm = make_admin_event(event.event_type, metrics=event.metrics,
                                   trip_stats=event.trip_stats)
            adm.timestamp = event.timestamp
            if self.sink:
                self.sink.emit(adm)
            self.last_admin_alert[event.event_type.value] = time.time()

    # ────────────────── per-frame pipeline ──────────────────
    def process_frame(self, frame: np.ndarray, frame_index: int = 0,
                      frame_dt: Optional[float] = None) -> np.ndarray:
        """
        frame_dt: seconds since the previous frame. If None, the loop derives it.
        """
        h, w = frame.shape[:2]
        out = frame.copy()

        # Resize to a smaller processing resolution to keep CPU low.
        proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        ph, pw = proc.shape[:2]

        # 1) YOLO on every Nth frame
        self.yolo_frame_counter += 1
        if self.yolo and (self.yolo_frame_counter % YOLO_FRAME_SKIP == 0):
            self.last_yolo_detections = self._run_yolo(proc)
            self.last_yolo_run_at = time.time()
        phone_seen = bool(self.last_yolo_detections)
        if phone_seen:
            for det in self.last_yolo_detections:
                x1, y1, x2, y2 = det["box"]
                sx, sy = w / pw, h / ph
                cv2.rectangle(out, (int(x1 * sx), int(y1 * sy)),
                              (int(x2 * sx), int(y2 * sy)), (0, 0, 255), 2)
                cv2.putText(out, f"phone {det['conf']:.2f}",
                            (int(x1 * sx), int(y1 * sy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if self._cooldown_ok(EventType.PHONE_USAGE.value, CRITICAL_ALERT_COOLDOWN_SECS):
                ev = make_event(EventType.PHONE_USAGE,
                                metrics={"confidence": self.last_yolo_detections[0]["conf"]},
                                trip_stats=self.trip.snapshot(),
                                frame_index=frame_index)
                ev.session_id = "trip-1"
                self._emit(ev, admin_mirror=True)

        # 2) MediaPipe face analysis
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        ear = 0.0
        mar = 0.0
        yaw = pitch = 0.0

        if not res.multi_face_landmarks:
            self.no_face_window.begin()
            self.ear_window.end()
            self.mar_window.end()
            self.gaze_window.end()
            if self.no_face_window.reached(NO_FACE_CONSEC_SECS) and \
                    self._cooldown_ok(EventType.NO_FACE.value, CRITICAL_ALERT_COOLDOWN_SECS):
                ev = make_event(EventType.NO_FACE,
                                metrics={"seconds": round(self.no_face_window.duration(), 2)},
                                trip_stats=self.trip.snapshot(),
                                frame_index=frame_index)
                self._emit(ev, admin_mirror=True)
        else:
            self.no_face_window.end()
            for face in res.multi_face_landmarks:
                landmarks = np.array(
                    [(int(lm.x * pw), int(lm.y * ph)) for lm in face.landmark]
                )

                # EAR — start/end a time-based window when eyes are below threshold
                ear = self._ear(landmarks)
                if ear < EAR_THRESHOLD:
                    self.ear_window.begin()
                else:
                    self.ear_window.end()

                if self.ear_window.reached(EAR_CONSEC_SECS) and \
                        self._cooldown_ok(EventType.DROWSINESS.value,
                                          CRITICAL_ALERT_COOLDOWN_SECS):
                    ev = make_event(EventType.DROWSINESS,
                                    metrics={"ear": round(ear, 3),
                                             "seconds": round(self.ear_window.duration(), 2)},
                                    trip_stats=self.trip.snapshot(),
                                    frame_index=frame_index)
                    self._emit(ev, admin_mirror=True)

                # MAR
                mar = self._mar(landmarks)
                if mar > MAR_THRESHOLD:
                    self.mar_window.begin()
                else:
                    self.mar_window.end()

                if self.mar_window.reached(MAR_CONSEC_SECS) and \
                        self._cooldown_ok(EventType.YAWN.value, ALERT_COOLDOWN_SECS):
                    ev = make_event(EventType.YAWN,
                                    metrics={"mar": round(mar, 3),
                                             "seconds": round(self.mar_window.duration(), 2)},
                                    trip_stats=self.trip.snapshot(),
                                    frame_index=frame_index)
                    self._emit(ev)

                # Head pose
                yaw, pitch = self._head_pose(landmarks, pw, ph)
                if yaw is not None:
                    self.pose_buffer.append((yaw, pitch))
                    sy = np.mean([p[0] for p in self.pose_buffer])
                    sp = np.mean([p[1] for p in self.pose_buffer])
                    if abs(sy) > HEAD_YAW_THRESHOLD or abs(sp) > HEAD_PITCH_THRESHOLD:
                        self.gaze_window.begin()
                    else:
                        self.gaze_window.end()

                    if self.gaze_window.reached(GAZE_AWAY_CONSEC_SECS) and \
                            self._cooldown_ok(EventType.DISTRACTION.value,
                                              ALERT_COOLDOWN_SECS):
                        ev = make_event(EventType.DISTRACTION,
                                        metrics={"yaw": round(sy, 1),
                                                 "pitch": round(sp, 1),
                                                 "seconds": round(self.gaze_window.duration(), 2)},
                                        trip_stats=self.trip.snapshot(),
                                        frame_index=frame_index)
                        self._emit(ev, admin_mirror=True)

        # 3) Continuous-drive timer
        snap = self.trip.snapshot()
        drive_min = snap["driving_minutes_since_break"]
        if drive_min >= CONTINUOUS_DRIVE_ALERT_MINS and \
                (time.time() - self.last_continuous_alert) > CONTINUOUS_DRIVE_REMIND_MINS * 60:
            self.last_continuous_alert = time.time()
            ev = make_event(EventType.CONTINUOUS_DRIVE,
                            metrics={"minutes_since_break": drive_min},
                            trip_stats=snap,
                            frame_index=frame_index)
            self._emit(ev, admin_mirror=True)

        # 4) HUD overlay
        self._draw_hud(out, ear, mar, yaw, pitch, frame_index)
        return out

    # ────────────────── visualisation ──────────────────
    def _draw_hud(self, img, ear, mar, yaw, pitch, frame_index):
        h, w = img.shape[:2]
        lines = [
            f"EAR: {ear:.2f}  (eye-closed {self.ear_window.duration():.2f}s)",
            f"MAR: {mar:.2f}  (mouth-open {self.mar_window.duration():.2f}s)",
            f"Yaw: {yaw:+.1f}  Pitch: {pitch:+.1f}  (gaze-away {self.gaze_window.duration():.2f}s)",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (10, 28 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Frame: {frame_index}", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ────────────────── helpers for UI server ──────────────────
    def ask(self, question: str) -> DMSEvent:
        q = DMSEvent(
            event_type=EventType.DRIVER_QUESTION,
            severity=Severity.LOW,
            audience=Audience.TRIP,
            message=question,
            trip_stats=self.trip.snapshot(),
        )
        if self.sink:
            self.sink.emit(q)

        stats = self.trip.snapshot()
        answer = None
        if self.llm_worker.llm is not None:
            answer = self.llm_worker.answer(question, stats)
        if not answer:
            answer = self._template_answer(question, stats)

        a = DMSEvent(
            event_type=EventType.DRIVER_ANSWER,
            severity=Severity.LOW,
            audience=Audience.TRIP,
            message=answer,
            trip_stats=stats,
        )
        if self.sink:
            self.sink.emit(a)
        return a

    @staticmethod
    def _template_answer(question: str, stats: dict) -> str:
        q = question.lower()
        c = stats["counts"]
        if "sleep" in q or "drowsy" in q or "nap" in q:
            return f"You showed drowsiness signs {c.get('DROWSINESS', 0)} times this trip."
        if "distract" in q or "away" in q or "looked" in q:
            return f"Distraction was logged {c.get('DISTRACTION', 0)} times."
        if "yawn" in q:
            return f"You yawned {c.get('YAWN', 0)} times."
        if "phone" in q:
            return f"Phone usage was detected {c.get('PHONE_USAGE', 0)} times."
        if "long" in q or "drive" in q or "break" in q:
            return (f"Continuous drive: {stats['driving_minutes_since_break']} min. "
                    f"Recommended break after 120 min.")
        if "summary" in q or "report" in q:
            return (f"Session {stats['session_minutes']} min. "
                    f"Drowsy {c.get('DROWSINESS', 0)}, "
                    f"distracted {c.get('DISTRACTION', 0)}, "
                    f"yawns {c.get('YAWN', 0)}, phone {c.get('PHONE_USAGE', 0)}.")
        return "I can answer about drowsiness, distraction, yawns, phone use, or drive time."

    def _run_yolo(self, frame: np.ndarray) -> list:
        detections = []
        try:
            results = self.yolo.predict(
                source=frame,
                imgsz=YOLO_INPUT_SIZE,
                conf=YOLO_CONFIDENCE,
                verbose=False,
            )
        except Exception as e:
            print(f"[YOLO] predict error: {e}")
            # explicit reset on exception (was previously left stale)
            self.last_yolo_detections = []
            return detections
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in YOLO_TARGET_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"box": (x1, y1, x2, y2), "conf": conf})
        return detections
