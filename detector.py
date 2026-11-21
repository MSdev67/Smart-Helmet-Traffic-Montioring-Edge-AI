
import os
import threading

import cv2
from ultralytics import YOLO


MODEL_PATH = os.getenv("HELMET_MODEL_PATH", "models/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("HELMET_CONFIDENCE", "0.50"))

_model = None
_model_lock = threading.Lock()
_model_error = None


def get_model_status():
    if not os.path.exists(MODEL_PATH):
        return False, f"Model file not found: {MODEL_PATH}"
    try:
        get_model()
        return True, MODEL_PATH
    except Exception as exc:
        return False, str(exc)


def get_model():
    global _model, _model_error
    with _model_lock:
        if _model is not None:
            return _model
        if not os.path.exists(MODEL_PATH):
            _model_error = f"Model file not found: {MODEL_PATH}"
            raise FileNotFoundError(_model_error)
        try:
            _model = YOLO(MODEL_PATH)
            _model_error = None
            return _model
        except Exception as exc:
            _model_error = str(exc)
            raise RuntimeError(f"Unable to load YOLO model: {exc}") from exc


def detect_frame(frame, confidence_threshold=CONFIDENCE_THRESHOLD):
    model = get_model()
    with_helmet = 0
    without_helmet = 0
    detections = []

    results = model(frame, verbose=False, conf=confidence_threshold)
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = str(model.names[class_id]).strip().lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            is_safe = "helmet" in class_name and "without" not in class_name
            if is_safe:
                with_helmet += 1
                color = (42, 199, 130)
                label = f"Helmet {confidence:.0%}"
                status = "safe"
            else:
                without_helmet += 1
                color = (44, 96, 255)
                label = f"No helmet {confidence:.0%}"
                status = "violation"

            detections.append({
                "status": status,
                "class_name": model.names[class_id],
                "confidence": confidence,
                "box": [x1, y1, x2, y2],
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0][0]
            cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + text_width + 14, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 7, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    if without_helmet:
        cv2.putText(
            frame,
            f"VIOLATION  {without_helmet}",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (44, 96, 255),
            2,
            cv2.LINE_AA,
        )

    return frame, with_helmet, without_helmet, detections