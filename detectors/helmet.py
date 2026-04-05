from ultralytics import YOLO
import os

class HelmetDetector:
    def __init__(self, model_path=None):
        paths_to_try = [
            model_path,
            r"C:\Users\prana\runs\detect\helmet_model4\weights\best.pt",
            r"C:\Users\prana\runs\detect\helmet_model3\weights\best.pt",
            "helmet-detection-yolov8/models/hemletYoloV8_100epochs.pt",
        ]

        loaded = False
        for path in paths_to_try:
            if path and os.path.exists(path):
                self.model = YOLO(path)
                print(f"[HelmetDetector] Loaded: {path}")
                loaded = True
                break

        if not loaded:
            print("[HelmetDetector] No helmet model found!")
            print("[HelmetDetector] Using default YOLOv8n")
            self.model = YOLO("yolov8n.pt")

        print(f"[HelmetDetector] Classes: {self.model.names}")

    def detect(self, frame):
        try:
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf     = float(box.conf[0])
                xyxy     = box.xyxy[0].tolist()
                violation = any(x in cls_name.lower() for x in
                               ["no_helmet", "without",
                                "nohelmet", "no helmet"])
                detections.append({
                    "box":        xyxy,
                    "class":      cls_name,
                    "confidence": round(conf, 3),
                    "violation":  violation
                })
            return detections
        except Exception as e:
            print(f"[HelmetDetector Error] {e}")
            return []