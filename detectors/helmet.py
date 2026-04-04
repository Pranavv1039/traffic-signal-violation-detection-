from ultralytics import YOLO
import os

class HelmetDetector:
    def __init__(self, model_path="helmet-detection-yolov8/best.pt"):
        if not os.path.exists(model_path):
            print(f"[HelmetDetector] Model not found at {model_path}")
            print("[HelmetDetector] Using default YOLOv8n model instead")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(model_path)
        print(f"[HelmetDetector] Model classes: {self.model.names}")

    def detect(self, frame):
        """Returns list of dicts: box, class_name, confidence, violation"""
        try:
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "box": xyxy,
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "violation": any(x in cls_name.lower() for x in
                                   ["no_helmet", "without", "nohelmet", "no helmet"])
                })
            return detections
        except Exception as e:
            print(f"[HelmetDetector Error] {e}")
            return []