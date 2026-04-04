import csv
import os
import cv2
from datetime import datetime

class ViolationLogger:
    def __init__(self, output_dir="violations", csv_path="violations_log.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.csv_path = csv_path

        # Create CSV with headers if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp", "violation_type", "plate_number",
                    "confidence", "screenshot_path"
                ])
                writer.writeheader()

    def log(self, frame, violation_type: str,
            plate_number: str = "N/A", confidence: float = 0.0):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.output_dir}/{violation_type}_{timestamp}.jpg"

        # Save screenshot
        cv2.imwrite(filename, frame)

        # Log to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "violation_type", "plate_number",
                "confidence", "screenshot_path"
            ])
            writer.writerow({
                "timestamp": timestamp,
                "violation_type": violation_type,
                "plate_number": plate_number,
                "confidence": confidence,
                "screenshot_path": filename,
            })

        print(f"[VIOLATION] {violation_type} | Plate: {plate_number} | {timestamp}")
        return filename