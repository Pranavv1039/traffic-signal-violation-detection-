import cv2
import numpy as np
import re
import easyocr

class PlateDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            'Traffic-Violation-Detection/haarcascade_russian_plate_number.xml'
        )
        print("[PlateDetector] Loading EasyOCR with GPU...")
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("[PlateDetector] EasyOCR ready on GPU!")

    def ocr_plate(self, plate_img):
        try:
            gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray  = cv2.resize(gray, None, fx=3, fy=3)
            gray  = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            results = self.reader.readtext(thresh, detail=1)

            for (bbox, text, conf) in results:
                text = ''.join(e for e in text if e.isalnum() or e == ' ')
                text = ' '.join(text.split()).upper()

                match = re.search(r'\b([A-Z]{2})\s*([0-9]{3,4})\b', text)
                if match and conf > 0.3:
                    return f"{match.group(1)} {match.group(2)}"

            return None
        except Exception as e:
            print(f"[OCR Error] {e}")
            return None

    def detect(self, frame):
        plates = []
        try:
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.cascade.detectMultiScale(
                gray, 1.07, 15, minSize=(20, 20))
            for (x, y, w, h) in detected:
                plate_crop = frame[y:y+h, x:x+w]
                text = self.ocr_plate(plate_crop)
                if text:
                    plates.append({
                        "plate_text": text.upper(),
                        "confidence": 0.8,
                    })
        except Exception as e:
            print(f"[PlateDetector Error] {e}")
        return plates

    def detect_from_vehicle(self, vehicle_crop):
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 40:
            return []

        # 1. Try normal Haar detection
        plates = self.detect(vehicle_crop)
        if plates:
            return plates

        # 2. Try bottom 40% center strip
        try:
            bottom_crop = vehicle_crop[
                int(h*0.55):h,
                int(w*0.05):int(w*0.95)
            ]
            if bottom_crop.size > 0:
                text = self.ocr_plate(bottom_crop)
                if text:
                    return [{"plate_text": text.upper(),
                             "confidence": 0.6}]
        except Exception as e:
            print(f"[Strategy 2 Error] {e}")

        # 3. Try bottom 25% narrow strip
        try:
            narrow_crop = vehicle_crop[
                int(h*0.72):int(h*0.95),
                int(w*0.15):int(w*0.85)
            ]
            if narrow_crop.size > 0:
                text = self.ocr_plate(narrow_crop)
                if text:
                    return [{"plate_text": text.upper(),
                             "confidence": 0.5}]
        except Exception as e:
            print(f"[Strategy 3 Error] {e}")

        # 4. Try Haar with relaxed parameters
        try:
            gray     = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            detected = self.cascade.detectMultiScale(
                gray, 1.05, 5, minSize=(15, 15))
            for (x, y, ww, hh) in detected:
                plate_crop = vehicle_crop[y:y+hh, x:x+ww]
                text = self.ocr_plate(plate_crop)
                if text:
                    return [{"plate_text": text.upper(),
                             "confidence": 0.7}]
        except Exception as e:
            print(f"[Strategy 4 Error] {e}")

        return []