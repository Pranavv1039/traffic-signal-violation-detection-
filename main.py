import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from detectors.plate import PlateDetector
from detectors.helmet import HelmetDetector
from violation_logger import ViolationLogger
import threading
import queue


def detect_traffic_light_color(image, rect):
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower    = np.array([0,  120,  70])
    red_upper    = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    red_mask    = cv2.inRange(hsv, red_lower,    red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    font = cv2.FONT_HERSHEY_TRIPLEX

    if cv2.countNonZero(red_mask) > 0:
        text_color = (0, 0, 255)
        message    = "Detected Signal Status: Stop"
        color      = 'red'
    elif cv2.countNonZero(yellow_mask) > 0:
        text_color = (0, 255, 255)
        message    = "Detected Signal Status: Caution"
        color      = 'yellow'
    else:
        text_color = (0, 255, 0)
        message    = "Detected Signal Status: Go"
        color      = 'green'

    cv2.putText(image, message, (15, 70),
                font, 1.5, text_color, 3, cv2.LINE_AA)
    cv2.putText(image, 34*'-', (10, 115),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return image, color


class LineDetector:
    def __init__(self, num_frames_avg=10):
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue   = deque(maxlen=num_frames_avg)

    def detect_white_line(self, frame, color,
                          slope1=0.03, intercept1=920,
                          slope2=0.03, intercept2=770,
                          slope3=-0.8, intercept3=2420):

        def get_color_code(c):
            return {'red': (0, 0, 255), 'green': (0, 255, 0),
                    'yellow': (0, 255, 255)}.get(c.lower())

        frame_org = frame.copy()

        def line1(x): return slope1 * x + intercept1
        def line2(x): return slope2 * x + intercept2
        def line3(x): return slope3 * x + intercept3

        height, width, _ = frame.shape

        mask1 = frame.copy()
        for x in range(width):
            mask1[int(line1(x)):, x] = 0

        mask2 = mask1.copy()
        for x in range(width):
            mask2[:int(line2(x)), x] = 0

        mask3 = mask2.copy()
        for y in range(height):
            mask3[y, :int(line3(y))] = 0

        gray    = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(blurred)
        edges   = cv2.Canny(gray, 30, 100)
        dilated = cv2.dilate(edges, None, iterations=1)
        edges   = cv2.erode(dilated, None, iterations=1)
        lines   = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                  minLineLength=160, maxLineGap=5)

        x_start = 0
        x_end   = width - 1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope     = (y2-y1) / (x2-x1+np.finfo(float).eps)
                intercept = y1 - slope*x1
                self.y_start_queue.append(int(slope*x_start + intercept))
                self.y_end_queue.append(int(slope*x_end + intercept))

        avg_y_start = int(sum(self.y_start_queue)/len(self.y_start_queue)) \
                      if self.y_start_queue else 0
        avg_y_end   = int(sum(self.y_end_queue)/len(self.y_end_queue)) \
                      if self.y_end_queue else 0

        ratio           = 0.32
        x_start_adj     = x_start + int(ratio*(x_end-x_start))
        avg_y_start_adj = avg_y_start + int(ratio*(avg_y_end-avg_y_start))

        mask_draw = np.zeros_like(frame)
        cv2.line(mask_draw,
                 (x_start_adj, avg_y_start_adj),
                 (x_end, avg_y_end),
                 (255, 255, 255), 4)

        color_code = get_color_code(color)
        ch = [1] if color_code == (0, 255, 0) else \
             [2] if color_code == (0, 0, 255) else [1, 2]

        for c in ch:
            frame[mask_draw[:, :, c] == 255, c] = 255

        slope_avg     = (avg_y_end-avg_y_start) / \
                        (x_end-x_start+np.finfo(float).eps)
        intercept_avg = avg_y_start - slope_avg*x_start
        mask_line     = np.copy(frame_org)
        for x in range(width):
            y_line = slope_avg*x + intercept_avg - 35
            mask_line[:int(y_line), x] = 0

        return frame, mask_line, avg_y_start_adj, avg_y_end


def draw_traffic_light_indicator(frame, color):
    h, w   = frame.shape[:2]
    bx, by = w-130, 20
    bw, bh = 100, 220
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 255, 255), 2)
    circles = [('red', by+50), ('yellow', by+110), ('green', by+170)]
    bgr     = {'red': (0, 0, 255), 'yellow': (0, 255, 255),
               'green': (0, 255, 0)}
    for name, cy in circles:
        c = bgr[name] if color == name else (60, 60, 60)
        cv2.circle(frame, (bx+bw//2, cy), 28, c, -1)
    return frame


def draw_fined_plates(frame, fined_plates):
    if not fined_plates:
        return frame
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, 'Fined license plates:',
                (25, 180), font, 1.0, (255, 255, 255), 2)
    for i, text in enumerate(fined_plates):
        cv2.putText(frame, '->  '+text,
                    (40, 240+i*60), font, 1.0, (255, 255, 255), 2)
    return frame


# ── Background OCR worker thread ─────────────────────────────────────────────
ocr_request_queue = queue.Queue(maxsize=3)
ocr_result_dict   = {}
ocr_lock          = threading.Lock()


def ocr_worker(plate_det):
    while True:
        item = ocr_request_queue.get()
        if item is None:
            break
        tid, crop = item
        try:
            plates = plate_det.detect_from_vehicle(crop)
            result = None
            for p in plates:
                if p["plate_text"] not in ["UNREAD", ""]:
                    result = p["plate_text"]
                    break
            with ocr_lock:
                ocr_result_dict[tid] = result
        except Exception as e:
            print(f"[OCR Worker Error] {e}")
        ocr_request_queue.task_done()


def main():
    print("[INFO] Loading models...")
    vehicle_model = YOLO("yolov8n.pt")
    helmet_det    = HelmetDetector(
        "helmet-detection-yolov8/models/hemletYoloV8_100epochs.pt")
    plate_det     = PlateDetector()
    logger        = ViolationLogger()
    detector      = LineDetector()

    # Start background OCR thread
    ocr_thread = threading.Thread(
        target=ocr_worker, args=(plate_det,), daemon=True)
    ocr_thread.start()

    rect            = (1700, 40, 100, 250)
    penalized_texts = []
    crossed_ids     = set()
    pending         = {}
    frame_count     = 0
    all_plates      = []

    cap = cv2.VideoCapture(
        'Traffic-Violation-Detection/traffic_video.mp4')

    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video FPS: {fps}")
    print("[INFO] Running — press ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Show every frame smoothly, process every 5th
        if frame_count % 5 != 0:
            resized = cv2.resize(frame, (1280, 720))
            cv2.imshow("Traffic Violation Detection", resized)
            if cv2.waitKey(8) == 27:
                break
            continue

        clean_frame = frame.copy()

        # 1. Traffic light detection
        frame, color = detect_traffic_light_color(frame, rect)

        # 2. Stop line detection
        frame, mask_line, line_y_start, line_y_end = \
            detector.detect_white_line(frame, color)
        stop_line_y = (line_y_start + line_y_end) // 2

        # 3. Traffic light indicator
        frame = draw_traffic_light_indicator(frame, color)

        # 4. YOLOv8 vehicle tracking with GPU
        results = vehicle_model.track(
            clean_frame, persist=True, verbose=False,
            imgsz=640, classes=[2, 3, 5, 7],
            device=0)

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.tolist()
            track_ids = results[0].boxes.id.int().tolist()
            class_ids = results[0].boxes.cls.int().tolist()

            for box, tid, cid in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = [int(c) for c in box]
                vehicle_bottom  = y2
                vehicle_crop    = clean_frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue

                box_color = (0, 255, 0)

                # RED LIGHT VIOLATION — add to pending
                if (color == 'red'
                        and vehicle_bottom > stop_line_y
                        and tid not in crossed_ids
                        and tid not in pending):
                    pending[tid] = 30
                    # Send to background OCR thread
                    try:
                        ocr_request_queue.put_nowait((tid, vehicle_crop.copy()))
                    except queue.Full:
                        pass

                # Check OCR results from background thread
                if tid in pending:
                    with ocr_lock:
                        plate_txt = ocr_result_dict.pop(tid, None)

                    pending[tid] -= 1

                    if plate_txt:
                        crossed_ids.add(tid)
                        del pending[tid]
                        if plate_txt not in penalized_texts:
                            penalized_texts.append(plate_txt)
                            logger.log(frame, "red_light_jump", plate_txt)
                            print(f"[VIOLATION] Plate: {plate_txt}")
                    elif pending[tid] <= 0:
                        crossed_ids.add(tid)
                        del pending[tid]
                        if "N/A" not in penalized_texts:
                            penalized_texts.append("N/A")
                        logger.log(frame, "red_light_jump", "N/A")
                        print("[VIOLATION] Plate: N/A")
                    else:
                        # Keep updating crop for better detection
                        try:
                            ocr_request_queue.put_nowait(
                                (tid, vehicle_crop.copy()))
                        except queue.Full:
                            pass

                if tid in crossed_ids or tid in pending:
                    box_color = (0, 0, 255)
                    plate_label = next(
                        (p for p in penalized_texts if p != "N/A"), "")
                    cv2.putText(frame,
                                f"VIOLATION! {plate_label}",
                                (x1, y1-40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

                # HELMET VIOLATION
                if cid == 3 and frame_count % 25 == 0:
                    h_res = helmet_det.detect(vehicle_crop)
                    for h in h_res:
                        if h["violation"] and h["confidence"] > 0.5:
                            logger.log(frame, "helmetless_riding", "N/A",
                                       h["confidence"])
                            box_color = (0, 165, 255)
                            cv2.putText(frame, "NO HELMET!",
                                        (x1, y1-40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 165, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"ID:{tid}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, box_color, 2)

        # 5. Fined plates list
        frame = draw_fined_plates(frame, penalized_texts)

        # 6. Frame counter
        cv2.putText(frame, f"Frame: {frame_count}",
                    (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 7. Display
        resized = cv2.resize(frame, (1280, 720))
        cv2.imshow("Traffic Violation Detection", resized)

        if cv2.waitKey(8) == 27:
            break

    # Stop OCR thread
    ocr_request_queue.put(None)
    ocr_thread.join(timeout=3)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Frames: {frame_count}")
    print(f"[DONE] Fined plates: {penalized_texts}")


if __name__ == "__main__":
    main()