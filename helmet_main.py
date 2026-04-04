import cv2
from ultralytics import YOLO
from detectors.plate import PlateDetector
from violation_logger import ViolationLogger
import threading
import queue

MODEL_PATH = r"C:\Users\prana\runs\detect\helmet_model4\weights\best.pt"

# Background OCR thread
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
    helmet_model = YOLO(MODEL_PATH)
    plate_det    = PlateDetector()
    logger       = ViolationLogger(
        output_dir="violations/helmet",
        csv_path="helmet_violations_log.csv"
    )

    print(f"[INFO] Helmet model classes: {helmet_model.names}")

    # Start OCR thread
    ocr_thread = threading.Thread(
        target=ocr_worker, args=(plate_det,), daemon=True)
    ocr_thread.start()

    # Use webcam or video
    source = "smart-traffic-monitor/media/image1.png"
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[INFO] Using webcam...")
        cap = cv2.VideoCapture(0)

    fined_plates  = []
    frame_count   = 0
    pending       = {}

    print("[INFO] Running — press ESC to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()

        # Run helmet detection
        results = helmet_model.track(
            frame, persist=True, verbose=False,
            device=0, imgsz=416)

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.tolist()
            track_ids = results[0].boxes.id.int().tolist()
            class_ids = results[0].boxes.cls.int().tolist()

            for box, tid, cid in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = [int(c) for c in box]
                cls_name  = helmet_model.names[cid]
                crop      = frame[y1:y2, x1:x2]

                # Check for no helmet violation
                is_violation = any(x in cls_name.lower() for x in
                                  ['no_helmet', 'no helmet',
                                   'nohelmet', 'without'])

                if is_violation and tid not in pending:
                    pending[tid] = 20
                    try:
                        ocr_request_queue.put_nowait(
                            (tid, crop.copy()))
                    except queue.Full:
                        pass

                if tid in pending:
                    with ocr_lock:
                        plate_txt = ocr_result_dict.pop(tid, None)

                    pending[tid] -= 1

                    if plate_txt:
                        del pending[tid]
                        if plate_txt not in fined_plates:
                            fined_plates.append(plate_txt)
                            logger.log(annotated,
                                      "helmetless_riding", plate_txt)
                            print(f"[VIOLATION] No Helmet | Plate: {plate_txt}")
                    elif pending[tid] <= 0:
                        del pending[tid]
                        logger.log(annotated, "helmetless_riding", "N/A")
                        print("[VIOLATION] No Helmet | Plate: N/A")

                # Draw box
                color = (0, 165, 255) if is_violation else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name} ID:{tid}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                if is_violation:
                    cv2.putText(annotated, "NO HELMET VIOLATION!",
                                (x1, y1-35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

        # Show fined plates
        if fined_plates:
            cv2.putText(annotated, "Fined Plates:",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            for i, plate in enumerate(fined_plates):
                cv2.putText(annotated, f"-> {plate}",
                            (20, 90 + i*35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(annotated, f"Frame: {frame_count}",
                    (10, annotated.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        resized = cv2.resize(annotated, (1280, 720))
        cv2.imshow("Helmet Violation Detection", resized)

        if cv2.waitKey(8) == 27:
            break

    # Stop OCR thread
    ocr_request_queue.put(None)
    ocr_thread.join(timeout=3)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Frames: {frame_count}")
    print(f"[DONE] Fined plates: {fined_plates}")


if __name__ == "__main__":
    main()