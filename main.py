import cv2
import numpy as np
from ultralytics import YOLO
from detectors.plate import PlateDetector
from detectors.helmet import HelmetDetector
from violation_logger import ViolationLogger

def detect_traffic_light_color(image, rect):
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    font = cv2.FONT_HERSHEY_TRIPLEX
    if cv2.countNonZero(red_mask) > 0:
        text_color = (0, 0, 255)
        message = 'Detected Signal Status: Stop'
        color = 'red'
    elif cv2.countNonZero(yellow_mask) > 0:
        text_color = (0, 255, 255)
        message = 'Detected Signal Status: Caution'
        color = 'yellow'
    else:
        text_color = (0, 255, 0)
        message = 'Detected Signal Status: Go'
        color = 'green'
    cv2.putText(image, message, (15, 70), font, 1.5, text_color, 3, cv2.LINE_AA)
    cv2.putText(image, 34*'-', (10, 115), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return image, color

def draw_traffic_light_indicator(frame, color):
    h, w = frame.shape[:2]
    bx, by = w-130, 20
    bw, bh = 100, 220
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255,255,255), 2)
    circles = [('red', by+50), ('yellow', by+110), ('green', by+170)]
    bgr = {'red': (0,0,255), 'yellow': (0,255,255), 'green': (0,255,0)}
    for name, cy in circles:
        c = bgr[name] if color == name else (60,60,60)
        cv2.circle(frame, (bx+bw//2, cy), 28, c, -1)
    return frame

def draw_stop_line(frame, stop_line_y, color):
    fw = frame.shape[1]
    line_color = (0,0,255) if color == 'red' else (0,255,0)
    x1 = int(fw * 0.25)
    x2 = int(fw * 0.90)
    cv2.line(frame, (x1, stop_line_y), (x2, stop_line_y), line_color, 4)
    cv2.putText(frame, 'STOP LINE', (x1, stop_line_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
    return frame

def draw_fined_plates(frame, fined_plates):
    if not fined_plates:
        return frame
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, 'Fined license plates:', (25, 180), font, 1.0, (255,255,255), 2)
    for i, text in enumerate(fined_plates):
        cv2.putText(frame, '->  '+text, (40, 240+i*60), font, 1.0, (255,255,255), 2)
    return frame

def draw_boxes_on_frame(frame, last_boxes, crossed_ids, pending, penalized_texts):
    for tid, (x1, y1, x2, y2) in last_boxes.items():
        box_color = (0,255,0)
        if tid in crossed_ids or tid in pending:
            box_color = (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
        cv2.putText(frame, f'ID:{tid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        if tid in crossed_ids or tid in pending:
            plate_label = next((p for p in penalized_texts if p != 'N/A'), 'reading...')
            cv2.putText(frame, f'VIOLATION! {plate_label}', (x1, max(y1-40,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    return frame

def main():
    print('[INFO] Loading models...')
    vehicle_model = YOLO('yolov8n.pt')
    helmet_det = HelmetDetector()
    plate_det = PlateDetector()
    logger = ViolationLogger()

    rect = (1700, 40, 100, 250)
    penalized_texts = []
    crossed_ids = set()
    pending = {}
    violation_crops = {}
    frame_count = 0
    last_boxes = {}
    last_color = 'green'
    STOP_LINE_Y = 850

    cap = cv2.VideoCapture('Traffic-Violation-Detection/traffic_video.mp4')
    if not cap.isOpened():
        print('[ERROR] Cannot open video')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'[INFO] Video FPS: {fps}')
    print(f'[INFO] Stop line Y: {STOP_LINE_Y}')
    print('[INFO] Running - press ESC to quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 5 != 0:
            frame = draw_traffic_light_indicator(frame, last_color)
            frame = draw_stop_line(frame, STOP_LINE_Y, last_color)
            frame = draw_boxes_on_frame(frame, last_boxes, crossed_ids, pending, penalized_texts)
            frame = draw_fined_plates(frame, penalized_texts)
            cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            resized = cv2.resize(frame, (1280, 720))
            cv2.imshow('Traffic Violation Detection', resized)
            if cv2.waitKey(1) == 27:
                break
            continue

        clean_frame = frame.copy()
        frame, color = detect_traffic_light_color(frame, rect)
        last_color = color
        frame = draw_stop_line(frame, STOP_LINE_Y, color)
        frame = draw_traffic_light_indicator(frame, color)

        results = vehicle_model.track(clean_frame, persist=True, verbose=False, imgsz=640, classes=[2,3,5,7], device=0)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.tolist()
            track_ids = results[0].boxes.id.int().tolist()
            class_ids = results[0].boxes.cls.int().tolist()

            current_tids = set()
            for box, tid, cid in zip(boxes, track_ids, class_ids):
                x1,y1,x2,y2 = [int(c) for c in box]
                last_boxes[tid] = (x1,y1,x2,y2)
                current_tids.add(tid)
            last_boxes = {k:v for k,v in last_boxes.items() if k in current_tids}

            for box, tid, cid in zip(boxes, track_ids, class_ids):
                x1,y1,x2,y2 = [int(c) for c in box]
                vehicle_bottom = y2
                vehicle_cx = (x1+x2)//2
                fh, fw = clean_frame.shape[:2]

                pad_x = int((x2-x1)*0.3)
                pad_y = int((y2-y1)*0.2)
                cx1 = max(0, x1-pad_x)
                cy1 = max(0, y1-pad_y)
                cx2 = min(fw, x2+pad_x)
                cy2 = min(fh, y2+pad_y)
                vehicle_crop = clean_frame[cy1:cy2, cx1:cx2]

                if vehicle_crop.size == 0:
                    continue

                box_color = (0,255,0)
                in_lane = int(fw*0.25) < vehicle_cx < int(fw*0.90)

                if (color == 'red' and vehicle_bottom > STOP_LINE_Y and in_lane and tid not in crossed_ids and tid not in pending):
                    pending[tid] = 80
                    violation_crops[tid] = vehicle_crop.copy()
                    print(f'[PENDING] Vehicle {tid} | bottom={vehicle_bottom} | stop={STOP_LINE_Y}')

                if tid in pending and frame_count % 10 == 0:
                    prev = violation_crops.get(tid)
                    if prev is None or vehicle_crop.size > prev.size:
                        violation_crops[tid] = vehicle_crop.copy()
                    print(f'[OCR] Running on vehicle {tid}...')
                    plates = plate_det.detect_from_vehicle(violation_crops[tid])
                    plate_txt = None
                    for p in plates:
                        if p['plate_text'] not in ['UNREAD', '']:
                            plate_txt = p['plate_text']
                            break
                    if plate_txt:
                        crossed_ids.add(tid)
                        del pending[tid]
                        violation_crops.pop(tid, None)
                        if plate_txt not in penalized_texts:
                            penalized_texts.append(plate_txt)
                            logger.log(frame, 'red_light_jump', plate_txt)
                            print(f'[VIOLATION] Plate: {plate_txt}')

                if tid in pending:
                    pending[tid] -= 1
                    if pending[tid] <= 0:
                        crossed_ids.add(tid)
                        del pending[tid]
                        violation_crops.pop(tid, None)
                        logger.log(frame, 'red_light_jump', 'N/A')
                        print(f'[VIOLATION] Vehicle {tid} | N/A')

                if tid in crossed_ids or tid in pending:
                    box_color = (0,0,255)
                    plate_label = next((p for p in penalized_texts if p != 'N/A'), 'reading...')
                    cv2.putText(frame, f'VIOLATION! {plate_label}', (x1, max(y1-40,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                if cid == 3 and frame_count % 25 == 0:
                    h_res = helmet_det.detect(vehicle_crop)
                    for h in h_res:
                        if h['violation'] and h['confidence'] > 0.5:
                            logger.log(frame, 'helmetless_riding', 'N/A', h['confidence'])
                            box_color = (0,165,255)
                            cv2.putText(frame, 'NO HELMET!', (x1, max(y1-40,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
                cv2.putText(frame, f'ID:{tid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        frame = draw_fined_plates(frame, penalized_texts)
        cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        resized = cv2.resize(frame, (1280, 720))
        cv2.imshow('Traffic Violation Detection', resized)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'\n[DONE] Frames: {frame_count}')
    print(f'[DONE] Fined plates: {penalized_texts}')

if __name__ == '__main__':
    main()
