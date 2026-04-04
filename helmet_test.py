from ultralytics import YOLO
import cv2
import os

MODEL_PATH = r'C:\Users\prana\runs\detect\helmet_model4\weights\best.pt'

model = YOLO(MODEL_PATH)
print('Classes:', model.names)

image_folder = 'smart-traffic-monitor/media'
images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f'Found {len(images)} images')

for img_file in images:
    frame = cv2.imread(os.path.join(image_folder, img_file))
    if frame is None:
        continue
    results = model(frame, verbose=False)[0]
    annotated = results.plot()
    resized = cv2.resize(annotated, (1280, 720))
    for box in results.boxes:
        print(f'  [{img_file}] {model.names[int(box.cls[0])]}: {float(box.conf[0]):.2f}')
    cv2.imshow(f'Helmet - {img_file}', resized)
    if cv2.waitKey(0) == 27:
        break
    cv2.destroyAllWindows()
