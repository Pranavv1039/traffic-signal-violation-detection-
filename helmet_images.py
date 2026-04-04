import cv2, os, easyocr, re
from ultralytics import YOLO
from violation_logger import ViolationLogger

MODEL_PATH = r'C:\Users\prana\runs\detect\helmet_model4\weights\best.pt'
print('[INFO] Loading EasyOCR...')
reader = easyocr.Reader(['en'], gpu=True)
print('[INFO] EasyOCR ready!')

def read_plate(frame, box):
    x1,y1,x2,y2 = [int(c) for c in box]
    pad=15; h,w=frame.shape[:2]
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w,x2+pad); y2=min(h,y2+pad)
    crop=frame[y1:y2,x1:x2]
    if crop.size==0: return 'N/A'
    crop=cv2.resize(crop,None,fx=3,fy=3)
    results=reader.readtext(crop,detail=1)
    all_text = ' '.join([text.upper() for (bbox,text,conf) in results])
    all_text = all_text.replace('VAMAHA','').replace('YAMAHA','').replace('MESN','').replace('NESO','')
    all_text = re.sub(r'(?<=[A-Z])O(?=\d)','0',all_text)
    all_text = re.sub(r'(?<=\d)O(?=[A-Z\d])','0',all_text)
    all_text = all_text.replace('IO','10').replace('I0','10')
    all_text = ' '.join(all_text.split())
    print(f'OCR JOINED: {all_text}')
    m=re.search(r'([A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{3,4})',all_text)
    if m: return m.group(1).strip()
    m=re.search(r'(\d{2}[\s-][A-Z]\d[\s-]?\d{4})',all_text)
    if m: return m.group(1).strip()
    compact = all_text.replace(' ','')
    m=re.search(r'([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4})',compact)
    if m: return m.group(1).strip()
    return all_text if len(all_text)>3 else 'N/A'

def main():
    print('[INFO] Loading helmet model...')
    helmet_model = YOLO(MODEL_PATH)
    logger = ViolationLogger(output_dir='violations/helmet', csv_path='helmet_violations_log.csv')
    print(f'[INFO] Classes: {helmet_model.names}')
    image_folder = 'smart-traffic-monitor/media'
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg','.jpeg','.png'))]
    print(f'[INFO] Found {len(images)} images')
    fined = []
    for img_file in images:
        frame = cv2.imread(f'{image_folder}/{img_file}')
        if frame is None: continue
        print(f'\n[INFO] Processing {img_file}...')
        ann = frame.copy()
        res = helmet_model(frame, verbose=False)[0]
        nh=[]; lp=[]
        for box in res.boxes:
            x1,y1,x2,y2=[int(c) for c in box.xyxy[0].tolist()]
            cn=helmet_model.names[int(box.cls[0])]; cf=float(box.conf[0])
            print(f'  Detected: {cn} {cf:.2f}')
            if 'no helmet' in cn.lower() and cf>0.3:
                nh.append((x1,y1,x2,y2,cf))
                cv2.rectangle(ann,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(ann,f'NO HELMET {cf:.2f}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            elif cn=='LP':
                lp.append((x1,y1,x2,y2,cf))
                cv2.rectangle(ann,(x1,y1),(x2,y2),(255,165,0),2)
            elif 'helmet' in cn.lower():
                cv2.rectangle(ann,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(ann,f'HELMET {cf:.2f}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        if nh:
            box=lp[0][:4] if lp else [0,int(frame.shape[0]*0.6),frame.shape[1],frame.shape[0]]
            pt=read_plate(frame,box)
            print(f'[VIOLATION] No Helmet | Plate: {pt}')
            if pt not in fined:
                fined.append(pt)
                logger.log(ann,'helmetless_riding',pt)
            if lp:
                x1,y1,x2,y2=[int(c) for c in lp[0][:4]]
                cv2.putText(ann,f'PLATE: {pt}',(x1,y2+25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,165,0),2)
            else:
                cv2.putText(ann,f'PLATE: {pt}',(20,frame.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,165,0),2)
        if fined:
            cv2.putText(ann,'Fined Plates:',(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            for i,p in enumerate(fined):
                cv2.putText(ann,f'-> {p}',(20,80+i*35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        resized=cv2.resize(ann,(1280,720))
        cv2.imshow(f'Helmet Violation - {img_file}',resized)
        print('Press any key for next, ESC to quit')
        key=cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key==27: break
    cv2.destroyAllWindows()
    print(f'\n[DONE] Fined plates: {fined}')

if __name__ == '__main__':
    main()
