from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('C:/Users/arjav/OneDrive/Desktop/object_detection (1)/people_counter/people_counter.mp4')

mask = cv2.imread('C:/Users/arjav/OneDrive/Desktop/object_detection (1)/people_counter/mask.png')

model = YOLO('C:/Users/arjav/OneDrive/Desktop/object_detection (1)/yolov8n.pt')

yolo_map = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

traker = Sort(max_age=20,min_hits=2,iou_threshold=0.3)
# trakerD = Sort(max_age=20,min_hits=2,iou_threshold=0.3)

countU = []
countD = []
while True:
    
    suc,img = cap.read()
    
    imgd = cv2.bitwise_and(img,mask)
    res = model(imgd,stream=True)
    
    detection = np.empty((0,5))
    # detectionD = np.empty((0,5))
    # cv2.imshow("img1",imgd)
    
    for r in res:
        boxes = r.boxes
        for b in boxes:
            x1,y1,x2,y2 = b.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(234,34,123))q
            
            bbox = x1,y1,x2-x1,y2-y1
            # cvzone.cornerRect(img,bbox,l=6)
            conf = (math.ceil(b.conf*100))/100
            cla = int(b.cls[0])
            
            if yolo_map[cla] == 'person' and conf>0.3:
                # cvzone.putTextRect(img,f'{conf} {yolo_map[cla]}',(max(0,x1),max(0,y1)),scale = 1 ,thickness=1,colorB = (23,100,100))
                # cvzone.cornerRect(img,bbox,l=20,t=3,rt=1)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack([detection,currentArray])
                
    resultTracker = traker.update(detection)
    cv2.line(img,(450,90),(500,150),(45,34,234),thickness=2)
    cv2.line(img,(530,170),(590,240),(45,34,234),thickness=2)
    
    for r in resultTracker:
        x1,y1,x2,y2,id = r
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        
        cx,cy = (x1+x2)//2,(y1+y2)//2
        
        cv2.circle(img,(cx,cy),3,(46,210,200),cv2.FILLED)
        cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=5,t=1,rt=0)
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(0,y1)),scale =1 ,thickness = 1,offset=1)
        
        if 450 < cx < 500 and 110 < cy < 120:
            if countD.count(id)==0:
                countD.append(id)
                cv2.line(img,(450,90),(500,150),(145,134,34),thickness=2)
        
        if 530 < cx < 590 and 190 < cy < 220:
            if countU.count(id)==0:
                countU.append(id)
                cv2.line(img,(530,170),(590,240),(145,134,34),thickness=2)

                
    up = cv2.imread('up arrow.png',cv2.IMREAD_UNCHANGED)
    if up.shape[2] == 3:
        up = cv2.cvtColor(up, cv2.COLOR_BGR2BGRA)
    
    do = cv2.imread('down arrow.png',cv2.IMREAD_UNCHANGED)
    if do.shape[2] == 3:
        do = cv2.cvtColor(do, cv2.COLOR_BGR2BGRA)
        
    wi = cv2.imread('white_bg.jpg',cv2.IMREAD_UNCHANGED)
    if wi.shape[2] == 3:
        wi = cv2.cvtColor(wi, cv2.COLOR_BGR2BGRA)
    # print(up.shape)
    img = cvzone.overlayPNG(img,up,[30,30])
    img = cvzone.overlayPNG(img,do,[130,30])
    img = cvzone.overlayPNG(img,wi,[80,30])
    img = cvzone.overlayPNG(img,wi,[180,30])
    cv2.putText(img,f'{len(countU)}',(85,75),2,2,(34,13,3),cv2.FONT_HERSHEY_COMPLEX)
    cv2.putText(img,f'{len(countD)}',(185,75),2,2,(34,13,3),cv2.FONT_HERSHEY_COMPLEX)
    # cv2.putText(img,f'Up:{len(countU)} Down:{len(countD)}',(100,100),2,2,(34,131,3),cv2.FONT_HERSHEY_COMPLEX)
    cv2.imshow("img",img)
    
    intr  = cv2.waitKey(1)
    if intr & 0xFF == ord('q'):
        break