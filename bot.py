import cv2 as cv
import pandas as pd
import numpy as np
import math
from ultralytics import YOLO
from sort import *
from math import dist

#model save
model = YOLO('yolov8s.pt')
#model.eval()

def YOLOV8(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        #print(colorsBGR)        
cv.namedWindow('YOLOV8')
cv.setMouseCallback('YOLOV8',YOLOV8)

my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count=0
counterin=[]
line = [1000,0,1000,780]
#tracker=Tracker()
tracker = Sort(max_age=20,min_hits=3)

cap = cv.VideoCapture('Data/bot.mp4')
fw = int(cap.get(3)) 
fh = int(cap.get(4))
size = (fw, fh)
result = cv.VideoWriter('Data/output.avi',  cv.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:    
    ret,frame = cap.read()
    #frame=cv.resize(frame,(1150,900))

    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    detections = np.empty((0, 5))
             
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        confi=int(row[4])
        conf = math.ceil(confi * 100)
        d=int(row[5])
        c=class_list[d]
        if c == 'bottle':
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            current_detections = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, current_detections))
    tracker_result = tracker.update(detections)
    cv.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 5)

    for track_result in tracker_result:
        x3, y3, x4, y4, id = track_result
        x3, y3, x4, y4, id = int(x3), int(y3), int(x4), int(y4), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        cv.rectangle(frame,(x3,y3),(x4,y4),(170,0,250),2)
        cv.putText(frame, f'{id}', [x3 + 8, y3 - 12],cv.FONT_HERSHEY_COMPLEX,0.8,(0,80,200),2)

        if line[1] < cy < line[3] and line[2] - 10< cx < line[2] + 10:
            cv.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)
            if counterin.count(id) == 0:
                counterin.append(id)
    cv.putText(frame, f'Total Drinks = {len(counterin)}', [500, 34],cv.FONT_HERSHEY_COMPLEX,0.8,(160,60,255),2)

    result.write(frame)
    cv.imshow("YOLOV8", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv.destroyAllWindows()