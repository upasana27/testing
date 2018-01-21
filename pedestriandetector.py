import numpy as np
import cv2
#img = cv2.imread('C:\Python27\pil_test.jpg',0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #ret = cap.set(3,480)
    #ret = cap.set(4,640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects= hog.detectMultiScale(frame, winStride=(4, 4),padding=(8,8), scale=1.15)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('img',frame)
    cv2.waitKey(2)
