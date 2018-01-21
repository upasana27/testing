import cv2
import numpy as np
import pytesseract
from PIL import Image
import math
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    img_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = mask0+mask1
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255,255,255),5)

    areas=[cv2.contourArea(c)for
    c in contours]
    max=np.argmax(areas)
    cnt=contours[max]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow(frame)
    cv2.waitkey(0)
