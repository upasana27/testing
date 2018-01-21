import cv2
import numpy as np
import pytesseract
from PIL import Image
import math
#img=cv2.imread('C:\Users\SR\Desktop\img\speed16.jpg')
#img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

    areas=[cv2.contourArea(c)
         for c in contours]
                    max=np.argmax(areas)
    cnt=contours[max]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    img=img[y:y+h,x:x+w]
    '''kernel=np.ones((5,5),np.uint8)
    img=cv2.dilate(img,kernel,iterations=5)
    img=cv2.erode(img,kernel,iterations=5)'''

    height,width,depth=img.shape 
    print img.shape
 
    scalingx=float(100)/width
    scalingy=float(100)/height
    res=cv2.resize(img,None,fx=scalingx,fy=scalingy,interpolation=cv2.INTER_LINEAR)
    res=res[20:70,20:85]
    cv2.imwrite('pil_test.jpg',res)
    img =Image.open('C:\Python27\pil_test.jpg')
    img.load()
    i=pytesseract.image_to_string(img,lang='eng')
    print i
    if(len(i)==0):
        print "empty"
    cv2.imshow('im',res)
    cv2.waitKey(1)
