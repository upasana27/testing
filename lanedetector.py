import cv2
import numpy as np
from PIL import Image
import math
cap = cv2.VideoCapture(1)
while(True):
    ret, frame = cap.read()
    img=cv2.imread('C:\\Users\\SR\\Desktop\\cordova1\\f00206.png')
    res =img[220:300,250:400] 
    gray =cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((6,6),np.uint8)
    top= cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    print res.shape
   

    kernel_size = 1
    #blur = cv2.GaussianBlur(top,(1,1),0)

    v = np.median(gray)
    sigma=0.33
    lower = int(max(0, (1.0 - sigma)*v))
    upper = int(min(255, (1.0 + sigma)*v))
    canny = cv2.Canny(gray,lower,upper)

    lines = cv2.HoughLinesP(canny,1,np.pi/180,10,minLineLength=1,maxLineGap=1)
    lmaxx2=0;
    lminy2=0;
    lminx1=0;
    lmaxy1=0;

    rminx1=150;
    rminy1=0;
    rmaxx2=150;
    rmaxy2=0;


    for x1,y1,x2,y2 in lines[0]:

        tan=(y2-y1)/(x2-x1);
    #print tan
    angle=np.arctan(tan)*180/np.pi
    #cv2.line(res,(x1,y1),(x2,y2),(0,255,0),2)
    #print angle
        if angle<0:
        
            if(x2>lmaxx2):
                lmaxx2=x2;
            if(y2<lminy2):
                lminy2=y2;
            if(x1<lminx1):
                lminx1=x1;
            if(y1>lmaxy1):
                lmaxy1=y1;
        if angle>0:
            cv2.line(res,(x1,y1),(x2,y2),(255,0,0),2)
            print x1,y1,x2,y2
            if(x1<rminx1):
                rminx1=x1;
    
        
            if(y2>rmaxy2):
                rmaxy2=y2;

    #cv2.circle(img,((lmaxx2+220),(lminy2+250)),5, (0,0,255), -1)
    #cv2.circle(res,((lminx1+220),(lmaxy1+250)),5, (0,0,255), -1)
    #cv2.circle(res,(rminx1,rminy1),5, (0,0,255), -1)
    #v2.circle(res,(rmaxx2,rmaxy2),5, (0,0,255), -1)
    cv2.line(img,((lminx1+200),(lmaxy1+295)),((lmaxx2+230),(lminy2+250)),(0,0,255),5)
    cv2.line(img,((rminx1+240),(rminy1+220)),((rmaxx2+295),(rmaxy2+320)),(0,0,255),5)
    cv2.imshow('mask',img)

    cv2.waitKey(1)
