# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:18:05 2022

@author: 진박완
"""

import cv2 as cv

img = cv.imread("apples.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 

apples = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200,
                         param1 = 150,param2=20,minRadius=50,maxRadius=120)
'''
cv.HoughCircles(img, method, dp, min_dist, parameter1, parameter2, min_Radius, max_Radius)
method - cv.HOUGH_GRADIENT: 에지 방향 정보를 추가로 사용하는 방법, dp - 영상 해상도,
param1 - canny edge에서 높은 threshold값, param2 - 원 검출을 위한 정보, 
min_Radius - 검출 될 원의 최소 반지름, max_Radius - 검출 될 원의 최대 반지름
'''

for i in apples[0]:
    cv.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)


cv.imshow('Apple detection',img)

cv.waitKey()
cv.destroyAllWindows()
