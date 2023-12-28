# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:29:59 2022

@author: 진박완
"""

import cv2 as cv
#import numpy as np

img = cv.imread('rose.png')
patch = img[100:200,100:200,:]
# 100*100 패치를 잘라서 patch 객체에 저장

img = cv.rectangle(img,(100,100),(200,200),(255,0,0),2)
'''
잘라낼 곳을 (255,0,0)색으로 표시한다 
cv.rectangle(img,pt1,pt2,color,thickness): img - 영상, pt1 - 시작점 좌표, pt2 - 종료점 좌표
                                           color - 색상, thickness - 선 두께
'''
patch1 = cv.resize(patch, dsize=(0,0), fx = 5, fy = 5, interpolation = cv.INTER_NEAREST)
patch2 = cv.resize(patch, dsize=(0,0), fx = 5, fy = 5, interpolation = cv.INTER_LINEAR)
patch3 = cv.resize(patch, dsize=(0,0), fx = 5, fy = 5, interpolation = cv.INTER_CUBIC)
'''
각각 interpolation을 Inter_nearest, Inter_linear, Inter_cubic 보간 방법 적용하여 5배 확대하여 저장
cv.INTER_NEAREST - 가장 빠르지만 퀄리티가 떨어진다
cv.INTER_LINEAR  - 4개의 픽셀을 이용한다, 속도도 빠르고 퀄리티도 괜찮다
cv.INTER_CUBIC   - 16개의 픽셀을 이용한다, INTER_LINEAR보다는 느리지만 퀄리티는 더 좋다
'''

cv.imshow('original',img)
cv.imshow('resize nearest',patch1)
cv.imshow('resize nearest',patch2)
cv.imshow('resize nearest',patch3)
# 확대한 영상을 보여준다


cv.waitKey()
cv.destroyAllWindows()


