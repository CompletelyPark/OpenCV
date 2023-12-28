# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:14:10 2022

@author: 진박완
"""

import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 

canny = cv.Canny(gray,100,200)
'''
cv.canny(img,dst,threshold1,threshold2,aperturesize,L2gradiant)
img - 영상, dst - 출력 영상, threshold1 - 하위 입계값, threshold2 - 상위 임계값
aperturesize - 소벨 연산 마스크 크기
L2gradiant - L2norm으로 정확하게 계산할 것인지, 아니면 L1norm으로 빠르게 계산할 것인지 결정
'''

contour,hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
'''
경계선을 찾아서 contour 객체에 저장
cv.findContours(img, mode, method, contours, hierarchy)
mode - 콘투아 제공방식 cv.RETR_LIST: 가장 바깥쪽 라인만 생성
method - 근사값 방식 cv.CHAIN_APPROX_NONE: 근사 없이 모든 좌표 제공 
contours - 검출한 콘투아 좌표, hierarchy - 콘투아 계층 정보
'''

lcontour = []

for i in range(len(contour)):
    if contour[i].shape[0]>100:
        lcontour.append(contour[i])

cv.drawContours(img,lcontour,-1,(0,255,0),3)
'''
경계선 집합을 지정한 영상에 그려준다
cv.drawContours(img, contours, contourIdx, color, thickness)
contours - 검출한 콘투아 좌표, contourIdx - 외곽선 인덱스 음수일 시 모든 외곽선을 그려준다
color - (0,255,0)녹색, thickness - 두 
'''

cv.imshow('original with contours',img)
cv.imshow('canny',canny)

cv.waitkey()
cv.destroyAllWindows()

