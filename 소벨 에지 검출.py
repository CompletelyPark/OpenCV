# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:09:18 2022

@author: 진박완
"""

import cv2 as cv

img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 명암 영상으로 변환

grad_x = cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)
grad_y = cv.Sobel(gray,cv.CV_32F,0,1,ksize=3)
'''
sovel함수 각각 x방향 y방향으로 적용
cv.sobel(img, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None)
img - 영상, ddepth - 출력영상의 데이터 타입, dx - x방향 미분 차수, dy - y방향 미분 차수,
dst - 출력영상(행렬), ksize - 커널크기 기본값이 3, scale - 연산 결과에 추가로 곱할 값, 
delta - 연산 결과에 추가로 더할 값, borderType - 가장자리 픽셀 확장 방식
지금은 gray 영상에 32bit 데이터타입, x,y 각각 미분해주고 있는 형식이
'''

sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)
'''
음수가 포함된 영상을 양수 영상으로 변환시켜준다
cv.convertScaleAbs()
'''

edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
'''
sobel_x 와 sobel_y에 0.5를 곱해서 더한 결과를 저장
cv.addWeighted(img1,a,img2,b,c) img1*a+img2*b+c를 계산해준다
'''

cv.imshow('original',gray)
cv.imshow('sobelx',sobel_x)
cv.imshow('sobely',sobel_y)
cv.imshow('edge strength',edge_strength)

cv.waitKey()
cv.destroyAllWindows()
