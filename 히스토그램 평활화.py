# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:12:16 2022

@author: 진박완
"""

import cv2 as cv

import matplotlib.pyplot as plt

img = cv.imread('mistyroad.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 명암영상으로 변환하기 위하여 cv.COLOR_BGR2GRAY를 사용하여서 변환

plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([gray],[0],None,[256],[0,256])
plt.plot(h,color='r',linewidth=1), plt.show()

'''
영상의 히스토그램을 구하여서 출력
cv.calcHist(img, channels, mask, histsize, ranges[, hist[, accumulate]])
img = uint8 또는 float32 타입의 이미지를 사용해야하며 대괄호[] 안에 입력해야 한다
channels = 히스토그램을 계산할 채널의 인덱스, 대괄호[] 안에 입력
            gray스케일 이미지 라면 0, 컬러 이미지라면 0,1,2 파랑, 녹색, 빨강의 순서 
mask = 마스크 이미지, 전체 이미지에 대한 히스토그램을 구할 거면 None을 사용
histsize = 계산할 히스토그램 막대의 개수, 대괄호[] 안에 입력, 전체 영역이라면 256
ranges = 히스토그램을 계산할 범위, 전체 범위라면 [0,256]

plt.plot(img, color, linewidth)
img = 영상, color = 색깔, linewidth = 굵기
'''


equal = cv.equalizeHist(gray)
plt.imshow(equal,cmap = 'gray'), plt.xticks([]), plt.yticks([]), plt.show()
'''
명암 영상을 히스토그램 평활화를 적용시켜서 그 결과 영상을 출력한다
cv.equalizeHist(src,dst)
src = 대상 이미지 8비트 1채널, dst = 선택사항으로 결과 이미지

'''
h = cv.calcHist([equal],[0],None,[256],[0,256])
plt.plot(h,color='r',linewidth=1), plt.show()
# 히스토그램 평활화된 영상의 히스토그램을 구하고 출력해준다








