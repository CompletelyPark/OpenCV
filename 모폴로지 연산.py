# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:00:55 2022

@author: 진박완
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('JohnHancocksSignature.png',cv.IMREAD_UNCHANGED)

t,bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY+ cv.THRESH_OTSU)

'''
img의 3번 채널에 오츄 이진화를 적용항 결과를 bin_img에 저장한다
cv.threshold(img, threshold_value, value, flag)
img = grayscale 이미지, threshold_value = 픽셀의 문턱값, value = 픽셀 문턱값보다 클 때 적용되는 최대값
flag = 문턱값 적용 방법 또는 스타일
cv.THRESH_BINARY = 픽셀 값이 threshold_value보다 크면 value 작으면 0으로 할당
cv.THRESH_OTSU = 오츄 알고리즘 적용하기 위하여 사용
'''

plt.imshow(bin_img,cmap='gray'),plt.xticks([]),plt.yticks([])
# imshow 함수로 bin_img를 출력 cmap='gray'는 영상을 명암으로 출력하기 위하여 지정
plt.show()

b= bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
plt.imshow(b,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

'''
모폴로지 효과를 확인할 목적으로 영상의 일부를 잘라서 b에 저장하여 잘라낸 b를 출력하는 코드
모폴로지 연산에는 침식, 팽창, 열림, 닫힘이 있다
'''

se = np.uint8([[0,0,1,0,0],
              [0,1,1,1,0],
              [1,1,1,1,1],
              [0,1,1,1,0],
              [0,0,1,0,0]]
              )
# 8비트로 구조요소를 저장하기 위하여 넘파이 함수를 사용했다

b_dilation = cv.dilate(b,se,iterations = 1)
plt.imshow(b_dilation,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

'''
팽창 연산
cv.dilate(img, kernel, iterations)
img = 팽창을 위한 이미지, kernel = 팽창을 위한 커널 (위의 넘파이로 만들어진 구조요소),
iterations = 팽창 반복 횟수
'''

b_erosion = cv.erode(b,se,iterations = 1)
plt.imshow(b_erosion,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

'''
침식연산
cv.erode(img, kernel, iterations)
img = 침식을 위한 이미지, kernel = 침식을 위한 커널 (위의 넘파이로 만들어진 구조요소),
iterations = 침식 반복 횟수
'''

b_closing = cv.erode(cv.dilate(b,se,iterations = 1),se,iterations=1)
plt.imshow(b_closing,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

'''
팽창후 침식 - 닫기 연산
침식후 팽창 - 열기 연산
'''