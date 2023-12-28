# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:16:01 2022

@author: 진박완
"""

import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
img = cv.resize(img, dsize=(0,0),fx=0.4,fy=0.4) # 영상 축소
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 명암 영상으로 변환
cv.putText(gray,'soccer',(10,20),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

'''
스무딩 효과를 보기 위하여 영상에 글씨를 써넣는다
스무딩 효과 = 노이즈 제거 효과, 잡음을 누그러뜨린다

cv2.putText(img, text, org, font, fontScale, color, thickness)
img = 영상, text = 적혀질 텍스트, org = 그려질 텍스트의 왼쪽 아래 모서리 좌표 (x좌표, y좌표),
font = 글꼴, fontscale = 글꼴 크기, color = 텍스트의 색상, thickness = 굵기
'''

cv.imshow('Original',gray)

smooth = np.hstack((cv.GaussianBlur(gray, (5,5), 0.0),cv.GaussianBlur(gray, (9,9), 0.0),
                    cv.GaussianBlur(gray, (15,15), 0.0)))
cv.imshow('Smooth',smooth)

'''
필터 크기 5*5 9*9 15*15로 GaussianBlur 함수를 각각 적용하여 얻은 영상을 np.hstack으로 붙인다
가우시안블러는 중심에 있는 픽셀에 높은 가중치를 부여

cv.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
src = img, ksize= 가우시안 커널의 크기, sigmaX = x방향 sigma, sigmay = y방향 sigma
borderType = 가장자리의 픽셀 확장 방식
'''

femboss = np.array([[-1.0,0.0,0.0],
                    [0.0,0.0,0.0],
                    [0.0,0.0,1.0]
                    ])
# 엠보싱 필터(kernel)

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))
emboss_bad = np.uint8(cv.filter2D(gray16,-1,femboss) + 128)
emboss_worse = cv.filter2D(gray, -1, femboss)

'''
엠보싱 필터는 오른쪽 아래에서 왼쪽 위 화소를 빼기 때문에 음수가 발생
gray 배열은 np.uint8 타입 - 부호가 없는 1바이트 정수 타입
filter2D를 적용한 결과 영상도 값은 타입의 배열을 출력하기 때문에 uint8타입

gray16 = 기존 8비트 타입을 음수를 표현하기 위하여 int16을 사용하여 부호가 있는 16비트 타입으로 변환
결과 영상에 np.uint8을 적용하여 np.uint함수를 적용하여 emboss에 저장
emboss_bad는 np.clip을 적용하지 않았을 때 발생하는 부작용 확인
emboss_worse는 np.uint16을 적용하지 않았을때의 부작용 확인

np.clip(array, min, max)
array 안에서 min 값보다 작은 값은 min 값으로 변경하고 max 값보다 큰 값은 max 값으로 변경

cv.filter2D(src, ddepth, kernel)
src = img, ddepth = 출력 영상의 데이터 타입 -1은 src와 같은 타입, kernel = 필터 마스크 행렬(실수형)
'''

cv.imshow('Emboss',emboss)
cv.imshow('Emboss_bad',emboss_bad)
cv.imshow('Emboss_worse',emboss_worse)

cv.waitKey()
cv.destroyAllWindows()


