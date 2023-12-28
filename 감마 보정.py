# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:08:27 2022

@author: 진박완
"""

# 감마보정을 위한 프로그램
import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
img = cv.resize(img, dsize=(0,0),fx=0.25,fy=0.25)
# 이미지 사이즈 4분의 1 크기로 조정

def gamma(f, gamma=1.0): 
    
    '''
    f = 감마보정을 적용할 영상 
    gamma = 감마보정값, 기본값 : 1.0
    '''
    
    f1 = f/255.0 
    # 최대 gray level을 256이라고 가정하고 255.0으로 나누어서 [0,1] 범위로 정규화를 시킨다
    
    return np.uint8(255*(f1**gamma))
    # 화소값에 gamma제곱을 적용한다음에 (L-1)값을 곱한다 그 결과를 8비트 정수형으로 바꾸어서 변환

gc = np.hstack((gamma(img,0.5),gamma(img,0.75),gamma(img,1.0),gamma(img,2.0),gamma(img,3.0)))

'''
gamma를 각 값으로 변화시키면서 gamma 함수를 적용한 영상 5개를 hstack 함수로 이어 붙인다
np.hstack() 배열을 옆으로 붙이고 싶을때 사용한다
지금 같은 경우네는 감마보정이 된 이미지 5개를 이어 붙이기 위해 사용한다
'''

cv.imshow('gamma',gc)
# 감마보정된 이미지 출력

cv.waitkey()
cv.destroyAllWindows()









