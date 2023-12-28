# 특징점 검출 알고리즘

import cv2 as cv
import numpy as np

img = np.array([[0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0,0,0],
               [0,0,0,1,1,0,0,0,0,0],
               [0,0,0,1,1,1,0,0,0,0],
               [0,0,0,1,1,1,1,0,0,0],
               [0,0,0,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0]],dtype=np.float32)
# float형태의 10*10 크기 입력영상 만들기

ux = np.array([[-1,0,1]]) # 1행 3열 형태의 미분을 위판 필터
uy = np.array([-1,0,1]).transpose() # 3행 1열 형태의 미분을 위한 필터
k = cv.getGaussianKernel(3, 1) # 3*3 가우시안 필터 만든다
g = np.outer(k,k.transpose()) # 3*3 가우시안 필터 만든다
# np.outer(a,b): a,b 벡터의 외적을 계산하기 위함

dy = cv.filter2D(img,cv.CV_32F,uy) 
dx = cv.filter2D(img,cv.CV_32F,uy)
dyy = dy*dy
dxx = dx*dx
dyx = dy*dx
# 식 5.7의 dx, dy, dyy, dxx, dyx를 만든다
# 식 5.7은 실수 계산이 가능하며, (v, u)를 변화시키면서 맵을 생성하는 과정을 거치지 않아도 되고, 
# 식 5.7만 분석하면 지역 특징 여부를 판단할 수 있다.

gdyy = cv.filter2D(dyy, cv.CV_32F, g) # G*dy*dy
gdxx = cv.filter2D(dxx, cv.CV_32F, g) # G*dx*dx
gdyx = cv.filter2D(dyx, cv.CV_32F, g) # G*dx*dy
C = (gdyy*gdxx-gdyx*gdyx)-0.04*(gdyy+gdxx)*(gdyy+gdxx)
# C는 식 5.9의 특징 가능성 맵을 계산하기 위한 식
# 식 5.9는 해리스의 지역 특징일 가능섯ㅇ의 식에서 고유값 게산을 피한 식이다 

# 비최대 억제 적용
for j in range(1,C.shape[0]-1):
    for i in range(1,C.shape[1]-1):
        if C[j,i]>0.1 and sum(sum(C[i,j]>C[j-1:j+2,i-1:i+2]))==8:
            img[j,i]=9
        # C가 0.1보다 커야 하고 인접한 이웃 8개보다 커야지만 극점이 된다
            
np.set_printoptions(precision=2) # 자리수를 설정해주어 반올림하여 출력하게 설정한다
print(dy)
print(dx)
print(dyy)
print(dxx)
print(dyx)
print(gdyy)
print(gdxx)
print(gdyx)
print(C)
print(img)

popping = np.zeros([160,160],np.uint8) 
print(popping)
# 원소가 0인 행렬 만들기 위함
# 16배로 확대하여 화소를 확인 가능하게 하기 위해

for j in range(0,160):
    for i in range(0,160):
        popping[j,i]=np.uint8((C[j//16,i//16]+0.06)*700)
        
cv.imshow('image Display2',popping)
cv.waitKey()
cv.destroyAllWindows()











