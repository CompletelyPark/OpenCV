# RANSAC을 이용한 호모그래피 추정 
import cv2 as cv
import numpy as np

img1 = cv.imread('mot_color70.jpg')[190:350,440:560] # mot_color70 영상에서 일부분만 물체 모델 영상을 정해주기 위함
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2 = cv.imread('mot_color83.jpg')                  # 장면 영상을 정해준다
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create() 
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)
'''
두 영상 각각에서 SIFT 특징점을 검출하고 기술자를 추출한다.
cv.SIFT_create(	[, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]])
detectAndCompute(image, mask, decriptors, useProvidedKeypoints)
image = 영상, mask = 특징점 검출에 사용할 마스크, descriptors = 계산된 특징점, 
useProvidedKeypoints(optional): True인 경우 특징점 검출을 수행하지 않음
'''

flann_matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1,des2,2)
'''
FLANN 라이브러리 사용하여 flann_matcher 객체 생성 
des1과 des2 인수를 주어서 knnmatch 함수를 통해 최근점 2개를 찾으라고 한다.
DescriptorMatcher_create(matcherType) 매칭기 생성자, matcherType에는 Bruteforce, bruteforcehamming, Flanbased등이 있다.
knnMatch(queryDescriptors, trainDescriptors, k, mask): k개의 가장 인접한 매칭 찾기 
'''

T = 0.7
good_match = []
for nearest1,nearest2 in knn_match:
    if(nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)
# 최근접 이웃 거리 비율 전략을 이용하여 최근접 점 2개쌍들의 거리를 계산하여 t보다 작은 좋은 쌍을 찾아내어 비어있는 리스트에 넣어준다        

points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])
# 좋은 쌍 들이 들어있는 리스트에서 매칭된 쌍들에서 첫번째 영상의 좌표를 알아내 points1에 저장하고 
# 두번째 영상의 특징점 좌표를 알아내 points2에 저장한다

H,_ = cv.findHomography(points1, points2,cv.RANSAC)
'''
findHomography함수를 이용해 호모그래피 행렬 추정후 H에 저장
findHomography함수는 points1,points2를 가지고 RANSAC알고리즘을 수행해서 행렬 추정
cv.RANSAC 모든 좌표를 사용하지 않고 임의의 좌표만 선정해서 만족도를 구하는 방식, 
이렇게 구한 만족도가 큰 것만 선정하여 계산
'''

h1, w1 = img1.shape[0], img1.shape[1] # 첫번째 영상의 높이 넓이
h2, w2 = img2.shape[0], img2.shape[1] # 두번째 영상의 높이 넓이


box1 = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(4,1,2) # 첫번째 영상의 모서리 4개의 좌표를 저장
box2 = cv.perspectiveTransform(box1, H)                                 
# reshape를 통해 box1의 배열 모양을 perspectiveTransform을 통하여 원하는 모양으로 변환 
# 첫번째 영상의 좌표에 호모그래피 행렬을 적용해 두번째 영상 투영하고 결과 저장

img2 = cv.polylines(img2, [np.int32(box2)], True, (0,255,0),8)
# polylines 함수로 box2를 2번째 영상에 그러녛는다.

img_match = np.empty((max(h1,h2),w1+w2,3),dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match,
               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

'''
두 영상을 나란히 띄우고 각 특징점 정보, 시프트 특징점과 기술자를 그려넣어 준다.
cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, flags=None)
img1, keypoints1 = 기준 영상과 영상에서 추출한 특징점 정보
img2, keypoints2 = 대상 영상과 영상에서 추출한 특징점 정보
matches1to2 = 매칭 정보, 현재는 좋은 쌍이 들어있는 리스트
outImag = 출력해줄 영상, flags: 매칭 정보 그리기 방법
'''

cv.imshow('Matches and Homography',img_match)

k = cv.waitKey()
cv.destroyAllWindows()

