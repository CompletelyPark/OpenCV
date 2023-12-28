import cv2 as cv
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
# resnet50 모듈에서 클래스와 함수를 불러온다.

model = ResNet50(weights = 'imagenet')
# resnet50 클래스로 예비학습 모델을 읽어 model에 저장 imagenet으로 학습된 가중치를 읽어오게 함

img = cv.imread('rabbit.jpg')
# 토끼 사진 읽어오기
x = np.reshape(cv.resize(img,(224,224)),(1,224,224,3))
# resnet50모델의 입력 크기인 224*224로 변환하고 224*224*3 텐서를 1*224*224*3 텐서로 변환
x = preprocess_input(x)
# resnet50 모델이 신경망 입력 전에 수행하는 전처리 적용

preds = model.predict(x)
# predict 함수로 예측 수행하고 저장
top5 = decode_predictions(preds,top=5)[0]
# 1000개 확률 중에 가장 큰 5개 확률 취하고 부류 이름을 같이 제공하게 한다.
print('예측 결과:',top5)

for i in range(5):
      cv.putText(img,top5[i][1]+':'+str(top5[i][2]),(10,20+i*20),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

cv.imshow('Recognition result',img)
# 영상의 정보를 윈도우에 표시

cv.waitKey()
cv.destroyAllWindows()