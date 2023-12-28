import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import winsound

model = tf.keras.lodels.load_model('cnn_v2.h5')
# 이전에 학습된 모델 불러오기

def reset():
      global img                                                        # img를 전역변수로 선언하여 다른 함수에서도 사용가능하게 한다.
      img = np.ones((200,520,3),dtype=np.uint8)*255                     # np.ones 함수를 통해서 200*520의 3채널 컬러 영상을 저장할 수 있는 배열 만든다 ones라서 1로 초기화 된 배열을 255를 곱해서 
                                                                        # 모든 화소가 흰색인 배열을 만들어준다.
      for i in range(5):
            cv.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255))
      cv.putText(img,'e:erase             s:show                  r:recognition q:quit',(10,40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1)
      # 지정한 위치에 빨간 박스를 그려주고 각 명령어에 해당하는 글씨를 font = FONT_HERSHEY_SIMPLEX, 크기 = 0.8, 색깔 = red 형태로 써넣어준다. 


def grab_numerals():
      numerals =[]                                                                  # 빈 리스트 생성
      for i in range(5):
            roi = img[51:149,11+i*100:9+(i+1)*100,0]                                # img에서 숫자를 꺼내준다           
            roi = 255-cv.resize(roi,(28,28),interpolation=cv.INTER_CUBIC)           # 28*28 크기로 변환하는 작업, 보간은 cv.INTER_CUBIC - 3차 회선 보간법 16개의 픽셀을 이용한다. INTER_LINEAR보다 느리지만 성능은 좋다
            numerals.append(roi)                                                    # 꺼내진 숫자를 빈 리스트에 추가한다. 
      numerals = np.array(numerals)                                                 # 리스트를 numpy 배열로 바꿔준다.
      return numerals                                                               # 배열 반환

def show(): # 5개의 숫자를 표시해주는 함수 
      numerals = grab_numerals()
      plt.figure(figsize = (25,5))                                                  # 넓이 25 높이 5인 화면 만들어준다.
      for i in range(5):
            plt.subplot(1,5,i+1)                                                    # 1행 5열의 subplot을 만들어준다. 각 index는 i+1
            plt.imshow(numerals[i],cmap = 'gray')                                   # 순서대로 img를 그려준다
            plt.xticks([]); plt.yticks([])                                          # x좌표와 y좌표의 눈금 표시해준다.
      plt.show()

def recognition():
      numerals = grab_numerals()                                                     
      numerals = numerals.reshape(5,784)                                            # 신경망에 입력을 위하여 배열의 형태를 바꾸어준다. 
      numerals = numerals.astype(np.float32)/255.0                                  # astype으로 실수타입으로 바꾸고 255로 나누어서 0~1사이의 범위로 변환한다.
      res = model.predict(numerals)                                                 # predict로 기존모델을 사용해서 현재 img를 예측한다.
      class_id = np.argmax(res,axis=1)                                              # 신경망 출력을 부류 정보로 해석하는 작업 예측하는 작업이다. 
      for i in range(5):
            cv.putText(img,str(class_id[i]),(50+i*100,180),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)                       # 인식결과를 나타내기 위한 작업
      winsound.Beep(1000,500)                                                       # 윈도우 사운드로 알려준다.

BrushSiz = 4                                                                         
LColor = (0,0,0)

def writing(event,x,y,flags,param):                                                 # 마우스 왼쪽 버튼을 누르거나 (cv.EVENT_LBUTTONDOWN), 누른채 이동하면 (cv.EVENT_FLAG_LBUTTON, cv.EVENT_MOUSEMOVE)
                                                                                    # BrushSiz 의 크기로 LColor의 색으로 원을 그려준다. 
      if event==cv.EVENT_LBUTTONDOWN:                                               
            cv.circle(img,(x,y),BrushSiz,LColor,-1)                                 
      elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:          
            cv.circle(img,(x,y),BrushSiz,LColor,-1)

reset()                                                                             # reset 함수 호출 img 영상을 만들어준다.
cv.namedWindow('Writing')                                                           # 뜨는 창의 이름을 Writing으로 한다
cv.setMouseCallback('Writing',writing)                                              # 윈도우의 callback 함수로 writing 함수를 사용한다.

while(True):                                                                        # 무한루프 돌면서
      cv.imshow('Writing',img)      
      key = cv.waitKey(1)                                                           # img 영상을 표시한다.
      if key == ord('e'):                                                           # e 키가 눌리면 reset 함수 사용
            reset()                                                     
      elif key == ord('s'):                                                         # s 키가 눌리면 show 함수 사용
            show()
      elif key == ord('r'):                                                         # r 키가 눌리면 recognition 함수 사용
            recognition()
      elif key == ord('q'):
            break

cv.destroyAllWindows()                                                              # 모든 창을 끄는 작업


