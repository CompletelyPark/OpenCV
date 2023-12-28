import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential # 모델 생성
from tensorflow.keras.layers import Dense      # fully connected
from tensorflow.keras.optimizers import Adam   # optimizer adam 사용

(x_train, y_train),(x_test,y_test) = ds.cifar10.load_data()       # cifar10 데이터셋 불러오기
x_train = x_train.reshape(50000,3072)                             # 훈련데이터의 형태를 numpy 이용해서 형태 변경
x_test = x_test.reshape(10000,3072)                               # 테스트 데이터의 형태를 numpy 이용해서 형태 변경
x_train = x_train.astype(np.float32)/255.0                        # 훈련데이터의 형태를 numpy 이용해서 형태 변경
x_test = x_test.astype(np.float32)/255.0                          # 테스트 데이터의 형태를 numpy 이용해서 형태 변경
y_train = tf.keras.utils.to_categorical(y_train,10)               # 원 핫 코드 방식으로 변경
y_test = tf.keras.utils.to_categorical(y_test,10)                 # 원 핫 코드 방식으로 변경

dmlp = Sequential()                                                           # 모델생성
dmlp.add(Dense(units = 1024, activation = 'relu', input_shape = (3072,)))     # 입력층 activation 함수 relu 사용, 인풋레이어의 층수 307개 요소를 갖는 1차원 구조로 바꾸어 입력 
dmlp.add(Dense(units = 512, activation = 'relu'))                             # 은닉층 fully connected, activation 함수 relu
dmlp.add(Dense(units = 512, activation = 'relu'))                             # 은닉층 fully connected, activation 함수 relu
dmlp.add(Dense(units = 10, activation = 'softmax'))                           # 출력층 10개, activation 함수 softmax

dmlp.compile(loss = 'categorical_crossentropy',optimizer = Adam(learning_rate = 0.0001),metrics = ['accuracy'])         # 손실함수 교차 엔트로피, optimizer adam에 학습률 0.0001 이용해서 학습
hist = dmlp.fit(x_train,y_train,batch_size = 128, epochs = 50, validation_data = (x_test,y_test),verbose=2)             # epoch 50으로 설정하여 학습시키는 과정, verbose=2 모든 정보 출력
print('정확률 = ',dmlp.evaluate(x_test,y_test,verbose=0)[1]*100)        

import matplotlib.pyplot as plt 

plt.plot(hist.history['accuracy'])                    # 모델의 정확도 출력
plt.plot(hist.history['val_accuracy'])                # 모델의 validation accuracy 출력
plt.title('Accuracy graph')                           # 그래프의 이름 설정
plt.xlabel('epochs')                                  # x좌표 이름 epoch
plt.ylabel('accuracy')                                # y좌표 이름 accuracy
plt.legend(['train','test'])                          # 그래프 안에 범례 추가하기 
plt.grid()                                            # 그래프 격자 표시
plt.show()

plt.plot(hist.history['loss'])                        # 모델의 정확도 출력
plt.plot(hist.history['val_loss'])                    # 모델의 validation accuracy 출력
plt.title('Loss graph')                               # 그래프의 이름 설정
plt.xlabel('epochs')                                  # x좌표 이름 epoch
plt.ylabel('loss')                                    # y좌표 이름 accuracy
plt.legend(['train','test'])                          # 그래프 안에 범례 추가하기 
plt.grid()                                            # 그래프 격자 표시
plt.show()
