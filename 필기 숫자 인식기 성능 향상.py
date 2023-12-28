import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
# mnist 데이터 불러오기
x_train=x_train.reshape(60000,28,28,1)
# train 데이터를 reshape 해준다.
x_test=x_test.reshape(10000,28,28,1)
# test 데이터를 reshape 해준다
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)


cnn=Sequential()
# model 생성
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))       # input 28*28*1 구조 텐서, 3*3 형태 커널 32개
cnn.add(Conv2D(32,(3,3),activation='relu'))                             # 3*3 형태 커널 32개
cnn.add(MaxPooling2D(pool_size=(2,2)))                                  # maxpooling 2*2
cnn.add(Dropout(0.25))                                                  # dropout 0.25 아래의 가중치 랜덤 제거
cnn.add(Conv2D(64,(3,3),activation='relu'))                             # 3*3 형태 커널 64개
cnn.add(Conv2D(64,(3,3),activation='relu'))                             # 3*3 형태 커널 64개
cnn.add(MaxPooling2D(pool_size=(2,2)))                                  # maxpooling 2*2
cnn.add(Dropout(0.25))                                                  # dropout 0.25 아래의 가중치 랜덤 제거
cnn.add(Flatten())                                                      # 특징 맵을 펼쳐준다.
cnn.add(Dense(512,activation='relu'))                                   # 완전 연결층 activation = relu  
cnn.add(Dropout(0.5))                                                   # dropout 0.5 아래의 가중치 랜덤 제거
cnn.add(Dense(10,activation='softmax'))                                 # 10개 유형 출력층 activation = softmax


# 신경망 모델을 학습하고 평가하기
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate = 0.001),metrics=['accuracy'])
# optimizer = adam, loss 함수 = categorical_crossentropy

hist = cnn.fit(x_train,y_train,batch_size=128,epochs=100,verbose=0)
# epoch 100, batch 크기 128

cnn.save('cnn_v2.h5')
# 학습 모델을 h5파일로 저장해준다.

res = cnn.evaluate(x_test,y_test,verbose=0)
print('정확률 = ',res[1]*100)