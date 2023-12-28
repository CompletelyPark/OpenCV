import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train,y_train),(x_test,y_test)=cifar10.load_data() # cifar 10 불러옴
x_train=x_train.astype(np.float32)/255.0              #  
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)     # 10개의 출력 유형을 위해
y_test=tf.keras.utils.to_categorical(y_test,10)       # 10개의 출력 유형을 위해


cnn=Sequential()                                                        # model 생성
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))       # input 32*32*3 구조 텐서, 3*3 형태 커널 32개
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
res = cnn.evaluate(x_test,y_test,verbose=0)
print('정확률 = ',res[1]*100)

import matplotlib.pyplot as plt

# 학습의 정확률 그려준다
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','validation'])
plt.grid()
plt.show()

# 손실 함수를 그려준다.
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','validation'])
plt.grid()
plt.show()