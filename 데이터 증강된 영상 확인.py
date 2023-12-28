import tensorflow.keras.datasets as ds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
# cifar10 데이터 불러오기
x_train=x_train.astype('float32')/255.0
x_train = x_train[0:15]; y_train = y_train[0:15]
class_names=['airplane','automobile','bird','cat','deer','dog','flog','horse','ship','truck']
# 각 클래스 이름 설정


plt.figure(figsize=(20,2))
plt.subtitle('first 15 images in the train set')
for i in range(15):
      plt.subplot(1,15,i+1)
      plt.imshow(x_train[i])
      plt.xticks([]); plt.yticks([])
      plt.tilte(class_names[int(y_train[i])])
plt.show()
# 15개의 영상을 보여준다.

batch_siz = 4
# 미니 배치 크기 4
generator = ImageDataGenerator(rotation_range=20.0,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip = True)
# 변환방식과 범위 설정: 회전 -20도에서 20도, 가로 방향 이동 20% 이내, 세로 방향 이동 20% 이내, 좌우반전 시도
gen= generator.flow(x_train, y_train, batch_size = batch_siz)
# x_train에서 데이터 증강 미니배치 크기 지정

for i in range(3):
      im,label = gen.next()
      # 호출할 때마다 지정한 만큼 데이터를 증강하여 생성, img와 label에 저장
      plt.figure(figsize=(8,2.4))
      plt.suptitle('Generatior trial'+str(a+1))
      for i in range(batch_siz):
            plt.subplot(1,batch_siz,i+1)
            plt.imshow(img[i])
            plt.xticks([]); plt.yticks([])
            plt.tilte(class_names[int(label[i])])
      plt.show()
      # 데이터 증강된 4장의 영상 보여준다.
