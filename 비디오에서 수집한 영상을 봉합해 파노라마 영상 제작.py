'''
파노라마 프로그램
파노라마 영상 제작 순서 
1. 영상 준비 (이 때 영상의 크기 큰 것을 대비해서)
2. resize (영상의 크기를 줄여서 빠르게 많은 양을 처리한다)
3. stitch (공통 되는 부분을 찾아서 2d변형 후 연결한다.)
# 2d 변형이 필요한 이유로는 같은 영상이더라도 위치나 각도에 따라 다르게 보이기 때문이다.
'''

from PyQt5.QtWidgets import * # pyqt5는 윈도우에서 gui를 위한 라이브러리
import cv2 as cv
import numpy as np
import winsound # 소리를 위한 라이브러리
import sys

class Panorama(QMainWindow):
    
    def __init__(self): # 전체 초기화 함수

        super().__init__()
        self.setWindowTitle('파노라마 영상') 
        self.setGeometry(200,200,700,200)
        # 열릴 윈도우의 이름과 위치 설정


        collectButton = QPushButton('영상 수집',self)
        self.showButton = QPushButton('영상보기',self)
        self.stitchButton = QPushButton('봉합',self)
        self.saveButton = QPushButton('저장',self)
        quitButton = QPushButton('나가기',self)
        self.label = QLabel('환영합니다',self)
        # 각 버튼의 이름 설정, label은 상태 표기를 위함


        collectButton.setGeometry(10,25,100,30)
        self.showButton.setGeometry(110,25,100,30)
        self.stitchButton.setGeometry(210,25,100,30)
        self.saveButton.setGeometry(310,25,100,30)
        quitButton.setGeometry(450,25,100,30)
        self.label.setGeometry(10,70,600,170)
        # 각 버튼의 위치(x,y,z) 및 크기 설정
        

        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        # 처음에 버튼 비활성화
        

        collectButton.clicked.connect(self.collectFunction)
        self.showButton.clicked.connect(self.showFunction)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
        # 각 버튼들에 해당하는 기능을 가진 함수를 매칭


    def collectFunction(self): # 영상 수집 버튼
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.label.setText('c를 여러번 눌러 수집하고 끝나면 q를 눌러 비디오를 끕니다')
        # 영상 수집 버튼을 누르면 다른 버튼들은 비활성화 되고 텍스트를 표시한다


        self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)
        #self.cap = cv.VideoCapture('http://192.168.0.3:4747/video',cv.CAP_DSHOW)
        if not self.cap.isOpened():sys.exit('카메라 연결 실패')
        # 연결되어 있는 카메라에 접속을 시도하며 카메라가 없을시 연결에 실패하고 종료한다.
        # 주석은 스마트폰과 연결하여서 프로그램 실행하기 위한 코드


        self.imgs=[]
        # 수집된 영상들을 저장할 비어있는 리스트 만들어 준다 
        while True:
            ret,frame = self.cap.read()
            if not ret: break
            
            cv.imshow('video display',frame)
            
            key = cv.waitKey(1)
            
            if key == ord('c'):
                self.imgs.append(frame)

            elif key == ord('q'):
                self.cap.release()
                cv.destroyWindow('video display')
                break

            # c 키를 누를때마다 영상을 imgs에 저장하고 q를 누르면 수집을 종료한다

        if len(self.imgs)>=2:
            self.showButton.setEnabled(True)
            self.stitchButton.setEnabled(True)
            self.saveButton.setEnabled(True)
        # 영상이 2장 이상 저장됐으면 처음에 비활성화 됐던 버튼들을 활성화상태로 바꾼다.
            

    def showFunction(self):
        self.label.setText('수집된 영상은 ' +str(len(self.imgs))+'장 입니다')
        stack = cv.resize(self.imgs[0],dsize=(0,0),fx=0.25,fy=0.25)
        for i in range(1,len(self.imgs)):
            stack = np.hstack((stack,cv.resize(self.imgs[i],dsize=(0,0),fx=0.25,fy=0.25)))
        cv.imshow('image collector',stack)
        
        # 수집한 영상을 보여주기 위한 함수
        # resize 함수를 통해서 수집한 영상을 4분의 1로 줄여주고 hstack 함수로 이어붙인다.

    def stitchFunction(self):
        stitcher = cv.Stitcher_create()
        status, self.img_stitched = stitcher.stitch(self.imgs)

        #print(status)
        if status == cv.STITCHER_OK:
            cv.imshow('image stitched panorama',self.img_stitched)
        else :
            winsound.Beep(3000, 500)
            self.label.setText('파노라마 제작에 실패')
        
        '''
        수집한 영상을 기준점들을 잡아서 파노라마영상을 제작해준다.
        status가 0인 경우(stitching 성공) 이미지를 아까 받은 이름으로 저장하고 보여준다.
        status가 0이 아닌 경우(stitching 실패 예외)의 메시지를 출력한다.
        1 : 이미지를 연결 시키기에 match point가 부족해서 나오는 에러, 이미지를 더 추가시켜줘야 한다.
        2 : 2D 이미지 변환을 하지 못하는 에러, 이미지를 다시 찍어야 한다..
        3 : 카메라 위치의 에러, 카메라의 방향이 잘못돼서 나오는 에러, 
            입력 이미지들을 같은 방향으로 회전시키거나 새로운 이미지를 찍어야 한다.
        '''
        

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self,'파일 저장','./')
        cv.imwrite(fname[0],self.img_stitched)
        
        # 파노라마 영상을 설정한 이름으로 저장해준다.


    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()

        # 종료해주는 함수

app = QApplication(sys.argv)
win = Panorama()
win.show()
app.exec_()
        
'''
처음에 프로그램을 맞게 짰음에도 불구하고 stitch가 되지 않아서 status를 출력해봤더니 1이 계속 나왔습니다.
인터넷에 찾아보니 1 같은 경우에는 match point 즉 연결되는 부분에 대한 정보가 부족하다고 해서 
너무 끊어서 찍은 것이 아닐까 생각하고 좀 더 붙여서 찍어서 실행해봤더니 성공적으로 stitch가 됐습니다.
추가로 6.6의 스마트폰과 연동하여서 하는 코드도 실행해봤습니다.
'''
        
        
