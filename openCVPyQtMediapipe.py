from PyQt5.QtWidgets import *
import cv2 as cv
from mediapipe.python.solutions import face_detection
import numpy as np
import mediapipe as mp
import sys

class PProject(QMainWindow):

      def __init__(self):

            super().__init__()
            self.setWindowTitle('프로젝트')
            self.setGeometry(200,200,900,800)


            Button1 = QPushButton('사용법',self)
            faceButton1 = QPushButton('얼굴 인식1',self)
            faceButton2 = QPushButton('얼굴 인식2',self)
            handbutton = QPushButton('손',self)
            posebutton = QPushButton('포즈',self)
            quitButton = QPushButton('나가기',self)

            self.label = QLabel('안녕하세요',self)

            Button1.setGeometry(10,25,100,30)
            faceButton1.setGeometry(110,25,100,30)
            faceButton2.setGeometry(210,25,100,30)
            handbutton.setGeometry(310,25,100,30)
            posebutton.setGeometry(410,25,100,30)
            quitButton.setGeometry(650,25,100,30)
            self.label.setGeometry(10,70,700,400)
  
            
            Button1.clicked.connect(self.Button1F)
            faceButton1.clicked.connect(self.facefunction1)
            faceButton2.clicked.connect(self.facefunction2)
            handbutton.clicked.connect(self.handfunction)
            posebutton.clicked.connect(self.posefunction)
            quitButton.clicked.connect(self.quitFunction)


      def Button1F(self): # 설명

            self.label.setText(' 버튼을 누르면 각 기능이 실행됩니다. 각 창은 q를 누르면 종료됩니다.\n\n 얼굴인식 1: 얼굴의 눈 코 입 귀 얼굴 부분을 보여줍니다.\n\n 얼굴인식 2: 얼굴을 mesh로 보여주고 눈 눈썹 입술을 표현합니다.\n\n 손 : 손의 각 관절 부분을 보여줍니다.\n\n 포즈 : 사람이 취하고 있는 포즈가 나타납니다')

      def facefunction1(self): 

            self.label.setText('얼굴의 눈 코 입 귀 얼굴 부분을 보여줍니다.') 

            mp_face_detection = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils

            self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)
            if not self.cap.isOpened():sys.exit('카메라 연결 실패')

            with mp_face_detection.FaceDetection(
                  model_selection = 0, min_detection_confidence = 0.5) as face_detection:

                  while self.cap.isOpened():
                        ret,frame = self.cap.read()

                        frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
                        frame.flags.writeable = False
                        results = face_detection.process(frame)

                        frame.flags.writeable = True
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                        if results.detections:
                              for detection in results.detections:
                                    mp_drawing.draw_detection(frame, detection)

                        cv.imshow('Detection', frame)
                        key = cv.waitKey(1)

                        if key == ord('q'):
                              self.cap.release()
                              cv.destroyWindow('Detection')
                              break
                     

      def facefunction2(self):

            self.label.setText('얼굴을 mesh로 보여주고 눈 눈썹 입술을 표현합니다.') 

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh = mp.solutions.face_mesh      
            drawingspec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)    

            self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)
            if not self.cap.isOpened():sys.exit('카메라 연결 실패')
            
            with mp_face_mesh.FaceMesh(
                  max_num_faces=1, refine_landmarks=True,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as face_mesh:

                  while self.cap.isOpened():
                        ret,frame = self.cap.read()
                        frame.flags.writeable = False
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        results = face_mesh.process(frame)

                        frame.flags.writeable = True
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

                        if results.multi_face_landmarks:
                              for face_landmarks in results.multi_face_landmarks:
                                    mp_drawing.draw_landmarks(
                                          image=frame, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_TESSELATION, 
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles
                                          .get_default_face_mesh_tesselation_style())
                
                                    mp_drawing.draw_landmarks(
                                          image=frame, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles
                                          .get_default_face_mesh_contours_style())
                        
                                    mp_drawing.draw_landmarks(
                                          image=frame, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_IRISES,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles
                                          .get_default_face_mesh_iris_connections_style())

                        cv.imshow('Mesh', cv.flip(frame, 1))

                        key = cv.waitKey(1)
                        if key == ord('q'):
                              self.cap.release()
                              cv.destroyWindow('Mesh')
                              break


      def handfunction(self):

            self.label.setText('손을 찾아줍니다.') 

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_hands = mp.solutions.hands 

            self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)
            if not self.cap.isOpened():sys.exit('카메라 연결 실패')

            with mp_hands.Hands(
                  model_complexity=0,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as hands:

                  while self.cap.isOpened():
                        ret, frame = self.cap.read()
                        frame.flags.writeable = False
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        results = hands.process(frame)

                        frame.flags.writeable = True
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                        if results.multi_hand_landmarks:
                              for hand_landmarks in results.multi_hand_landmarks:
                                    mp_drawing.draw_landmarks(
                                          frame,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                        cv.imshow('Hands', cv.flip(frame, 1))

                        key = cv.waitKey(1)
                        if key == ord('q'):
                              self.cap.release()
                              cv.destroyWindow('Hands')
                              break

      def posefunction(self):

            self.label.setText('몸의 움직임을 보여줍니다') 

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_pose = mp.solutions.pose
            
            self.cap=cv.VideoCapture(0,cv.CAP_DSHOW)
            if not self.cap.isOpened():sys.exit('카메라 연결 실패')

            with mp_pose.Pose(
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

                  while self.cap.isOpened():      
                        ret,frame = self.cap.read()

                        frame.flags.writeable = False
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        results = pose.process(frame)

                        frame.flags.writeable = True
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

                        mp_drawing.draw_landmarks(
                              frame,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                        cv.imshow('Pose', cv.flip(frame, 1))
                        
                        key = cv.waitKey(1)
                        if key == ord('q'):
                              self.cap.release()
                              cv.destroyWindow('Pose')
                              break


      def quitFunction(self):
            self.cap.release()
            cv.destroyAllWindows()
            self.close()


app = QApplication(sys.argv)
win = PProject()
win.show()
app.exec_()