from PyQt5.QtWidgets import *  # PyQt5 라이브러리에서 모든 위젯을 가져옴
import cv2 as cv  # OpenCV 라이브러리를 cv로 가져옴
from mediapipe.python.solutions import face_detection  # Mediapipe에서 face_detection 솔루션을 가져옴
import numpy as np  # NumPy 라이브러리를 np로 가져옴
import mediapipe as mp  # Mediapipe 라이브러리를 mp로 가져옴
import sys  # 시스템 모듈을 가져옴

# 메인 윈도우 클래스를 정의 (QMainWindow를 상속)
class PProject(QMainWindow):

      def __init__(self):
            super().__init__()

            # 윈도우 창의 제목과 크기 설정
            self.setWindowTitle('프로젝트')
            self.setGeometry(200, 200, 900, 800)

            # 버튼 및 라벨 생성
            Button1 = QPushButton('사용법', self)
            faceButton1 = QPushButton('얼굴 인식1', self)
            faceButton2 = QPushButton('얼굴 인식2', self)
            handbutton = QPushButton('손', self)
            posebutton = QPushButton('포즈', self)
            quitButton = QPushButton('나가기', self)
            self.label = QLabel('안녕하세요', self)

            # 버튼 위치 및 크기 설정
            Button1.setGeometry(10, 25, 100, 30)
            faceButton1.setGeometry(110, 25, 100, 30)
            faceButton2.setGeometry(210, 25, 100, 30)
            handbutton.setGeometry(310, 25, 100, 30)
            posebutton.setGeometry(410, 25, 100, 30)
            quitButton.setGeometry(650, 25, 100, 30)
            self.label.setGeometry(10, 70, 700, 400)

            # 버튼 클릭 시 실행할 함수 연결
            Button1.clicked.connect(self.Button1F)
            faceButton1.clicked.connect(self.facefunction1)
            faceButton2.clicked.connect(self.facefunction2)
            handbutton.clicked.connect(self.handfunction)
            posebutton.clicked.connect(self.posefunction)
            quitButton.clicked.connect(self.quitFunction)

      # 사용법 설명을 위한 함수
      def Button1F(self): 
            self.label.
            setText(' 버튼을 누르면 각 기능이 실행됩니다. 각 창은 q를 누르면 종료됩니다.\n\n 얼굴인식 1: 얼굴의 눈 코 입 귀 얼굴 부분을 보여줍니다.\n\n 얼굴인식 2: 얼굴을 mesh로 보여주고 눈 눈썹 입술을 표현합니다.\n\n 손 : 손의 각 관절 부분을 보여줍니다.\n\n 포즈 : 사람이 취하고 있는 포즈가 나타납니다')

      # Mediapipe를 사용해 얼굴 인식 및 얼굴 주요 부위 표시
      def facefunction1(self): 
            self.label.setText('얼굴의 눈 코 입 귀 얼굴 부분을 보여줍니다.') 

            mp_face_detection = mp.solutions.face_detection  # 얼굴 인식 솔루션
            mp_drawing = mp.solutions.drawing_utils  # 얼굴 탐지를 그리기 위한 유틸리티

            self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라 초기화
            if not self.cap.isOpened(): sys.exit('카메라 연결 실패')  # 카메라 연결 실패 시 종료

            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                  while self.cap.isOpened():
                        ret, frame = self.cap.read()  # 카메라에서 프레임 읽기

                        # 프레임을 RGB로 변환 후 처리
                        frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
                        frame.flags.writeable = False  # 프레임을 불변으로 설정
                        results = face_detection.process(frame)  # 얼굴 탐지 처리

                        frame.flags.writeable = True  # 프레임을 다시 쓰기 가능으로 변경
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # 다시 BGR로 변환
                        if results.detections:  # 얼굴이 탐지되면 얼굴에 사각형 그리기
                              for detection in results.detections:
                                    mp_drawing.draw_detection(frame, detection)

                        # 결과를 보여줌
                        cv.imshow('Detection', frame)
                        key = cv.waitKey(1)

                        if key == ord('q'):  # 'q'를 누르면 창 닫기
                              self.cap.release()
                              cv.destroyWindow('Detection')
                              break

      # Mediapipe를 사용해 얼굴 메쉬를 표시
      def facefunction2(self):
            self.label.setText('얼굴을 mesh로 보여주고 눈 눈썹 입술을 표현합니다.') 

            mp_drawing = mp.solutions.drawing_utils  # 그리기 유틸리티
            mp_drawing_styles = mp.solutions.drawing_styles  # 스타일 유틸리티
            mp_face_mesh = mp.solutions.face_mesh  # 얼굴 메쉬 솔루션
            drawingspec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)  # 그리기 옵션

            self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라 초기화
            if not self.cap.isOpened(): sys.exit('카메라 연결 실패')  # 카메라 연결 실패 시 종료

            with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                  while self.cap.isOpened():
                        ret, frame = self.cap.read()  # 카메라에서 프레임 읽기
                        frame.flags.writeable = False  # 프레임을 불변으로 설정
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # BGR을 RGB로 변환
                        results = face_mesh.process(frame)  # 얼굴 메쉬 처리

                        frame.flags.writeable = True  # 다시 쓰기 가능으로 변경
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # 다시 BGR로 변환

                        if results.multi_face_landmarks:  # 얼굴 랜드마크가 있으면 그리기
                              for face_landmarks in results.multi_face_landmarks:
                                    # 얼굴 메쉬, 윤곽선, 홍채 그리기
                                    mp_drawing.draw_landmarks(
                                          image=frame, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                                    mp_drawing.draw_landmarks(
                                          image=frame, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                                    mp_drawing.draw_landmarks(
                                          image=frame, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_IRISES,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                        # 결과를 보여줌
                        cv.imshow('Mesh', cv.flip(frame, 1))

                        key = cv.waitKey(1)
                        if key == ord('q'):  # 'q'를 누르면 창 닫기
                              self.cap.release()
                              cv.destroyWindow('Mesh')
                              break

      # Mediapipe를 사용해 손을 탐지하고 표시
      def handfunction(self):
            self.label.setText('손을 찾아줍니다.') 

            mp_drawing = mp.solutions.drawing_utils  # 그리기 유틸리티
            mp_drawing_styles = mp.solutions.drawing_styles  # 스타일 유틸리티
            mp_hands = mp.solutions.hands  # 손 탐지 솔루션

            self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라 초기화
            if not self.cap.isOpened(): sys.exit('카메라 연결 실패')  # 카메라 연결 실패 시 종료

            with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                  while self.cap.isOpened():
                        ret, frame = self.cap.read()  # 카메라에서 프레임 읽기
                        frame.flags.writeable = False  # 프레임을 불변으로 설정
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # BGR을 RGB로 변환
                        results = hands.process(frame)  # 손 탐지 처리

                        frame.flags.writeable = True  # 다시 쓰기 가능으로 설정
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # 다시 BGR로 변환
                        if results.multi_hand_landmarks:  # 손 랜드마크가 있으면 그리기
                              for hand_landmarks in results.multi_hand_landmarks:
                                    mp_drawing.draw_landmarks(
                                          frame,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                        # 결과를 보여줌
                        cv.imshow('Hands', cv.flip(frame, 1))

                        key = cv.waitKey(1)
                        if key == ord('q'):  # 'q'를 누르면 창 닫기
                              self.cap.release()
                              cv.destroyWindow('Hands')
                              break

      # Mediapipe를 사용해 몸의 포즈를 탐지하고 표시
      def posefunction(self):
            self.label.setText('몸의 움직임을 보여줍니다') 

            mp_drawing = mp.solutions.drawing_utils  # 그리기 유틸리티
            mp_drawing_styles = mp.solutions.drawing_styles  # 스타일 유틸리티
            mp_pose = mp.solutions.pose  # 포즈 탐지 솔루션

            self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라 초기화
            if not self.cap.isOpened(): sys.exit('카메라 연결 실패')  # 카메라 연결 실패 시 종료

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                  while self.cap.isOpened():
                        ret, frame = self.cap.read()  # 카메라에서 프레임 읽기

                        frame.flags.writeable = False  # 프레임을 불변으로 설정
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # BGR을 RGB로 변환
                        results = pose.process(frame)  # 포즈 탐지 처리

                        frame.flags.writeable = True  # 다시 쓰기 가능으로 변경
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # 다시 BGR로 변환

                        mp_drawing.draw_landmarks(
                              frame,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                        # 결과를 보여줌
                        cv.imshow('Pose', cv.flip(frame, 1))

                        key = cv.waitKey(1)
                        if key == ord('q'):  # 'q'를 누르면 창 닫기
                              self.cap.release()
                              cv.destroyWindow('Pose')
                              break

      # '나가기' 버튼을 눌렀을 때 창을 닫고 카메라 자원 해제
      def quitFunction(self):
            self.cap.release()
            cv.destroyAllWindows()  # 모든 OpenCV 창 닫기
            self.close()  # 어플리케이션 종료

# PyQt5 어플리케이션 실행
app = QApplication(sys.argv)
win = PProject()
win.show()  # 메인 윈도우 표시
app.exec_()  # 이벤트 루프 실행
