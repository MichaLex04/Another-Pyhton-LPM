from turtle import right
import cv2
from matplotlib.dates import MO
import mediapipe as mp
mp_face_mesh= mp.solutions.face_mesh
face_mesh= mp_face_mesh.FaceMesh(max_num_faces=1)
EYE_LEFT=[33,133]
EYE_RIGHT=[362,263]
MOUTH=[76,73,84,314,303,292]
cap= cv2.VideoCapture(0)
while True:
    ret, frame= cap.read()
    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results= face_mesh.process(frame_rgb)
    for face_landmarks in results.multi_face_landmarks:
        h,w,_= frame.shape

        #Ojo izquierdo

        left_inner= (int(face_landmarks.landmark[EYE_LEFT[0]].x*w), 
                                     int(face_landmarks.landmark[EYE_LEFT[0]].y*h))
        left_outer= (int(face_landmarks.landmark[EYE_LEFT[1]].x*w), 
                                    int(face_landmarks.landmark[EYE_LEFT[1]].y*h))
        
        cv2.circle(frame, left_inner, 3, (0,0,255), -1)
        cv2.circle(frame, left_outer, 3, (0,0,255), -1)

        #Ojo derecho

        right_inner= (int(face_landmarks.landmark[EYE_RIGHT[0]].x*w), 
                                     int(face_landmarks.landmark[EYE_RIGHT[0]].y*h))
        right_outer= (int(face_landmarks.landmark[EYE_RIGHT[1]].x*w), 
                                    int(face_landmarks.landmark[EYE_RIGHT[1]].y*h))
        
        cv2.circle(frame, right_inner, 3, (0,0,255), -1)
        cv2.circle(frame, right_outer, 3, (0,0,255), -1)

        #Boca

        mouth_left= (int(face_landmarks.landmark[MOUTH[0]].x*w), 
                                     int(face_landmarks.landmark[MOUTH[0]].y*h))
        mouth_center_left= (int(face_landmarks.landmark[MOUTH[1]].x*w), 
                                    int(face_landmarks.landmark[MOUTH[1]].y*h))
        mouth_center2_left= (int(face_landmarks.landmark[MOUTH[2]].x*w), 
                                     int(face_landmarks.landmark[MOUTH[2]].y*h))
        mouth_center2_right= (int(face_landmarks.landmark[MOUTH[3]].x*w), 
                                     int(face_landmarks.landmark[MOUTH[3]].y*h))
        mouth_center_right= (int(face_landmarks.landmark[MOUTH[4]].x*w), 
                                    int(face_landmarks.landmark[MOUTH[4]].y*h))
        mouth_right= (int(face_landmarks.landmark[MOUTH[5]].x*w), 
                                     int(face_landmarks.landmark[MOUTH[5]].y*h))
        
        cv2.circle(frame, mouth_left, 3, (0,0,255), -1)
        cv2.circle(frame, mouth_center_left, 3, (0,0,255), -1)
        cv2.circle(frame, mouth_center2_left, 3, (0,0,255), -1)
        cv2.circle(frame, mouth_center2_right, 3, (0,0,255), -1)
        cv2.circle(frame, mouth_center_right, 3, (0,0,255), -1)
        cv2.circle(frame, mouth_right, 3, (0,0,255), -1)
        
    cv2.imshow("Resultados: ", frame)
    tecla= cv2.waitKey(1)
    if tecla==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()