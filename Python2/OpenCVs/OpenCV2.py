import cv2

cap=cv2.VideoCapture("OpenCVs\OpenCVids\correr1.mp4")
while True:
    ret, frame=cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(720,480)) 
    #Lineas Horizontales
    cv2.line(frame, (90, 70), (140, 70), (255,0,0), 4)
    cv2.line(frame, (560, 70), (610, 70), (255,0,0), 4)
    cv2.line(frame, (90, 400), (140, 400), (255,0,0), 4)
    cv2.line(frame, (560, 400), (610, 400), (255,0,0), 4)
    #Lineas Verticales
    cv2.line(frame, (230, 70), (230, 200), (0,255,0), 4)
    cv2.line(frame, (230, 270), (230, 400), (0,255,0), 4)
    cv2.line(frame, (470, 70), (470, 200), (0,255,0), 4)
    cv2.line(frame, (470, 270), (470, 400), (0,255,0), 4)
    #Lineas Diagonales
    cv2.line(frame, (40, 140), (250, 20), (0,0,255), 4)
    cv2.line(frame, (40, 340), (250, 450), (0,0,255), 4)
    cv2.line(frame, (450, 20), (670, 140), (0,0,255), 4)
    cv2.line(frame, (450, 450), (670, 340), (0,0,255), 4)
    #Rectangulos
    cv2.rectangle(frame, (40,20), (670,450), (0,0,0), 4) #Rectangulo Mayor
    cv2.rectangle(frame, (70,50), (250,220), (255,0,255), 4)
    cv2.rectangle(frame, (70,250), (250,420), (255,0,255), 4)
    cv2.rectangle(frame, (450,50), (630,220), (225,0,255), 4)
    cv2.rectangle(frame, (450,250), (630,420), (255,0,255), 4)
    cv2.imshow("Resultados", frame)
    tecla=cv2.waitKey(20)
    if tecla==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()