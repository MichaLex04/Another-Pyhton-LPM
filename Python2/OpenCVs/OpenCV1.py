import cv2

cap=cv2.VideoCapture("OpenCVs\OpenCVids\correr1.mp4")
x1=100 #coordenada de punto horizontal
y1=100 #coordenada de punto vertical
while True:
    ret, frame=cap.read()
    if not ret:
        break
    #ancho x alto : frame.shape[1] x frame.shape[0]
    frame=cv2.resize(frame,(720,480)) 
    #ALTERNATIVAS PARA DIBUJAR LINEAS VERTICALES
    #cv2.line(frame, (x1,0), (x1, frame.shape[0]), (0,255,0), 2)
    cv2.line(frame, (100, 0), (100, 480), (0,255,0), 2)
    #ALTERNATIVAS PARA DIBUJAR LINEAS HORIZONTALES
    #cv2.line(frame, (0,y1), (frame.shape[1], y1,) (0,0,255), 2)
    cv2.line(frame, (0, 100), (720, 100), (0,0,255), 2)
    cv2.imshow("Resultados", frame)
    tecla=cv2.waitKey(10)
    if tecla==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()