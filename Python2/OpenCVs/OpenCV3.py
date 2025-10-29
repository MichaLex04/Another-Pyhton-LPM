import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model=load_model("OpenCVs/FacesConv.h5")
img=cv2.imread("Entrenamiento/train/happy/103.jpg", cv2.IMREAD_GRAYSCALE)
img_resized= cv2.resize(img, (48,48))
img_input= img_resized.reshape(1,48,48,1)
predict=model.predict(img_input)

print(predict)
plt.imshow(img)
plt.show()