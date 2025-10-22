import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test)=fashion_mnist.load_data()#dividimos el dataset
print(x_train.shape)#forma
print(x_train[60000])#mostrar el tensor de la primera imagen
plt.imshow(x_train[60000])#mostramos las escalas de valores numéricos
plt.bar()#mostramos la escala con 
plt.show()#mostramos el gráfico 