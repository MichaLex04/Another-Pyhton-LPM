import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(x_train[0])
plt.imshow(x_train[0])
plt.bar()
plt.show()