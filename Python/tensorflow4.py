import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x=iris.data #Datos
y=iris.target #Etiquetas

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train, epochs=100)
y_pred=(model.predict(np.array([[5.9,3.0,5.1,1.8]])))
salida=np.argmax(y_pred,1)
print(salida)
print(iris.target_names[salida])