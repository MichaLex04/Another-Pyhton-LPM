import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
x=iris.data
y=iris.target

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=42)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation="relu",input_shape=(4,)),
    tf.keras.layers.Dense(8,activation="relu"),
    tf.keras.layers.Dense(3,activation="softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")
model.fit(xtrain,ytrain,epochs=100)
y_pred=model.predict(xtest[0].reshape(1,-1))
salida=np.argmax(y_pred,1)
print(salida)
print(iris.target_names[salida])

