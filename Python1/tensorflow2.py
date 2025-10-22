import tensorflow as tf
import numpy as np

#crea un array desde -10 hasta 10 con 200 elementos entre ellos (decimales)
#un array que devuelve 1 si el numero es menor que 0 (negativo) y 1 si es positivo
x=np.linspace(-10,10,200).reshape(-1,1)
y=np.array([1 if i > 0 else 0 for i in x])

#activa la secuencia de forma sigmoide, lo que da resutados de 0 y 1 (binario)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="sigmoid", input_shape=(1,)),
])

#
tf.keras.utils.set_random_seed(1)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(x,y,epochs=2, vorbose=0)

#
y_pred=model.predict(np.array([[-3],[-2],[-20],[50]]))
print(y_pred.round())


