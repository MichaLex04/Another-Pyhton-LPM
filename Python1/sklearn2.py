from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

mnist= fetch_openml('mnist_784', version=1, as_frame=False)
x= mnist.data
y= mnist.target
x=x/255.0

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

model=MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", solver="adam", max_iter=20, random_state=42)

model.fit(x_train, y_train)
y_pred=model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

entrada=256
muestra= x_test[entrada].reshape(1,-1)
muestra.shape
pred_muestra= model.predict(muestra)
print("Etiqueta Real", y_test[0])
print("Prediccion Individual", pred_muestra[0])

plt.imshow(x_test[0].reshape(28,28))
plt.title(f"Etiqueta Real: {y_test[0]} - Prediccion: {pred_muestra[0]}" )
plt.show()