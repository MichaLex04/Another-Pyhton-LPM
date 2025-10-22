from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
x=iris.data
y=iris.target

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

model=MLPClassifier(hidden_layer_sizes=(20,15), activation="relu", solver="adam", max_iter=20, random_state=42)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print("etiquetas reales", y_test)
print("predicciones", y_pred)

muestra= x_test[0].reshape(1,-1)
muestra.shape
pred_muestra= model.predict(muestra)
print("Etiqueta Real", y_test[0])
print("Prediccion Individual", pred_muestra)