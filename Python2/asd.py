from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
#OPCIONAL
import matplotlib.pyplot as plt

mnist=fetch_openml("mnist_784", version=1, as_frame=False)

x=mnist.data
y=mnist.target
x=x/255.0

print(x[0])
print(y[0])
plt.imshow(x[0].reshape(28,28))
plt.show()