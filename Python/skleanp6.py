import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
x=iris.data
model= PCA(n_components=2)
iris_reduced = model.fit_transform(x)
iris_reduced[0:5]