from sklearn.linear_model import LinearRegression
import numpy as np

x=np.array([[1], [2], [3], [4], [5]])
y=np.array([10, 20, 30, 40, 50])
model=LinearRegression()
model.fit(x,y)
model.predict([[20]])