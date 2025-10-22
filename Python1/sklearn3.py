import pandas as pd
import numbers as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame({
    "tamano":[50,80,60,120,100],
    "habitaciones":[2, 3, 2, 4, 3],
    "zona": ["Norte", "Sur", "Este", "Norte", "Oeste"],
    "antiguedad": [10, 5, 20, 2, 15],
    "precio": [120000, 200000, 150000, 300000, 250000]
}) 

x=data[["tamano","habitaciones","zona","antiguedad"]]
y=data["precio"]

scaler=StandardScaler()
x_num=scaler.fit_transform(data[["tamano","habitaciones","antiguedad"]])
encoder=OneHotEncoder(drop="first", sparse_output=False)
x_cat=encoder.fit_transform(data[["zona"]])
print(x_num)
print(x_cat)

x_final= np.hstack([x_num, x_cat])
print(x_final)

model=Lasso(alpha=0.1)
model.fit(x_final, y)

y_pred=model.predict(x_final)
print("Predicciones", y_pred)
print("Coeficientes", model.coef_)
print("Intercepto", model.intercept_)
print("R2_SCORE", r2_score(y, y_pred))
print("MSE", mean_squared_error(y, y_pred))

#Iris, Wine, Mnist, House California, Casa Zonas(este)