import pandas as pd
import matplotlib.pyplot as plt

escuela=pd.DataFrame({
    "idescuela":[1,2],
    "nombreescuela":["ETI","ENI"]
})

carrera=pd.DataFrame({
    "idcarrera":[1,2,3,4],
    "nombrecarrera":["Software","Redes","Administracion","Contabilidad"],
    "idescuela":[1,1,2,2]
})

print("--ESCUELAS--")
print(escuela)
print("--CARRERAS--")
print(carrera)
reporte1=pd.merge(carrera,escuela, on="idescuela", how="inner")
print(reporte1)
conteo=reporte1.groupby("nombreescuela")["idescuela"].count()
print(conteo)
"""
plt.bar(conteo.index, conteo.values)
plt.show()
"""
plt.pie(conteo, labels=conteo.index, autopct="%1.1f%%")
plt.show()