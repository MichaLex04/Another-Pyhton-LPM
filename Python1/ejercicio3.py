import pandas as pd

Alumnos=pd.Series(["Aria","Kayla","Aria","Yue"])
Puntos=pd.Series([1,2,3,2,1])
Intervenciones=pd.DataFrame({"Alumnos":Alumnos,"Puntos":Puntos})
#print(Intervenciones)

Reporte = Intervenciones.groupby("Alumnos")["Puntos"].sum()
print(Reporte)