import pandas as pd

empleados = pd.read_csv("empleados.csv")
Meses = 10
empleados = pd.DataFrame({
    "Departamento": empleados["Departamento"],
    "Nombre": empleados["Nombre"],
    "Meses": Meses,
    "Salario": empleados["Salario"],
    "Total": empleados["Salario"] * Meses
})
Reporte = empleados.groupby("Departamento")["Total"].sum()
print(Reporte)