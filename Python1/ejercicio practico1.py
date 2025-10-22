    import pandas as pd

empleados=pd.read_csv("Empleados3.csv")

ReporteEPD = empleados.groupby("Departamentos")["Empleados"].count()

ReporteESH=pd.DataFrame({"Hijos": empleados["Hijos"]})
ReporteESH["Tiene Hijos"] = ["Si" if x >= 1 else "No" for x in ReporteESH["Hijos"]]

Reporte=pd.DataFrame({
    "Departamentos": empleados["Departamentos"],
    "Empleados":empleados["Empleados"],
    "Dias":empleados["Dias"],
    "Diario":empleados["Diario"],
    "Sueldo_Bruto": empleados["Dias"] * empleados["Diario"],
    "Hijos": empleados["Hijos"],
    "Bonificacion_Hijos": empleados["Hijos"] * 100,
    "Sueldo_Total": (empleados["Dias"] * empleados["Diario"]) + (empleados["Hijos"] * 100)
})

print(ReporteEPD)
print(ReporteESH)
print(Reporte)