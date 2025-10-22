#Implementar un data frame con 4 columnas (nombres, notas1-3) 
#y en una nueva columna calcular el promedio
import pandas as pd

Nombres=pd.Series(["Aria","Kayla","Yue"])
Teoria=pd.Series([15,20,10])
Desarrollo=pd.Series([15,17,20])
Practica=pd.Series([15,14,20])
Promedio=(Teoria+Desarrollo+Practica)/3

Informe=pd.DataFrame({
    "Nombres":Nombres,
    "Teoria":Teoria,
    "Desarrollo":Desarrollo,
    "Practica":Practica,
    "Promedio":Promedio
})
print(Informe)


#Informe=pd.DataFrame({"Nombres:":["Aria","Kayla","Yue"], "Teoria:":[15,20,10], "Desarrollo:":[15,20,10], "Practica:":[15,20,10]});Informe["Promedio:"]=(Informe["Teoria:"]+Informe["Desarrollo:"]+Informe["Practica:"])/3
#print(Informe)