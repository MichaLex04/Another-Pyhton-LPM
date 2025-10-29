import pandas as pd

Edades=pd.Series([25,30,35,40,10,15,17])
Filtro_18=Edades[Edades>18]
print(Filtro_18)

nombres=pd.Series(["A","B","C","D","E","F","G"])
frame=pd.DataFrame({
    "Edades":Edades,
    "Nombres":nombres})
print(frame)

Filtro_18_2=frame[frame["Edades"]>18].Nombres
print(Filtro_18_2)  