import pandas as pd

nombres=pd.Series(["Aria","Kayla","Aira","Yue"])
print(nombres)

notas=pd.Series([15,20,5])
print(notas)

boleta=pd.DataFrame({
    "nombres":nombres,
    "notas":notas
})
print(boleta.nombres)

boleta.cursos=pd.Series(["PHP","JAVA","PYTHON"])