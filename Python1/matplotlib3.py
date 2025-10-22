import pandas as pd
import matplotlib.pyplot as plt

categorias=pd.DataFrame({
    "idcategoria":[1,2,3],
    "nombrecategoria":["Gaseosas","Leches","Jugos"]
})

productos=pd.DataFrame({
    "idproducto":[1,2,3,4,5,6],
    "nombreproducto":["KR-2L","BLACK 3L","ESCOSESA 2L","GLORIA 360ML", "FRUGOS DEL VALLE 500ML","GLORIA 500ML"],
    "monto":[5.5 ,6.5 ,5.5 ,4.5 ,4.5 ,5.5],
    "stock":[100, 100 ,100 ,100 ,50 ,50],
    "idcategoria":[1,1,1,2,3,3]
})

reporte=pd.merge(productos,categorias,on="idcategoria",how="inner")
print (reporte)
conteo=reporte.groupby("nombrecategoria")["stock"].sum()
print(conteo)
"""
plt.pie(conteo, labels=conteo.index, autopct="%1.1f%%")
plt.show()
"""
plt.bar(conteo.index, conteo.values)
plt.show()