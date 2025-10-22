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

reporte1=pd.merge(productos, categorias, on="id_categoria", how="inner")
print(reporte1)
reporte2=reporte1.query("stock<100")
print(reporte2)