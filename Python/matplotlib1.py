import matplotlib.pyplot as plt

"""
#Grafico de barras
etiquetas=["A","B","C"]
valores=[100,200,300]

plt .bar(etiquetas,valores, color=["r" , "y" , "g"])
plt. title("Gráfico de Barras")
plt. xlabel("Etiquetas")
plt. ylabel("Valores" )
plt. grid(True)
plt. show()
"""

#Histograma
notas=[5,5,6,7,8,8,9,9,9,10,15,15,20]
plt.hist(notas, bins=5, edgecolor="black", color=["r"])
plt.show()

"""
x=[1,2,3,4,5]
y=[2,4,6,8,10]
plt.scatter(x, y)
plt.show( )


x=[1,2,3,4,5,6,6,6,7,8,9,10]
plt.plot(x, marker="o")
plt.show()


labels=["A","B","C"]
valores=[100,100,100]
plt.pie(valores, labels=labels, autopct="%1.1f%%")
plt.show()
"""