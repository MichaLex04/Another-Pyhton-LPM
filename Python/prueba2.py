ventas = [100,500,1200,2000,2500]

for v in ventas:
    print(v)
print("ventas")
for i in range(0,5,1):
    print(ventas[i])
    
print("ventas")
print([v for v in ventas if v>1000])