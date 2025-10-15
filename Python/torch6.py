import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer

datos= pd.DataFrame({
    "leng":["Java","PHP","Java","PHP","Java",],
    "cert":["NO","NO","SI","SI","NO",],
    "rang":["J","J","S","S","S"],
    "contrato":[0,0,1,0,0]
})