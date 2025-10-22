#Importamos librerias
import nltk1
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
nltk1.download('punkt')
nltk1.download('stopwords')

#Que es la tokenizacion y conversion a minusculas
texto="Hola ¿Cómo estás? hoy aprenderemos Procesamiento de Lenguaje Natural"
tokens=word_tokenize(text=texto.lower())
print("TOKINZADO CON word_tokenize")
print(tokens)

#Tokenizacion con TweetTokenizer
tokenizer=TweetTokenizer()
tokens=tokenizer.tokenize(texto.lower())
print("TOKINZADO CON TweetTokenizer")
print(tokens)

#Que son las stopwords
print("Trabajando con stopwords")
stop_words=stopwords.words('spanish')
#print(stop_words)

#Limpieza de stopwords
print("Limpieza de stopwords")
texto_impio=[t for t in tokens if t .isalpha() and t not in stop_words]
print(texto_impio)

