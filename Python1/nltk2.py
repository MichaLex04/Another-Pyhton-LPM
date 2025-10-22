#importamos librerias
import spacy
#Importamos el modelo de lenguaje en español
nlp = spacy.load("es_core_news_sm")
texto="Hola ¿Cómo estás? hoy aprenderemos Procesamiento de Lenguaje Natural"
#Convertimos el texto a minúsculas
doc=nlp(texto.lower())
print("Texto procesado con Spacy")
print(doc)
#Iteramos sobre los tokens y mostramos su parte del habla
for token in doc:
    print(token, token.lemma_)