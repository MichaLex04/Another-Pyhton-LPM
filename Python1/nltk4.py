#Importamos librerias
import spacy
#Importamos el modelo de lenguaje en español
nlp = spacy.load("es_core_news_sm")
texto="Hola ¿Cómo estás? hoy aprenderemos Procesamiento de Lenguaje Natural"
doc=nlp(texto.lower())
print("Texto procsado con Spacy")
print(doc)
for token in doc:
    print(token, token.pos)