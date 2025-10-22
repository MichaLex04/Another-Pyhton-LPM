#importamos librerias
import spacy
#Importamos el modelo de lenguaje en español
nlp = spacy.load("es_core_news_sm")
#En este caso no se convierte a minúsculas ya que facilita el análisis de entidades nombradas
texto="El diretor de SENATI publicó que se apertura una oficina nueva en Puno"
doc=nlp(texto)
for token in doc.ents:
    print(token)