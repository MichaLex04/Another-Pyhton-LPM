import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "entrenamiento/train", image_size=(48,48), color_mode="grayscale", batch_size=32)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "entrenamiento/test", image_size=(48,48), color_mode="grayscale", batch_size=32)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "entrenamiento/validation", image_size=(48,48), color_mode="grayscale", batch_size=32)

class_names = train_ds.class_names

# Normalización - forma más explícita
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = train_ds.map(normalize_img)
test_ds = test_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)

# Modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48,48,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

print(model.summary())
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Entrenamiento
history = model.fit(train_ds, epochs=10, validation_data=val_ds)

# Gráfica de precisión
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión del Modelo")
plt.grid()
plt.legend()
plt.show()

# **CÓDIGO CORREGIDO PARA PREDICCIONES**
for images, labels in test_ds.take(1):
    # Hacer predicciones
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = labels.numpy()
    
    plt.figure(figsize=(12,16))
    for i in range(min(18, len(images))):
        plt.subplot(3,6,i+1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")  # .squeeze() para quitar dimensión extra
        
        # Verificar si la predicción es correcta
        is_correct = predicted_classes[i] == true_classes[i]
        color = 'green' if is_correct else 'red'
        
        plt.title(f"Real: {class_names[true_classes[i]]}\nPred: {class_names[predicted_classes[i]]}", 
                 color=color, fontsize=8)
        plt.axis("off")
    
    # Calcular precisión en este batch
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Precisión en este batch: {accuracy:.2%}")
    
plt.tight_layout()
plt.show()

# Evaluación final del modelo
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Precisión final en test: {test_accuracy:.2%}")