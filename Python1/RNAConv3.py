import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    "entrenamiento/train", image_size=(48,48), color_mode="grayscale", batch_size=32)
test_ds=tf.keras.preprocessing.image_dataset_from_directory(
    "entrenamiento/test", image_size=(48,48), color_mode="grayscale", batch_size=32)
val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    "entrenamiento/validation", image_size=(48,48), color_mode="grayscale", batch_size=32)

class_names= train_ds.class_names

train_ds=train_ds.map(lambda x,y: (x/255.0, y))
test_ds=test_ds.map(lambda x,y: (x/255.0, y))
val_ds=val_ds.map(lambda x,y: (x/255.0, y))

model= tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48,48,1)),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

print(model.summary())
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history= model.fit(train_ds, epochs=15, validation_data= val_ds)

plt.plot(history.history["accuracy"], label= "Entrenamiento")
plt.plot(history.history["val_accuracy"], label= "Validación")
plt.title("Precisión del Modelo")
plt.grid()
plt.legend()
plt.show()

for images, labels in test_ds.take(1):
    plt.figure(figsize=(16,24))
    for i in range(18):
        plt.subplot(3,6,i+1)
        plt.imshow(images[i].numpy(),cmap="gray")
        plt.title(f"Predict:{class_names[labels[i].numpy()]}")
        plt.axis("off")
plt.show()