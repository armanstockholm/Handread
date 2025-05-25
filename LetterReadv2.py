import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom
from tensorflow.keras.models import Sequential

# Ladda EMNIST-bokstäver datasetet
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],  # Dela upp i träning och test
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Skriv ut hur många exempel som finns
print("Train samples:", ds_info.splits['train'].num_examples)
print("Test samples:", ds_info.splits['test'].num_examples)

# Normalisera bilderna (gör pixelvärden mellan 0–1)
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(normalize_img).batch(128).prefetch(1)
ds_test = ds_test.map(normalize_img).batch(128).prefetch(1)

# Data Augmentation (gör bilden lite olika varje gång för att förbättra generalisering)
data_augmentation = Sequential([
    RandomRotation(0.1),
    RandomTranslation(0.1, 0.1),
    RandomZoom(0.1)
])

# Skapa modellen – CNN + Data Augmentation + Dropout
model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(27, activation='softmax')  # 27 klasser: A-Z + extra klass
])

# Kompilera modellen
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Lägg till early stopping – avbryt träning om den inte förbättras
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Träna modellen
model.fit(
    ds_train,
    epochs=20,
    validation_data=ds_test,
    callbacks=[early_stopping]
)

# Utvärdera modellen på testdata
loss, accuracy = model.evaluate(ds_test)
print(f"\n✅ Modellens träffsäkerhet på testdata: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Visa 25 förutsägelser från testdatan
for images, labels in ds_test.take(1):
    predictions = model.predict(images)

    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(tf.squeeze(images[i]), cmap="gray")

        pred_label = np.argmax(predictions[i])
        true_label = labels[i].numpy()

        pred_char = chr(pred_label + 64)
        true_char = chr(true_label + 64)

        plt.title(f"Prediction: {pred_char}\nTrue: {true_char}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()