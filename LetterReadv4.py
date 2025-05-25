import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom



# Ladda EMNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

print("Train samples:", ds_info.splits['train'].num_examples)
print("Test samples:", ds_info.splits['test'].num_examples)

# Normalisera bilder till [0,1]
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128  # större batch är OK med GPU

ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
# CNN-modell som fungerar väldigt bra på EMNIST
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(27, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test,
    callbacks=[early_stopping]
)

# Utvärdera modellen
loss, acc = model.evaluate(ds_test)
print(f"\n✅ Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

# Visa 25 predictions
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

        plt.title(f"P: {pred_char}\nT: {true_char}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()