import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Ladda EMNIST-bokstäver datasetet
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],  #Dela upp i träning och test
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

#Visa information om datasetet. 
#Train samples: 88800
#Test samples: 14800
print("Train samples:", ds_info.splits['train'].num_examples)
print("Test samples:", ds_info.splits['test'].num_examples)

# Normalisera bilder (från 0–255 till 0–1)
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(normalize_img).batch(128).prefetch(1)
ds_test = ds_test.map(normalize_img).batch(128).prefetch(1)


# Första försöket av modellen 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # Gör om 2D-bilder till 1D vektor
    tf.keras.layers.Dense(256, activation='relu'),     # Första dolda lagret med 256 noder och ReLU-aktivering
    tf.keras.layers.Dropout(0.3),                      # Slumpmässigt stänger av 30% av noderna för att undvika överträning
    tf.keras.layers.Dense(27, activation='softmax')    # Utgångslager med 27 klasser (A–Z + okänd klass 0)
])

# Kompilera modellen
# Adam är en optimeringsalgoritm
# Sparse_categorical_crossentropy används för klassificering när etiketterna är siffror 
# Vi mäter träffsäkerhet (accuracy)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Träna modellen på träningsdata i 2 omgångar (epochs)
# Validera modellen med testdata under träningen
model.fit(ds_train, epochs=2, validation_data=ds_test)

# Hämta en batch från test-datasetet
for images, labels in ds_test.take(1):
    # Gör predictioner
    predictions = model.predict(images)

    # Plotta 25 bilder (5x5)
    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(tf.squeeze(images[i]), cmap="gray")

        pred_label = np.argmax(predictions[i])  # index från 0–26
        true_label = labels[i].numpy()

        # Konvertera till bokstäver: 1 → 'A', 2 → 'B', osv.
        pred_char = chr(pred_label + 64)  # 0 -> A, 1 -> B, ...
        true_char = chr(true_label + 64)  # 1 -> A, 2 -> B, ...

        plt.title(f"Prediction: {pred_char} / \nTrue: {true_char}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()