import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

(train_ds, val_ds), ds_info = tfds.load('tf_flowers',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True, with_info=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
    include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=5)
model.save("flower_classifier_model.h5")
