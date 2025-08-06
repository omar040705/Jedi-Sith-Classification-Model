#Imports
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import glob

#Setting the directory for the data
data_dir = r"C:\\Users\omar0\Documents\Ben and Omar Model\Ben and Omar Model\starwars\train"

# Getting everything within directories
image_count = len(list(glob.glob(os.path.join(data_dir, '*/**.jpg'))))
print(image_count)

jedi_count = len(list(glob.glob(os.path.join(data_dir, "jedi/*"))))
sith_count = len(list(glob.glob(os.path.join(data_dir, "sith/*"))))

print(jedi_count)
print(sith_count)

#Load data using keras and create dataset
batch_size = 32
img_height = 360
img_width = 360


# Keras Training

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Keras Validation
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Things
class_names = train_ds.class_names
print(class_names)


# Visualization

''''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
'''

# Standardization
normalization_layer = tf.keras.layers.Rescaling(1./255)


# Normalizing the dataset map
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Buffered Prefetching (wud tf)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Gaming time

num_classes = 2

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])


# Optimizing the model with adam
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
# Make model

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20
)

# Predict
test_dir = r"C:\\Users\omar0\Documents\Ben and Omar Model\Ben and Omar Model\starwars\test"
img = tf.keras.utils.load_img(
    test_dir, target_size=(360, 360)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



