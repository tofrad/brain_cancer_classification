import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import confusion_matrix

print(tf.version.VERSION)

data_dir = "dataset"

#create dataframe to analyse dataset-----------------------------------
folders = os.listdir(data_dir)
paths = []
labels = []

for folder in folders:
    path_to_folder = os.path.join(data_dir, folder)
    files = os.listdir(path_to_folder)

    for file in files:
        paths.append(os.path.join(data_dir, folder, file))
        labels.append(folder.split(' ')[0])

data_frame = pd.DataFrame(data = {"paths": paths, "labels": labels})

class_counts = data_frame["labels"].value_counts()
print(class_counts)

class_counts.plot(kind="barh", color="lightgreen")
plt.title("class distribution")
plt.xlabel("amount of images")
plt.ylabel("classes")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

#create datasets for training-------------------------------------------
img_height = 350
img_width = 350

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=17,
    validation_split=0.1,
    subset="both",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

#get sample from dataset-------------------------------------------------
class_names = train_ds.class_names
print("class names:", class_names)

images_batch, labels_batch = next(iter(train_ds))

# one hot code to string
labels_np = np.array([class_names[label] for label in labels_batch])

images_np = images_batch.numpy()
images_uint8 = images_np.astype(np.uint8)

plt.figure(figsize=(13, 13))
for i in range(16):

    plt.subplot(4, 4, i+1)
    plt.imshow(images_uint8[i])

    plt.title(f"{labels_np[i]}", fontsize=9, pad=5)
    plt.axis("off")

plt.tight_layout()
plt.show()
#------------------------------------------------------------------------

label_map = (train_ds.class_names)
class_cnt = len(label_map)

#create model-----------------------------------------------------------
layers = tf.keras.layers

resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(img_height, img_width, 3),
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
resnet.trainable = True

for layer in resnet.layers[:-7]:
    layer.trainable = False

#model_cnn
model = tf.keras.models.Sequential([
    resnet,

    layers.AveragePooling2D(pool_size=(4, 4)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),

    layers.Dense(class_cnt, activation='softmax')
])

#----------------------------------------------------------------------
savepoint = "model.h5"
checkpoint = ModelCheckpoint(savepoint,
                             monitor= 'accuracy',
                             save_best_only=True,
                             verbose=1)

# Trainingsparameter
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)

#Train
history = model.fit(train_ds,
                    batch_size=32,
                    validation_data=val_ds,
                    epochs=10,
                    callbacks=[checkpoint])


plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy over epochs')
plt.show()

#Predicts and Confusion Matrix--------------------------------------------------------
loss, accuracy = model.evaluate(val_ds)

predictions = model.predict(val_ds)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y for x, y in val_ds], axis=0)
classes = val_ds.class_names

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.title("Confusion Matrix", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------
