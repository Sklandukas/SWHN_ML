import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

with open("/content/drive/MyDrive/Colab_Notebooks/Number_test/train.npy", "rb") as f:
    data = np.load(f)
    images = data["images"]
    labels = data["labels"]

with open("/content/drive/MyDrive/Colab_Notebooks/Number_test/test.npy", "rb") as f:
    data = np.load(f)
    images_test = data["images"]

X_train = np.reshape(images.astype(np.float32), (73257, 32, 32 ,3)) / 255.0 - 0.5
y_train = tf.keras.utils.to_categorical(labels)

x_test = np.reshape(images_test.astype(np.float32), (26032, 32, 32 ,3)) / 255.0 - 0.5

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(10, activation="softmax")                       
])

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

model.summary()
model.fit(X_train, y_train, batch_size=128, epochs=50)

loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"Train accuracy = {accuracy:.4f}")

y_pred = model.predict(X_train)
predicted_categories = tf.argmax(y_pred, axis=1)
print(predicted_categories)
#prediction = pd.DataFrame(predicted_categories, columns=['predictions']).to_csv('/content/drive/MyDrive/Colab_Notebooks/prediction_train_04.01.csv', index=None)

y_pred_test = model.predict(x_test)
predicted_categories_test = tf.argmax(y_pred_test, axis=1)
print(predicted_categories_test)
#prediction_test = pd.DataFrame(predicted_categories_test, columns=['predictions']).to_csv('/content/drive/MyDrive/Colab_Notebooks/predicted_categories_test_04.01.csv', index=None)

test_pred = model.predict(X_train)

test_cast = np.argmax(test_pred, axis = 1)
accuracy = np.mean(test_cast == images[..., 0])

print(f"Test accuracy = { accuracy:.4f}")