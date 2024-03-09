# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:33:33 2023

@author: Dhrumit Patel
"""

import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from functions import print_image
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from keras.models import load_model

# Loading the nesscary the data and the models
model_data = np.load('new_model_data.npy')
model_data_pca = np.load('new_model_data_pca_without_autoencoder.npy')
label = np.load('new_label.npy')

kmeans_model = joblib.load('kmeans_model.pkl')

final_data = np.load('final_data__shuffled_normalized_nopca.npy')
final_labels = np.load('final_label.npy')

# Displaying the images with the respective labels
for i in range(len(final_data)):
    image = final_data[i]
    label = final_labels[i]
    if i > 0 and label != final_labels[i-1]:
        plt.show()
    print_image(image, t=f"Label: {label}")
plt.show()


# Splitting the data for validation
x_train, x_val, y_train, y_val = train_test_split(final_data, final_labels,
                                                    test_size=0.2,
                                                    random_state=42)
# Loading and predicting using our first model - CNN
model = load_model('model_1.keras')
predicted_labels = model.predict(x_val[:10])
predicted_labels = np.argmax(predicted_labels, axis=1)

# Displaying the predicted labels and actual labels for the first ten images
print("Predicted Labels: ", predicted_labels)
print("Actual Labels: ", y_val[:10])

# Plotting the images with predicted and actual labels
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_val[i].reshape(112, 92), cmap='gray')
    ax.set_title(f"Predicted: {predicted_labels[i]}, Actual: {y_val[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# Loading and predicting the second model - ResNet
model2 = load_model('model_resnet.keras')
x_val_rgb = np.repeat(x_val[..., np.newaxis], 3, -1)

# Predict the labels for the first ten images
predicted_labels = model2.predict(x_val_rgb[:10])
predicted_labels = np.argmax(predicted_labels, axis=1)

# Printing the predicted labels and actual labels
print("Predicted Labels: ", predicted_labels)
print("Actual Labels: ", y_val[:10])

# Plotting the images with predicted and actual labels
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_val[i].reshape(112, 92), cmap='gray')
    ax.set_title(f"Predicted: {predicted_labels[i]}, Actual: {y_val[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()