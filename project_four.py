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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

# Loading the data and the models
model_data = np.load('new_model_data.npy')
model_data_pca = np.load('new_model_data_pca_without_autoencoder.npy')
label = np.load('new_label.npy')

# Reshaping the model data
reshaped_data = model_data.reshape(815, -1)

# Shuffling the reshaped data and the model data
idx = np.random.permutation(len(reshaped_data))
reshaped_data_shuffled = reshaped_data[idx]
model_data_shuffled = model_data[idx]

# K-Means Clustering
# Initializing the clusters
n_clusters = 20

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(reshaped_data_shuffled)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
inertia = kmeans.inertia_
joblib.dump(kmeans, 'kmeans_model.pkl')

# Printing the results for K-Means
print("Cluster Labels: ", cluster_labels)
print("Cluster Centers: ", cluster_centers)
print("Inertia: ", inertia)

# Plotting the graph of K-Means
plt.scatter(reshaped_data_shuffled[:, 0], reshaped_data_shuffled[:, 1], c=cluster_labels)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
plt.title("Cluster Plot")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Agglomerative Clustering
n_clusters = 20
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_clustering.fit(reshaped_data_shuffled)
cluster_labels_agg = agg_clustering.labels_
joblib.dump(agg_clustering, 'agg_clustering_model.pkl')

# Printing the results for Agglomerative Clustering
print("Agglomerative Clustering - Cluster Labels: ", cluster_labels_agg)

# Plotting the graph of Agglomerative Clustering
plt.scatter(reshaped_data_shuffled[:, 0], reshaped_data_shuffled[:
, 1], c=cluster_labels_agg)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# DBSCAN Clustering
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan.fit(reshaped_data_shuffled)
cluster_labels_dbscan = dbscan.labels_
joblib.dump(dbscan, 'dbscan_model.pkl')

# Printing the results for DBSCAN Clustering
print("DBSCAN - Cluster Labels: ", cluster_labels_dbscan)

# Plotting the graph of DBSCAN Clustering
plt.scatter(reshaped_data_shuffled[:, 0], reshaped_data_shuffled[:
, 1], c=cluster_labels_dbscan)
plt.title("DBSCAN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# Reshaping data for visualization
final_data  = reshaped_data_shuffled.reshape(815,112,92)
final_labels = cluster_labels


# Displaying images with the respective labels
for i in range(len(model_data_shuffled)):
    image = model_data_shuffled[i]
    label = final_labels[i]
    if i > 0 and label != final_labels[i-1]:
        plt.show()
    print_image(image, t=f"Label: {label}")
plt.show()

# Saving the labeled data
np.save('final_data__shuffled_normalized_nopca.npy',final_data)
np.save('final_label.npy',final_labels)

# Loading the labeled data
final_data = np.load('final_data__shuffled_normalized_nopca.npy')
final_labels = np.load('final_label.npy')

# Using Train Test Split to split the data to 80:20 ratio
x_train, x_val, y_train, y_val = train_test_split(final_data, final_labels,
                                                    test_size=0.2,
                                                    random_state=42)


# Defining the model -  CNN
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(112, 92, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(20, activation="softmax"),
    ]
)

# Compiling the model and printing the summary
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Training the model
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

# Evaluating the model on test set and printing the test accuracy
score = model.evaluate(x_val, y_val, verbose=1)
print("Test accuracy:", score[1])

# Saving the model for CNN
model.save("model_1.keras")

# Plotting the training and validation accuracies and the loss for CNN model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Predicting the labels for the first ten images
predicted_labels = model.predict(x_val[:10])
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



# ResNet Model
# Repeating the grayscale image 3 times to create a 3D image
x_train_rgb = np.repeat(x_train[..., np.newaxis], 3, -1)
x_val_rgb = np.repeat(x_val[..., np.newaxis], 3, -1)
print(x_train_rgb[1])

# Loading the model without classifier layers
model = ResNet50(include_top=False, input_shape=(112, 92, 3))

# Adding layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu')(flat1)
output = Dense(20, activation='softmax')(class1)

# Defining the model - ResNet
model = Model(inputs=model.inputs, outputs=output)
model.summary()

# Compiling the model and printing out the summary of the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Trainin the model
history2 = model.fit(x_train_rgb, y_train, epochs=10, validation_data=(x_val_rgb, y_val))

# Evaluating the model on test set and printing the accuracy
score = model.evaluate(x_val_rgb, y_val, verbose=1)
print("Test accuracy:", score[1])

# Saving the model - ResNet
model.save("model_resnet.h5")
model.save("model_resnet.keras")

# Plotting the training and validation accuracies and the loss for ResNet model
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predicting the labels for the first ten images
predicted_labels = model.predict(x_val_rgb[:10])
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

