
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data_575 = pd.read_csv("C:/Users/Dhrumit Patel/College/3402 - Semester 5/COMP 257 - Unsupervised and Reinforcement Learning/Project/normalized_dataframe_575.csv")
# Convert to 2D
data_575_2d = np.array(data_575).reshape(575, 112, 92, 1)

X_train, X_val = train_test_split(data_575_2d, test_size=0.2, random_state=42)
X_train.shape
X_val.shape

# Adding noise to your data
noise_factor = 0.2
X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)

# Define an Autoencoder model with CNN
input_shape = (112, 92, 1)
code_size = 128

# Encoder
input_img = keras.layers.Input(shape=input_shape)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
code = keras.layers.Conv2D(code_size, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(code)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.models.Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Training the autoencoder
autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val))

# Using the trained autoencoder to denoise the data
denoised_data = autoencoder.predict(X_val)

# Visualizing the original data, noisy data, and denoised data
num_samples_to_visualize = 5

for i in range(num_samples_to_visualize):
    sample_index = np.random.randint(0, X_val.shape[0])

    original_sample = X_val[sample_index]
    noisy_sample = X_train_noisy[sample_index]
    denoised_sample = denoised_data[sample_index]

    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.imshow(original_sample.reshape(112, 92), cmap='gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(noisy_sample.reshape(112, 92), cmap='gray')
    plt.title('Noisy Image')

    plt.subplot(133)
    plt.imshow(denoised_sample.reshape(112, 92), cmap='gray')
    plt.title('Denoised Image')

    plt.show()

"""Hyperparameter Tuning - CNN"""
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Define a function to create the autoencoder model
def create_autoencoder(hidden_size, code_size, learning_rate):
    input_shape = (112, 92, 1)
    
    # Encoder
    input_img = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    code = keras.layers.Conv2D(code_size, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(code)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.models.Model(input_img, decoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
    
    return autoencoder

# Define the hyperparameter search space
param_grid = {
    'hidden_size': [512, 1024, 2048],
    'code_size': [128, 256],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Create a KerasRegressor for use with GridSearchCV
autoencoder = KerasRegressor(build_fn=create_autoencoder, epochs=50, batch_size=32, verbose=1)

# Create a grid search object
grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# Perform the hyperparameter search
grid_result = grid.fit(X_train_noisy, X_train)

# Get the best hyperparameters
best_hidden_size = grid_result.best_params_['hidden_size']
best_code_size = grid_result.best_params_['code_size']
best_learning_rate = grid_result.best_params_['learning_rate']

# Train the final model with the best hyperparameters
best_autoencoder = create_autoencoder(best_hidden_size, best_code_size, best_learning_rate)
best_autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val))

# Using the trained autoencoder with the best hyperparameters to denoise the data
denoised_data_best = best_autoencoder.predict(X_val)

best_autoencoder.save('autoencoder_d.h5')

# Visualizing the original data, noisy data, and denoised data using the best hyperparameters
num_samples_to_visualize = 5

for i in range(num_samples_to_visualize):
    sample_index = np.random.randint(0, X_val.shape[0])

    original_sample = X_val[sample_index]
    noisy_sample = X_train_noisy[sample_index]
    denoised_sample = denoised_data_best[sample_index]

    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.imshow(original_sample.reshape(112, 92), cmap='gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(noisy_sample.reshape(112, 92), cmap='gray')
    plt.title('Noisy Image')

    plt.subplot(133)
    plt.imshow(denoised_sample.reshape(112, 92), cmap='gray')
    plt.title('Denoised Image (Best Hyperparameters)')

    plt.show()