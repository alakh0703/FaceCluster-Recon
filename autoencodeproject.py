
from keras.layers import Cropping2D
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# Load the data
data = np.load('new_norm_575_data.npy')

# Shuffle the data
idx = np.random.permutation(len(data))
data_shuffled = data[idx]

# Split into training and validation set
train_set = data_shuffled[:-100]
val_set = data_shuffled[-100:]

# Add noise to data 
noise_factor = 0.5
train_noisy = train_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_set.shape)
val_noisy = val_set + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=val_set.shape)

# Define the input shapes and creating the model
input_shape_1 = ( 112, 92,1)

input_1 = Input(shape=input_shape_1)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_1)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded_1 = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded_1)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded_1 = Conv2D(1, (3, 3), activation='linear', padding='same')(x)

autoencoder_1 = Model(input_1, decoded_1)
autoencoder_1.compile(optimizer='adam', loss='mean_squared_error')
autoencoder_1.summary()

# Fit the model
history = autoencoder_1.fit(train_noisy, train_set,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(val_noisy, val_set))



# Save the model
autoencoder_1.save('new_autoencoder_1.h5')

# Plot the images from val set original, with noise, and predicted from model of first 5 
import matplotlib.pyplot as plt
decoded_imgs = autoencoder_1.predict(val_noisy)
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(val_set[i].reshape(112, 92))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(val_noisy[i].reshape(112, 92))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(decoded_imgs[i].reshape(112, 92))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
