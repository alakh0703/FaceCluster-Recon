# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:33:33 2023

@author: Dhrumit Patel
"""

import random
from skimage import exposure
from scipy.ndimage import rotate
from sklearn.model_selection import StratifiedShuffleSplit
import os 
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from functions import plot_all_images
from functions import print_image
from functions import plot_last_image_of_each
from functions import plot_last_image

# Loading the data
data = loadmat("umist_cropped.mat")
data.keys()

# Extract necessary data from the loaded mat file
x1 = data["facedat"]
x2 = data["dirnames"]

# Convert data to DataFrames and save as CSV files
df1 = pd.DataFrame(x1)
df2 = pd.DataFrame(x2)

df1.to_csv(r"facedat")
df2.to_csv(r"dirnames")

image_index = 1 
    
# Extracting the image and plotting the images of the data
img = df1[0][0][:,:,2]
print_image(img)    
plot_last_image(df1[0][0],0,image_index)
plot_all_images(df1[0][0])
plot_last_image_of_each(df1,18,image_index)


# Exploring the data structure
number_of_each_image =[]
df1[0][0].shape[2]

for i in range(20):
    x = df1[i][0].shape[2]
    number_of_each_image.append(x)
    
# Storing the images and labels
group = []
actual_y = []
for a in range(20):
    bc = df1[a][0].shape[2]
    for i in range(bc):
        group.append(df1[a][0][:,:,i])
        actual_y.append(a)
    
image_data = np.array(group)
actual_labels = np.array(actual_y)

image_data_reshaped = image_data.reshape(-1, 112 * 92)
actual_data = pd.DataFrame(image_data_reshaped)
actual_data.shape

# Importing the custom functions for preprocessing
from functions import normalize_grayscale_image
from functions import convert_to_dataframe

# Normalizing the grayscale images
image_data[0]
norm_data = normalize_grayscale_image(image_data)

# Converting normalized data to DataFrame
image_data_norm = convert_to_dataframe(norm_data)

# Saving the normalized DataFrame to a CSV file
csv_file_path = 'new_normalized_dataframe_of_575_images.csv'
image_data_norm.to_csv(csv_file_path, index=False)

# Adding Gaussian Noise to the normalized data
from functions import add_gaussian_noise
noisy_data = add_gaussian_noise(norm_data)


# Importing the image manipulation functions - for adding extra images
from functions import flip_image_horizontal
from functions import rotate_image
from functions import darken_image
from functions import zoom_image
from functions import random_shift_image

# Selecting example images for manipulation
img = df1[0][0][:,:,2]
I1 = df1[0][0][:,:,33]
I2 = df1[1][0][:,:,30]
I3 = df1[15][0][:,:,20]

# Plotting the original as well as manipulated images
print_image(I1)
print_image(I2)
print_image(I3)

# Flipped images horizontally
I1_flipped = flip_image_horizontal(I1)
I2_flipped = flip_image_horizontal(I2)
I3_flipped = flip_image_horizontal(I3)

print_image(I1_flipped)
print_image(I2_flipped)
print_image(I3_flipped)


# Rotating images
rotate_angle = 45
I1_rotated = rotate_image(I1, rotate_angle)
I2_rotated = rotate_image(I2, rotate_angle)
I3_rotated = rotate_image(I3, rotate_angle)

print_image(I1_rotated)


# Darkening Image
I1_darkened = darken_image(I1,2)
I2_darkened = darken_image(I2,2)
I3_darkened = darken_image(I3,3)

print_image(I1_darkened)
print_image(I2_darkened)
print_image(I3_darkened)


# Adjusting the image brightness - lightning the image
I1_light = darken_image(I1,0.2)
I2_light = darken_image(I2,0.5)
I3_light = darken_image(I3, 0.6)

print_image(I1_light)
print_image(I2_light)
print_image(I3_light)


# Zooming Image
zoom_factor = 1.35
I1_zoomed = zoom_image(I1, zoom_factor)
I2_zoomed = zoom_image(I2, zoom_factor)
I3_zoomed = zoom_image(I3, zoom_factor)

print_image(I1_zoomed)
print_image(I2_zoomed)
print_image(I3_zoomed)


# Shifting Image randomly
max_shift_amount = 15
I1_shifted = random_shift_image(I1, max_shift_amount)
I2_shifted = random_shift_image(I2, max_shift_amount)
I3_shifted = random_shift_image(I3, max_shift_amount)

print_image(I1_shifted)
print_image(I2_shifted)
print_image(I3_shifted)

# Importing additional custom functions
from functions import total_img_of_each_person
from functions import select_two_random_image
    
# Calculating the total number of images fir each person
number_of_images_of_each_person = total_img_of_each_person(df1)
number_of_images_of_each_person


# Initializing the list to store the indices of the image
r =  []
# Selecting 2 random images of each person and appending them to list 'r'
list_of_2_images_of_each_person = select_two_random_image(df1,number_of_images_of_each_person,r)
# Displaying the length of the list and displaying the image indices
len(list_of_2_images_of_each_person)
print(r)

# Looping through the selected images and printing each image
for i in list_of_2_images_of_each_person:
    print_image(i)

# Initializing the list to store modified images
the_other_images = []

# Looping through the selected images to apply the various image transformations 
# and creating modified versions
for i in list_of_2_images_of_each_person:
    random_shift_factor = random.randint(4,20)
    random_angle = random.randint(30, 120)
    random_gamma = random.uniform(1, 4)
    random_l_gamma = random.uniform(0.5,1)
    print(i)
    
    lighted_image = darken_image(i, random_l_gamma)
    darkened_image = darken_image(i, gamma = random_gamma)
    shifted_image = random_shift_image(i,random_shift_factor)
    zoomed_image = zoom_image(i, 1.35)
    rotated_image = rotate_image(i, random_angle) 
    flipped_image = flip_image_horizontal(i)
    
    the_other_images.append(np.array(lighted_image))
    the_other_images.append(np.array(darkened_image))
    the_other_images.append(np.array(zoomed_image))
    the_other_images.append(np.array(shifted_image))
    the_other_images.append(np.array(rotated_image))
    the_other_images.append(np.array(flipped_image))
    
# Looping through modified images and printing each image
for i in the_other_images:
    print_image(i)
    
# Appending all images (both original and modified)
for i in the_other_images:
    group.append(i)
    
# Stacking new images alongside a new axis
the_other_images = np.stack(the_other_images, axis=0)
# Normalizing the grayscale values of the modified images
the_other_images_norm = normalize_grayscale_image(the_other_images)
    
for i in group:
    print_image(i)
    
# Appending the images lables to the actual and then concatenating the original and modified images labels
actual_other_image_labels = np.array(r)
labels = np.concatenate((actual_labels, actual_other_image_labels), axis=0)

# Concatenating the complete data with total of 815 images
data_complete = np.concatenate((norm_data, the_other_images_norm), axis=0)

# Using PCA
from sklearn.decomposition import PCA
# Reshaping data for PCA
reshaped_data_complete = data_complete.reshape(815, 112 * 92)

# Storing 95% explained variance
pca = PCA(n_components=0.95, svd_solver='full')
pca.fit(reshaped_data_complete)
transformed_data = pca.transform(reshaped_data_complete)
transformed_data.shape

# Reconstructing the approximate data
approximated_data = pca.inverse_transform(transformed_data)
approximated_data = approximated_data.reshape(815, 112, 92)

# Fetching the PCA components and the explained variance ratio
components = pca.components_
explained_variance_ratio = pca.explained_variance_ratio_

# Displaying the number of components and explained variance ratio
print(f"Number of components: {pca.n_components_}")
print(f"Explained variance ratio: {explained_variance_ratio}")

# Displaying the shapes of transformed and approximate data
print(f"Transformed data shape: {transformed_data.shape}")
print(f"Approximated data shape: {approximated_data.shape}")

# Reshaping the transformed data
model_data_pca_without_autoencoder = transformed_data.reshape(815,12,10)
model_data_pca_without_autoencoder.shape


model_data  =  data_complete
model_data.shape
labels.shape

# Saving data arrays into the numpy files
np.save('new_model_data.npy', model_data)
np.save('new_model_data_pca_without_autoencoder.npy', model_data_pca_without_autoencoder)
np.save('new_label.npy', labels)
np.save('new_norm_575_data.npy', norm_data)
np.save('new_norm_240_data.npy', the_other_images_norm)
np.save('new_y_575.npy', actual_labels)
np.save('new_x_575.npy', actual_data)
