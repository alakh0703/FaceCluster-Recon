# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:52:13 2023

@author: gitan
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from scipy.ndimage import rotate
import pandas as pd
import cv2
import random



def total_img_of_each_person(df1):
    a = []
    for i in range(20):
        x = df1[i][0].shape[2]
        a.append(x)
    return a
    
def select_two_random_image(df1,number_of_images_of_each_person,r):
    new_list = []

    for i in range(20):
        random_1 = random.randint(0,(number_of_images_of_each_person[i]-1))
        random_2 = random.randint(0,(number_of_images_of_each_person[i]-1))
        img1 = df1[i][0][:,:,random_1]
        img2 = df1[i][0][:,:,random_2]
        new_list.append(img1)
        new_list.append(img2)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
        r.append(i)
    return  new_list
    

def random_shift_image(image, max_shift=10):
    # Generate random shifts in the horizontal and vertical directions
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)

    # Create a new image with white space
    new_image = np.zeros_like(image)
    
    # Calculate the new coordinates for placing the original image
    x1, x2 = max(0, shift_x), min(image.shape[1], image.shape[1] + shift_x)
    y1, y2 = max(0, shift_y), min(image.shape[0], image.shape[0] + shift_y)

    # Calculate the new coordinates for placing the image within the new image
    new_x1, new_x2 = max(-shift_x, 0), min(image.shape[1], image.shape[1] - shift_x)
    new_y1, new_y2 = max(-shift_y, 0), min(image.shape[0], image.shape[0] - shift_y)

    # Copy the original image to the new image
    new_image[new_y1:new_y2, new_x1:new_x2] = image[y1:y2, x1:x2]

    return new_image

def zoom_image(image, zoom_factor=2):
    if zoom_factor <= 1:
        return image  

    height, width = image.shape[:2]

    center_x, center_y = width // 2, height // 2

    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    x1 = center_x - new_width // 2
    x2 = x1 + new_width
    y1 = center_y - new_height // 2
    y2 = y1 + new_height

    zoomed_image = image[y1:y2, x1:x2]

    zoomed_image = cv2.resize(zoomed_image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return zoomed_image

def flip_image_horizontal(image):
    return np.fliplr(image)

def rotate_image(image, angle_degrees):
    angle_degrees = angle_degrees % 360 
    if angle_degrees > 180:
        angle_degrees -= 360
    rotated_image = rotate(image, angle_degrees, reshape=False, order=1, mode='constant', cval=0)
    return rotated_image

def darken_image(image, gamma = 0.5):
        darkened_face = exposure.adjust_gamma(image, gamma)  
        return darkened_face
    

def plot_all_images(image_data):
    num_images = image_data.shape[2]
    
    for image_index in range(num_images):
        image = image_data[:, :, image_index].astype(float)
        
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f"Image {image_index}")
        plt.show()

def print_image(image,image_shape=(112, 92), t = "Normal Image"):
    plt.imshow(image, cmap='gray')  
    plt.title(t)
    plt.axis('off')  # Turn off the axis
    plt.show()
    
    
def plot_last_image(image_data,x,image_index):    
    image = image_data[:, :, x].astype(float)
        
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(f"Image {image_index}")
    plt.show()


def plot_last_image_of_each(df,q,image_index):
    for i in range(20):
        plot_last_image(df[i][0],q,image_index)
    
def normalize_grayscale_image(image_data):
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)

    image_data /= 255.0 

    return image_data

def convert_to_dataframe(norm_data):
    flattened_matrices = []
    for matrix in norm_data:
        flattened_matrices.append(matrix.ravel())
    df = pd.DataFrame(flattened_matrices)
    return df

def add_gaussian_noise(image, mean=0, std_dev=0.1):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = np.clip(image + noise, 0, 1) 
    return noisy_image