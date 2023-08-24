"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import os
import pathlib
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.preprocessing.image import load_img, img_to_array
#===============Import the required libraries===============


def load_images_labels(image_paths):
    """
    * Load images and labels from list of file paths.

    * Parameters:
        - image_paths: dataset path from directory.

    * Return:
        - Numpy array --> array(features, labels).

    """
    images = []
    labels = []    

    for image_path in image_paths:

        image = load_img(image_path, target_size=(32,32), color_mode="grayscale")
        image = img_to_array(image)
        label = image_path.split(os.path.sep)[-2]
        label = "positive" in label
        label = float(label)

        images.append(image)
        labels.append(label)
    
    return np.array(images), np.array(labels)
# print(load_images_labels.__doc__)

s = float("post" in "post")
print(s)