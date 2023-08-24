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
from tensorflow.keras.preprocessing.image import *
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
        image = load
# print(load_images_labels.__doc__)