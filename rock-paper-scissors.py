"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import os
import pathlib
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import *
from keras.losses import CategoricalCrossentropy
#===============Import the required libraries===============


#===============import a list with the 3 classes and AUTOTUNE===============
CLASSES = {"rock", "paper", "scissors"}
AUTOTUNE = tf.data.experimental.AUTOTUNE
#===============import a list with the 3 classes and AUTOTUNE===============


def load_image_and_label(image_path, target_size = (32, 32)):
    
    #image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, np.float32)
    image = tf.image.resize(image, target_size)

    #label
    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CLASSES) # One-hot encoding
    label = tf.dtypes.cast(label, tf.float32)
    return image, label


