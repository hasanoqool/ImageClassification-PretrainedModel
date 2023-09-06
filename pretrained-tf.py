"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
from keras.models import Sequential
from keras.preprocessing.image import *
from keras.utils import get_file
#===============Import the required libraries===============


def model():
    """
    * Read ResNetV2152 model weights through provided link.
        
    * Return:
        - Sequential model.
    """
    url = ("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4")
    model = Sequential()
    model.add(hub.KerasLayer(url, input_shape = (224, 224, 3)))
