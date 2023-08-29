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
from keras import Sequential
from keras.layers import *
from keras.losses import CategoricalCrossentropy
#===============Import the required libraries===============


#===============import a list with the 3 classes and AUTOTUNE===============
CLASSES = ["rock", "paper", "scissors"]
AUTOTUNE = tf.data.experimental.AUTOTUNE
#===============import a list with the 3 classes and AUTOTUNE===============


def load_image_and_label(image_path, target_size = (32, 32)):
    """
    * Load the image and its label, one-hot encoding by comparing the name of the folder that
    contains the image (extracted from image_path) with the CLASSES list.

    * Parameters: 
    - image_path: older that contains the image.
    - target_size: height and width to resize image.

    * Return:
        - image and its label.
    """

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


def build_network():
    """
    * Build simple ConvNets.

    * Return:
        - Sequential model.
    """

    model = Sequential()
    model.add(Input(shape=(32,32,1)))
    model.add(Conv2D(32, kernel_size=(3,3),activation="relu", padding="same"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    return model


def prepare_dataset(dataset_path, buffer_size, batch_size, shuffle=True):
    """
    * Prepare dataset using TF infrastructure.

    * Parameters: 
    - dataset_path: images path (dataset).
    - buffer_size: buffer size to shuffle the images.
    - batch_size: the number of images in each batch.
    - shuffle: bool -> True do shuffle images.

    * Return:
        - tf.data.Dataset instance of images and labels.
    """

    dataset = (tf.data.Dataset.from_tensor_slices(dataset_path).map(load_image_and_label, num_parallel_calls=AUTOTUNE))

    if shuffle:
        dataset.shuffle(buffer_size=buffer_size)
    
    dataset = (dataset.batch(batch_size=batch_size).prefetch(buffer_size=buffer_size))
    return dataset


def imagespaths_into_list():
    """
    * Load the image paths into a list.

    * Return:
        - Dataset paths.
    """
    files_pattern = pathlib.Path.home() /'Desktop'/'datasets'/ 'rockpaperscissors' / 'rps-cv-images' / '*'/ '*.png'
    
    files_pattern = str(files_pattern)
    dataset_paths = [*glob.glob(files_pattern)]
    return dataset_paths


def train_valid_test_paths():
    '''
    * Create train, test, and validation subsets of image paths

    * Return:
        - tuple of train, valid, test paths.
    '''

    dataset_paths = imagespaths_into_list()
    train_paths, test_paths = train_test_split(dataset_paths, test_size=0.2, random_state=42)
    train_paths, valid_paths = train_test_split(train_paths, test_size=0.2, random_state=42)
    return train_paths, valid_paths, test_paths


def train_valid_test_datasets():
    '''
    * Prepare train, test, and validation subsets of datasets.

    * Return:
        - tuple of train, valid, test datasets.
    '''

    BATCH_SIZE = 1024
    BUFFER_SIZE = 1024

    train_paths, valid_paths, test_paths = train_valid_test_paths()

    train_dataset = prepare_dataset(train_paths, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    valid_dataset = prepare_dataset(valid_paths, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset = prepare_dataset(test_paths,buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,shuffle=False)
    return train_dataset, valid_dataset, test_dataset


def build_compile(model):
    """ 
    * model compile

    * Parameters: 
    - model: model architecture.
    """

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def main():

    #data
    train_dataset, valid_dataset, test_dataset = train_valid_test_datasets()

    #fit
    EPOCHS = 150
    model = build_network()
    build_compile(model)

    model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS)

    #evaluate
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss} and Test Accuracy {test_accuracy}.")


#run script
if __name__ == "__main__":
    main()


