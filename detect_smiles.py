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
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
#===============Import the required libraries===============


def load_images_and_labels(image_paths):
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


def build_network():
    """
    * Build the neural network.

    * Return:
        - Sequential model.
    """
    model = Sequential()
    #block1
    model.add(Input(shape=(32,32,1)))
    model.add(Conv2D(filters=20, kernel_size=(5,5), activation="elu", padding="same", strides=(1,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    #block2
    model.add(Conv2D(filters=50, kernel_size=(5,5),activation="elu", padding="same", strides=(1,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    #block3
    model.add(Flatten())
    model.add(Dense(500, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid")) #binary classification  
    
    return model


def imagespaths_into_list():
    """
    * Load the image paths into a list.

    * Return:
        - Dataset paths.
    """
    files_pattern = pathlib.Path.home() /'Desktop'/'datasets'/ 'SMILEsmileD-master' / 'SMILEs' / '*'/ '*' / '*.jpg'
    
    files_pattern = str(files_pattern)
    dataset_paths = [*glob.glob(files_pattern)]
    return dataset_paths


def norm_count(X, y):
    """
    * Normalize the images and compute the number of positive, negative, and total
      examples in the dataset.

    * Parameters: 
        - X: features.
        - y: label.

    * Return:
        - Tuple of Numpy arrays --> (X, total, positive, negative).
    
    """
    X /= 255.
    total = len(y)
    positive = np.sum(y)
    negative = total - positive

    return X, total, positive, negative


def load_memory():
    """ 
    * Load the dataset into memory 

    * Return:
        - X, y --> data to split it
    """
    dataset_paths = imagespaths_into_list()
    X, y = load_images_and_labels(dataset_paths)
    return X, y


def train_test_valid(X, y):
    """ 
    * split data

    * Return:
        - Tuple of Numpy arrays --> (X_train, y_train),(X_test, y_test),(X_val, y_val)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return (X_train, y_train),(X_test, y_test),(X_val, y_val)


def model_compile(model):
    """ 
    * model compile
    """
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def main():

    BATCH_SIZE = 32
    EPOCHS = 15

    model = build_network()
    model_compile(model)

    X, y = load_memory()

    X, total, positive, negative = norm_count(X, y)

    # plt.imshow(X[50],cmap='gray')

    (X_train, y_train),(X_test, y_test),(X_val, y_val) = train_test_valid(X, y)

    print((X_train.shape, y_train.shape),(X_test.shape, y_test.shape),(X_val.shape, y_val.shape))

    model.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight={
    1.0: total / positive,
    0.0: total / negative})


    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f"Test loss: {test_loss} and Test Accuracy {test_accuracy}.")

#run script
if __name__ == "__main__":
    main()