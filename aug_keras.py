"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import *
#===============Import the required libraries===============

SEED = 999
np.random.seed(SEED)

def load_images_and_labels(image_paths, target_size=(64, 64)):
    """
    * Load images and labels from list of file paths.

    * Parameters: reame
        - image_paths: dataset path from directory.
        - target_size: image height and width.

    * Return:
        - Numpy array --> array(features, labels).
    """

    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)

        label = image_path.split(os.path.sep)[-2]
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


def build_network(width, height, depth, classes):
    """
    * build a smaller version of VGG.
    """

    input_layer = Input(shape=(width, height, depth))

    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)
    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(units=classes)(x)

    output = Softmax()(x)
    return Model(input_layer, output)


def plot_model_history(model_history, metric, plot_name):
    """
    * plot and save a model's training curve
    """

    plt.style.use('seaborn-darkgrid')
    plotter = tfdocs.plots.HistoryPlotter()
    plotter.plot({'Model': model_history}, metric=metric)
    plt.title(f'{metric.upper()}')
    plt.ylim([0, 1])
    plt.savefig(f'{plot_name}.png')
    plt.close() 


def load_paths():
    """ 
    * Load the paths to all images in the dataset, excepting the ones of the BACKGROUND_Google class
    """

    base_path = (pathlib.Path.home() /'Desktop' / 'datasets' /'caltech101' /'101_ObjectCategories')
    images_pattern = str(base_path / '*' / '*.jpg')
    image_paths = [*glob(images_pattern)]
    image_paths = [p for p in image_paths if p.split(os.path.sep)[-2] !=  'BACKGROUND_Google']

    classes = {p.split(os.path.sep)[-2] for p in image_paths}

    return image_paths, classes


def main():

    #Load the dataset into memory, normalizing the images and one-hot encoding the labels
    image_paths, classes = load_paths()
    X, y = load_images_and_labels(image_paths)
    X = X.astype('float') / 255.0
    y = LabelBinarizer().fit_transform(y)

    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=SEED)

    EPOCHS = 40
    BATCH_SIZE = 64 

    # model = build_network(64, 64, 3, len(classes))

    # model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])        
        
    # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # result = model.evaluate(X_test, y_test)

    # print(f'Test accuracy: {result[1]}')

    # plot_model_history(history, 'accuracy', 'normal')

    #after Augmentation

    model = build_network(64, 64, 3, len(classes))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    augmenter = ImageDataGenerator(horizontal_flip=True, rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, fill_mode='nearest')
    train_generator = augmenter.flow(X_train, y_train, BATCH_SIZE)

    hist = model.fit(train_generator, validation_data=(X_test, y_test), epochs=EPOCHS)

    result = model.evaluate(X_test, y_test)

    print(f'Test accuracy: {result[1]}')

    plot_model_history(hist, 'accuracy', 'augmented')

#run script
if __name__ == "__main__":
    main()