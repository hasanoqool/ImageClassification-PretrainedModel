"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import os 
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
AUTOTUNE = tf.data.experimental.AUTOTUNE

def build_network(width, height, depth, classes):
    input_layer = Input(shape=(width, height, depth))
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32,
    kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)
    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=classes)(x)
    output = Softmax()(x)
    return Model(input_layer, output)

def image_paths():
    base_path = (pathlib.Path.home() /'Desktop' / 'datasets' /'caltech101' /'101_ObjectCategories')

    images_pattern = str(base_path / '*' / '*.jpg')
    image_paths = [*glob(images_pattern)]
    image_paths = [p for p in image_paths if
    p.split(os.path.sep)[-2] !='BACKGROUND_Google']
    return image_paths

def plot_model_history(model_history, metric, plot_name):
    plt.style.use('seaborn-darkgrid')
    plotter = tfdocs.plots.HistoryPlotter()
    plotter.plot({'Model': model_history}, metric=metric)
    plt.title(f'{metric.upper()}')
    plt.ylim([0, 1])
    plt.savefig(f'{plot_name}.png')
    plt.close()

image_paths = image_paths()

CLASSES = np.unique([p.split(os.path.sep)[-2] for p in image_paths])

def load_image_and_label(image_path, target_size=(64, 64)):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.convert_image_dtype(image, np.float32)

    image = tf.image.resize(image, target_size)

    label = tf.strings.split(image_path, os.path.sep)[-2]

    label = (label == CLASSES) # One-hot encode.

    label = tf.dtypes.cast(label, tf.float32)

    return image, label


def augment(image, label):

    image = tf.image.resize_with_crop_or_pad(image, 74, 74)

    image = tf.image.random_crop(image, size=(64, 64, 3))

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, 0.2)

    return image, label


def prepare_dataset(data_pattern):
    return (tf.data.Dataset.from_tensor_slices(data_pattern).map(load_image_and_label, num_parallel_calls=AUTOTUNE))


def main():

    train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=SEED)

    BATCH_SIZE = 64
    BUFFER_SIZE = 1024
    EPOCHS = 40

    #without augmentation
    train_dataset = (prepare_dataset(train_paths).batch(BATCH_SIZE).shuffle(buffer_size=BUFFER_SIZE).prefetch(buffer_size=BUFFER_SIZE))
    test_dataset = (prepare_dataset(test_paths).batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE))

    model = build_network(64, 64, 3, len(CLASSES))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(train_dataset,
    validation_data=test_dataset, epochs=EPOCHS)
    result = model.evaluate(test_dataset)
    print(f'Test accuracy: {result[1]}')
    plot_model_history(history, 'accuracy', 'normal')
    #without augmentation

    #with augmentation
    train_dataset = (prepare_dataset(train_paths).map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).shuffle(buffer_size=BUFFER_SIZE).prefetch(buffer_size=BUFFER_SIZE))
    test_dataset = (prepare_dataset(test_paths).batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE))

    model = build_network(64, 64, 3, len(CLASSES))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS)
    result = model.evaluate(test_dataset)
    print(f'Test accuracy: {result[1]}')
    plot_model_history(history, 'accuracy', 'augmented')
    #with augmentation


#run script
if __name__ == "__main__":
    main()