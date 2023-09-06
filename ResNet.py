"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import os
import numpy as np
import tarfile
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
from keras.utils import get_file
#===============Import the required libraries===============

AUTOTUNE = tf.data.experimental.AUTOTUNE


def residual_module(data, filters, stride, reduce=False, reg=0.0001, bn_eps=2e-5, bn_momentum=0.9):
    """
    * Create a residual module in ResNet architecture

    * Parameters: 
    - data: model layers.
    - filters: height and width to resize image.
    - stride: stride
    - reduce: defalut false if true will add new conv2d
    - reg: lambda value
    - bn_eps: epsilon value for normalization
    - bn_momentum: momentum

    * Return:
        - residual module.
    """

    model = Sequential()

    model.add(BatchNormalization(epsilon=bn_eps, momentum=bn_momentum))
    model.add(ReLU())

    model.add(Conv2D(filters=int(filters/4.), kernel_size=(1,1), use_bias=False, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(epsilon=bn_eps, momentum=bn_momentum))
    model.add(ReLU())

    model.add(Conv2D(filters=int(filters/4.), kernel_size=(1,1), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(epsilon=bn_eps, momentum=bn_momentum))
    model.add(ReLU())

    model.add(Conv2D(filters=filters, kernel_size=(1,1), use_bias=False, kernel_regularizer=l2(reg)))
    if reduce:
        model.add(Conv2D(filters=filters, kernel_size=(1,1), strides=stride, use_bias=False, kernel_regularizer=l2(reg)))
    return model


def build_resnet(input_shape, classes, stages, filters, reg=1e-3, bn_eps=2e-5, bn_momentum=0.9):
    """
    * Build custom ResNet

    * Parameters: 
    - input_shape: image shape (height,width,CH).
    - classes: # classes in dataset.
    - stages: to use (1,1) or (2,2) strides
    - filters: height and width to resize image.
    - reg: lambda value
    - bn_eps: epsilon value for normalization
    - bn_momentum: momentum

    * Return:
        - residual module.
    """

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(BatchNormalization(epsilon=bn_eps, momentum=bn_momentum))
    model.add(Conv2D(filters[0], kernel_size=(3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg)))

    for i in range(len(stages)):

        stride = (1, 1) if i == 0 else (2, 2)
        model.add(residual_module(data=model, filters=filters[i + 1], stride=stride, reduce=True, bn_eps=bn_eps, bn_momentum=bn_momentum))

        for j in range(stages[i] - 1):
            model.add(residual_module(data=model, filters=filters[i + 1], stride=(1, 1), bn_eps=bn_eps, bn_momentum=bn_momentum))
        
    
    model.add(BatchNormalization(epsilon=bn_eps, momentum=bn_momentum))
    model.add(ReLU())
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(classes,activation="softmax", kernel_regularizer=l2(reg)))
    
    return model


def load_image_and_label(image_path, target_size=(32,32)):
    """
    * Load images and one-hot labels .

    * Parameters: reame
        - image_paths: image path.
        - target_size: target width,height.

    * Return:
        - Numpy array --> image, label.
    """
    CINIC_MEAN_RGB = np.array([0.47889522, 0.47227842, 0.43047404])
    CINIC_10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "forg", "horse", "ship", "truck"]

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.float32)
    image -= CINIC_MEAN_RGB #mean normalize
    image = tf.image.resize(image, target_size)

    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CINIC_10_CLASSES) #one-hot encoding
    label = tf.dtypes.cast(label, tf.float32)

    return image, label

batch_size = 128
BUFFER_SIZE = 1024

def prepare_dataset(data_pattern, shuffle=True):
    """
    * Create tf.data.Dataset instance of images and labels.

    * Parameters: reame
        - data_pattern: data_pattern.
        - shuffle: shuffle data.

    * Return:
        - tf.data.Dataset instance.
    """


    dataset = (tf.data.Dataset.list_files(data_pattern).map(load_image_and_label, num_parallel_calls=AUTOTUNE).batch(batch_size))

    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)
    
    return dataset.prefetch(batch_size)


def download_prepare_CINIC1_dataset():

    DATASET_URL = ('https://datashare.is.ed.ac.uk/bitstream/handle/''10283/3192/CINIC-10.tar.gz?''sequence=4&isAllowed=y')

    DATA_NAME = 'cinic10'
    FILE_EXTENSION = 'tar.gz'

    FILE_NAME = '.'.join([DATA_NAME, FILE_EXTENSION])

    downloaded_file_location = get_file(origin=DATASET_URL,fname=FILE_NAME,extract=False)

    data_directory, _ = (downloaded_file_location.rsplit(os.path.sep, maxsplit=1))

    data_directory = os.path.sep.join([data_directory, DATA_NAME])
    
    tar = tarfile.open(downloaded_file_location)

    if not os.path.exists(data_directory):
        tar.extractall(data_directory)

    train_pattern = os.path.sep.join([data_directory, "train/*/*.png"])
    test_pattern = os.path.sep.join([data_directory, "test/*/*.png"])
    valid_pattern = os.path.sep.join([data_directory, "valid/*/*.png"])

    train_dataset = prepare_dataset(train_pattern, shuffle=True)
    test_dataset = prepare_dataset(test_pattern)
    vaid_dataset = prepare_dataset(valid_pattern)


    return train_dataset, test_dataset, vaid_dataset

def main():

    train_dataset, test_dataset, valid_dataset = download_prepare_CINIC1_dataset()
    
    model = build_resnet(input_shape=(32, 32, 3), classes=10, stages=(9,9,9), filters=(64,64,128,256),reg=5e-3)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    model_checkpoint_callback = ModelCheckpoint(filepath="./model.{epoch:02d}-{val_accuracy:.2f}.hdf5",save_best_only=False, monitor="val_accuracy")

    epochs = 1
    model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs,callbacks=[model_checkpoint_callback])

    #load model for inference
    model = load_model('model.38-0.72.hdf5')
    result = model.evaluate(test_dataset)
    print(f'Test accuracy: {result[1]}')
    # import matplotlib.pyplot as plt

    # ### To visualize the images
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_dataset.take(1):
    #     for i in range(batch_size):

    #         plt.imshow(images[i].numpy().astype("uint8"))

    #     plt.axis("off")

    # # Plotting the images
    # plt.show()

#run script
if __name__ == "__main__":
    main()