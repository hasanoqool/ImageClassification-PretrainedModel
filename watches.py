"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import os
import pathlib
from csv import DictReader
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing.image import *
#===============Import the required libraries===============

#for reproducibility:
SEED = 999
np.random.seed(SEED)

def build_network(width, height, CH, classes):
    """
    * Build ConvNets.

    * Parameters: 
    - width: image width (Pixels).
    - height: image height (Pixels).
    - depth: the number of image's color channels.
    - classes: number of classes in images dataset.

    * Return:
        - Sequential model.
    """

    model = Sequential()    
    model.add(Input(shape=(width,height,CH)))

    #block1
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())

    #block2
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #block3
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'))
    model.add(BatchNormalization())

    #block4
    model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #block5
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation="sigmoid"))
    return model


def load_images_and_labels(image_paths, styles, target_size):
    """
    * load all images and labels (gender, usage).

    * Parameters: 
    - image_paths: image path folder.
    - styles: dictionary of metadata.
    - target_size: tuple --> height, width.

    * Return:
        - np array --> image,label.
    """

    images = []
    labels = []

    for image_path in image_paths:
        #image path to then to np array
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)

        #extract image id
        image_id = image_path.split(os.path.sep)[-1][:-4]

        #extract image style
        image_style = styles[image_id]
        
        #image label (multi-label) --> gender, usage
        label = (image_style['gender'], image_style['usage'])

        images.append(image)
        labels.append(label)
    
    return np.array(images), np.array(labels)


def path_style():
    """
    * define the paths to the images and the styles.csv metadata file.

    * Return:
        - np array --> styles_path, image_paths.
    """

    base_path = (pathlib.Path.home() / 'Desktop' /'datasets' /'watches')
    styles_path = str(base_path / 'styles.csv')
    images_path_pattern = str(base_path / 'images/*.jpg')
    image_paths = glob.glob(images_path_pattern)
    
    return styles_path, image_paths


def extract_watches():
    '''
    * Keep only the Watches images for Casual, Smart Casual, and Formal usage, 
      suited to Men and Women.

    * Return:
        - np array --> image_paths and STYLES for each watch .
    '''

    styles_path, image_paths = path_style()

    with open (styles_path, 'r') as f:
        dict_reader = DictReader(f)
        STYLES = [*dict_reader]

        article_type = "Watches"

        genders = {'Men', 'Women'}

        usages = {'Casual', 'Smart Casual', 'Formal'}

        STYLES ={
        style['id']: style
        for style in STYLES
        if (style['articleType'] == article_type and style['gender'] in genders and style['usage'] in usages)       
        }
    
    image_paths = [*filter(lambda p: p.split(os.path.sep)[-1][:-4] in STYLES.keys(), image_paths)]

    return image_paths, STYLES


def watch_label():
    '''
    * Normalize the images and multi-hot encode the labels

    * Return:
        - np array --> image features(x), labels(y).
    
    '''

    image_paths, STYLES = extract_watches()
    image, label = load_images_and_labels(image_paths, STYLES, target_size=(64, 64))


    image = image.astype('float') / 255.
    mlb = MultiLabelBinarizer()
    label = mlb.fit_transform(label)
    classes = mlb.classes_

    return image, label, classes


def train_valid_test():
    '''
    * Prepare train, test, and validation subsets of datasets.

    * Return:
        - tuple of train, valid, test datasets and len of multilabel classes.
    '''

    image, label, classes = watch_label()

    X_train, X_test, y_train, y_test = train_test_split(image, label, test_size=0.2, random_state=SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size=0.2, random_state=SEED)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, classes


def build_compile(model):
    """ 
    * model compile

    * Parameters: 
    - model: model architecture.
    """

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def main():

    #data
    X_train, y_train, X_valid, y_valid, X_test, y_test, classes = train_valid_test()

    #fit
    EPOCHS = 20
    BATCH_SIZE = 64

    model = build_network(width=64, height=64, CH=3, classes=len(classes))
    build_compile(model)

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid),batch_size=BATCH_SIZE, epochs=EPOCHS)

    #evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f"Test loss: {test_loss} and Test Accuracy {test_accuracy}.")


#run script
if __name__ == "__main__":
    main()