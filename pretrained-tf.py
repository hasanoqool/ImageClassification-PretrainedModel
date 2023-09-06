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
    return model


def load_resize_image():
    """
    * Load the image to classify. ResNetV2152 takes 224x224x3 image.
    
    * Return:
        - preprocessed image shape --> (batch, height, width, CH).
    """
    image = load_img("./images/beetle.jpg", target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.
    image = np.expand_dims(image, axis=0) 
    return image


def predict_and_plot(model):
    """
    * Use the model to make predictions on the image then plot result.
    """
    #predict
    image = load_resize_image()  
    predictions = model.predict(image)
    predicted_index = np.argmax(predictions[0], axis=-1)
    #predict

    #download ImageNet labels
    file_name = "ImageNetLabels.txt"
    file_url = ('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    labels_path = get_file(file_name, file_url)

    with open (labels_path) as f:
        imagenet_labels = np.array(f.read().splitlines())

    predicted_class = imagenet_labels[predicted_index]
    #download ImageNet labels

    #plot
    plt.figure()
    plt.title(f"Label: {predicted_class}")
    original = load_img("./images/beetle.jpg")
    original = img_to_array(original)
    plt.imshow(original / 255.)
    plt.show()
    #plot


if __name__ == "__main__":
    model = model()
    predict_and_plot(model)