"""
CONTACT:
Hasan Al-Oqool : https://www.linkedin.com/in/hasanoqool/
"""

#===============Import the required libraries===============
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import *
from keras.preprocessing.image import *
#===============Import the required libraries===============


def load_resize_image():
    """
    * Load the image to classify. Inception takes 299x299x3 image.
    
    * Return:
        - preprocessed image shape --> (batch, height, width, CH).
    """
    image = load_img("./images/dog.png", target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)    
    return image

model = InceptionV3(weights="imagenet")

def predict_and_plot(model):
    """
    * Use the model to make predictions on the image and decode predictions,
      and examine the top 5 predictions along with their probability.
    """
    #predict
    image = load_resize_image()  
    predictions = model.predict(image)
    predictions_matrix = (imagenet_utils.decode_predictions(predictions))

    for i in range(5):
        _, label, probability = predictions_matrix[0][i]
        print(f"{i + 1}. {label}: {probability * 100:.3f}%")
    #predict

    #plot
    _, label, probability = predictions_matrix[0][0]
    plt.figure()
    plt.title(f"Label: {label} || {probability * 100:.3f}%")
    original = load_img("./images/dog.png")
    original = img_to_array(original)
    plt.imshow(original / 255.)
    plt.show()
    #plot


if __name__ == "__main__":
    predict_and_plot(model)