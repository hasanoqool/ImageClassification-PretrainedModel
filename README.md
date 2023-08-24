# ImageClassification-TransferLearning
Classification and Transfer Learning for images

# Overview
* This repo consists of 6 main topics:
    * Creating a binary classifier to detect smiles (positive, negative)
    * Creating a multi-class classifier to play Rock Paper Scissors
    * Creating a multi-label classifier to label watches
    * Implementing ResNet from scratch
    * Classifying images with a pre-trained network using the Keras API & TensorFlow Hub
    * Using data augmentation to improve performance with the Keras API, tf.data and tf.image APIs

#
## Running detect_smiles.py
* Train a smile classifier from scratch on the <b>SMILEs dataset</b>.

* <b>Sample Images</b> --> up (Negative) | down (Positive)

    ![Negative](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/negative.png)

    ![Positive](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/positive.png)

* <b>Model Evaluation</b>:

| train_loss  |  train_accuracy |
| ------------- | ------------- |
|  0.5435 |  0.8920 |

| val_loss  |  val_accuracy |
| ------------- | ------------- |
|  0.2501 |  0.9193 |

| test_loss  |  test_accuracy |
| ------------- | ------------- |
|  0.2077 |  0.9225 |
#