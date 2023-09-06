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
* Dataset LINK : <b>https://github.com/hromi/SMILEsmileD/tree/master</b>

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
## Running rock-paper-scissors.py
* Creating a multi-class classifier to play <b>rock paper scissors</b>.
* Dataset LINK : <b>https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors</b>

* <b>Sample Images</b> --> 1st (rock) | 2nd (paper) | 3rd (scissors)

    ![rock](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/rock.png)

    ![paper](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/paper.png)

    ![paper](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/scissors.png)

* <b>Model Evaluation</b>:

    | train_loss  |  train_accuracy |
    | ------------- | ------------- |
    |  0.1130 |  0.9821 |

    | val_loss  |  val_accuracy |
    | ------------- | ------------- |
    |  0.2074 |  0.9457 |

    | test_loss  |  test_accuracy |
    | ------------- | ------------- |
    |  0.2344 |  0.9338 |
#
## Running watches.py
* Creating a multi-label classifier to label <b>watches</b>.
* Dataset LINK : <b>https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small</b>

* <b>Sample Image from test data</b> 

    ![watch](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/watch.png)

* <b>Model Evaluation</b>:

    | train_loss  |  train_accuracy |
    | ------------- | ------------- |
    |  0.0288 |  0.6218 |

    | val_loss  |  val_accuracy |
    | ------------- | ------------- |
    |  0.2812 |  0.9708 |

    | test_loss  |  test_accuracy |
    | ------------- | ------------- |
    |  0.2616 |  0.9766 |

* <b>Conclusion</b>:

    | class  |  probability |
    | ------------- | ------------- |
    |  Casual |  92.31%|
    |  Formal |  3.89% |
    |  Smart Casual |  0.62% |
    |  Men |  0.34% |
    |  Women |  99.91% |
 
    Ground truth labels: [('Casual', 'Women')] --> CORRECT
#
## Running ResNet.py
* Implementing <b>ResNet</b> from <b>scratch</b> and run it on <b>CINIC-10 dataset</b>.
* Dataset LINK : <b>'https://datashare.is.ed.ac.uk/bitstream/handle/''10283/3192/CINIC-10.tar.gz?''sequence=4&isAllowed=y'</b>
* Test accuracy : <b>70%</b>
#
## Running pretrained-keras.py
* Implementing <b>ResNet</b> from <b>scratch</b>.
* Dataset LINK : <b>'https://datashare.is.ed.ac.uk/bitstream/handle/''10283/3192/CINIC-10.tar.gz?''sequence=4&isAllowed=y'</b>
* Test accuracy : <b>70%</b>
 

