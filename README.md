# ImageClassification-PretrainedModel
Classification, Pretrained models, and Augmentation for images.

# Overview
* The steps of this project are the following:
    * Train a binary classifier to classify face reactions (positive, negative)
    * Train classifier to play RPS (Multi-Class)
    * Train classifier to classify multi watches (Multi-Label)
    * Implementing ResNet from scratch
    * Classify images using a pre-trained network using the Keras API & TensorFlow Hub
    * Apply Data augmentation for improving performance with the Keras API, tf.data and tf.image APIs

#
## Running detect_smiles.py
* Train a binary classifier to classify face reactions (positive, negative) on the <b>SMILEs dataset</b>.
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
* Train classifier to play <b>RPS</b>.
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
* Train classifier to label <b>watches</b>.
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
* Dataset LINK : https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?/equence=4&isAllowed=y

* Test accuracy : <b>70%</b>
#
## Running pretrained-keras.py
* Classify images using a <b>pretrained neural network (InceptionV3)<b> using the Keras API.

    | Label | Class | Probability |
    | :---         |     :---:      |          ---: |
    | 1   | pug     | 89.632%    | 
    | 2     | Brabancon_griffon       | 0.339%    |
    | 3     | bull_mastiff       | 0.138%     |
    | 4     | French_bulldog       | 0.134%      |
    | 5     | Boston_bull       | 0.114%     |

* Sample Image Result. 

    ![dog](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/dog_result.png)
#
## Running pretrained-tf.py
* Classify images using a <b>pretrained neural network (ResNetV2152)<b> using the TF Hub.

* Sample Image Result. 

    ![car](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/beetle_result.png)
#
## aug_keras.py
* Apply <b>Data augmentation</b> for improving performance with the Keras API.
* Dataset LINK : https://data.caltech.edu/records/mzrjq-6wc02

* Before <b>Augmentation</b>: 
     Test accuracy: 0.62

    ![without](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/keras_without_aug.png)

* After <b>Augmentation</b>:
     Test accuracy: 0.66

    ![with](https://github.com/hasanoqool/ImageClassification-TransferLearning/blob/main/images/keras_augmented.png)    
 #
## Contact
* Reach me out here: https://www.linkedin.com/in/hasanoqool/
