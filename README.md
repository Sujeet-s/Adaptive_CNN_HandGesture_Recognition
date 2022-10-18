Hand Gesture Recognition Using an Adapted Convolutional Neural Network with Data Augmentation


# Problem Statement :
The problem taken up in the research paper can be succinctly described as given a dataset of images depicting hand gestures of a certain sign language, develop a robust classification model for recognizing hand gestures, minimizing the effects of factors like different hand sizes, variable illumination, hand postures, and the like.
The research paper has used Peru Sign Language with 6 different classes for testing and training of the model. There were 3750 data points in the dataset, and it was used with 70:30 split for training and testing. Due to unavailability of this in the public domain, we had to use American sign language dataset. The first 6 letters were used for classification. There are 3600 data points used in this case.

# Solution :
In the paper there are two algorithms used : Baseline CNN and Adaptive Convolution Network with data augmentation. CNN has three types of layers: convolutional layers, pooling layers, and fully-connected layers.Convolutional layers help in extracting out features, pooling layers help in calculation reduction by keeping the most important features, and fully connected layers are normal ANN. Bottom layers collect information about low level features. As we move to deeper and deeper layers, the features become more high-level and abstract. In our case max pooling layer will be used. The max-pooling layer reduces the spatial size of representation to reduce the number of parameters and computation in the network.

In case the size of the dataset is too small, the problem of overfitting can arise. In order to solve this problem, we make use of Adaptive convolution network with data augmentation.

To address the problem mentioned above, the ADCNN uses the following:
Data Augmentation: Applying transformations like rotations, left and right shifts, zoom in or zoom out can help in increasing the size of training dataset, thereby reducing the problem of overfitting. In our case, we have applied only vertical and horizontal shifts
Network initialization: Instead of training from completely random initial weights , the use of network initialization can help in reaching a good accuracy more quickly.
L2 Regularization : To decrease complexity of a model, parameters with large weight magnitudes are penalized using L2 norm with hyperparameter λ = 0.0001. These regularizations also add up weights in the error term to ensure this is possible.

There are also various preprocessing steps involved before the final results are achieved. In order to preprocess the data, we first converted the RGB images into grayscale. It helped in reducing the number of channels from 3 to 1 and thereby the number of parameters needed to be optimized. After converting into RGB format, data augmentation was applied. Also normalization of the values of pixels to range between 0 and 1 was done.

Research Gaps: 
In the research paper it was found that the training accuracy was less than the test accuracy. This can be indicative of a problem in splitting of the training data and testing dataset. It can also be indicative of testing done on a smaller set of dataset as the difference is not much and can even out to get more better results upon testing on a larger dataset.

On implementation of the algorithm on a different dataset, it was found that the accuracy obtained on the test dataset was 92% on Baseline CNN and 96% on ADCNN. Therefore, the algorithm works well on different datasets. However, the number of epochs for training the dataset had to be increased because the accuracy wasn't converging on the training dataset at 10 epochs.

The Transfer learning approach could also be used to help in reducing the amount of training required. It involves training of models parameters on a different problem and using it as a starting point for another problem. Some of the famous transfer learning models include VGG (e.g. VGG16 or VGG19) , GoogLeNet (e.g. InceptionV3), Residual Network (e.g. ResNet50).

