# Neural_Network

##Task-1

Ran a multilayer perceptron (feed forward neural network) with two hidden layers and rectified linear nonlinearities on the 
iris dataset using the keras Sequential interface (https://keras.io/getting-started/sequential-model-guide/).
Included code for selecting regularization strength and number of hidden units using GridSearchCV and evaluation
on an independent test-set.

Iris Dataset:
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. 
One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
Attribute Information:
1.	sepal length in cm
2.	sepal width in cm
3.	petal length in cm
4.	petal width in cm
5.	class: -- Iris Setosa -- Iris Versicolour -- Iris Virginica

##Task-2
Trained a multilayer perceptron on the MNIST dataset using the traditional train/test split as given by mnist.load_data in keras. 
Used a separate 10000 samples (from the training set) for model selection and to compute learning curves (accuracy vs epochs, 
not vs n_samples). Compared a “vanilla” model with a model using drop-out. Visualize learning curves for all models. 

MNIST Dataset (Modified National Institute of Standards and Technology database):
It is a large database of handwritten digits that is commonly used for training various image processing systems.
The MNIST database contains 60,000 training images and 10,000 testing images. 
Half of the training set and half of the test set were taken from NIST's training dataset,
while the other half of the training set and the other half of the test set were taken from NIST's testing dataset.

##Task 3:
Trained a convolutional neural network on the SVHN dataset (http://ufldl.stanford.edu/housenumbers/) 
in format 2 (single digit classification). Achieved 85% test-set accuracy with a base model.
Also built a model using batch normalization.

The Street View House Numbers (SVHN) Dataset
SVHN is obtained from house numbers in Google Street View images. 
SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal 
requirement on data preprocessing and formatting. 10 classes, 1 for each digit. Digit '1' has label 1, '9' has 
label 9 and '0' has label 10. 73257 digits for training, 26032 digits for testing, and 531131 additional, 
somewhat less difficult samples, to use as extra training data Comes in two formats:
1.	Original images with character level bounding boxes.
2.	MNIST-like 32-by-32 images centered around a single character (many of the images do contain some distractors at the sides).

##Task-4
Loaded the weights of a pre-trained convolutional neural network included in keras, see https://keras.io/applications/ , 
and used it as feature extraction method to train a linear model or MLP (scikit-learn or keras are fine) on the pets 
dataset (http://www.robots.ox.ac.uk/~vgg/data/pets/). Achieved 70% accuracy. 

Pet's Dataset:
A 37 category pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting. 
All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation.
