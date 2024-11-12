# Handwritten-Digit-Classifier-by-ANN-

# Project Overview
This project is a handwritten digit classifier developed using an Artificial Neural Network (ANN) with TensorFlow and Keras. The model is trained to identify handwritten digits from the popular MNIST dataset, achieving competitive accuracy through pattern recognition.

# Dataset
The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits, with labels from 0 to 9. Each image is 28x28 pixels, which serves as input data for the model.

# Model Architecture
The ANN model architecture includes:

Input Layer: Processes each 28x28 pixel image.
Hidden Layer: Dense layer with 128 neurons using ReLU activation.
Output Layer: Dense layer with 10 neurons using softmax activation for multi-class classification.

# Training
The model is trained on the MNIST dataset using sparse categorical cross-entropy as the loss function and the Adam optimizer. Training is conducted over 10 epochs with a validation split, enabling the model to learn from a portion of the training data while checking performance on unseen data.

# Evaluation
Model performance is evaluated based on accuracy, tested on the separate test dataset. The model demonstrates effective classification for handwritten digits.

# Usage
After training, the model can predict digit labels for new images by analyzing pixel patterns and returning the most likely digit.

# Contributing
Contributions are welcome to improve accuracy, expand functionality, or explore different architectures. Please fork the repository and submit a pull request.

# License
This project is available for personal and educational use.
