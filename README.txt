Project: Neural Network for Breast Cancer Prediction

Overview

This project provides a comprehensive implementation of a fully connected neural network from the ground up, designed to perform binary classification on the Wisconsin Breast Cancer Diagnostic dataset. The primary goal is to classify tumors as Malignant (M) or Benign (B) based on various cellular features.

The codebase is organized into a training script (Soft_alone_train.py) and a testing script (test.py). The core neural network logic, including forward and backward passes, loss computation, and parameter updates, is encapsulated within a single NeuralNetwork class.

Key Features

    Modular Architecture: The network is defined by a flexible layer_sizes parameter, allowing for easy configuration of hidden layers.

    Activation Functions: Utilizes ReLU for hidden layers and Sigmoid for binary classification or Softmax for multi-class classification in the output layer.

    Loss Functions: Supports both binary cross-entropy (for binary classification) and categorical cross-entropy (for multi-class classification) to cater to different problem types.

    Optimization: Employs mini-batch gradient descent for efficient training, using a configurable batch_size and a fixed learning_rate.

    Data Handling: Includes robust data preprocessing steps such as label encoding (M->1, B->0) and standardization using the mean and standard deviation of the training data.

    Serialization: The entire trained model, including weights, biases, and preprocessing parameters, is saved to a single model.pkl file for easy re-use in the testing script.

Getting Started

Prerequisites

    Python 3.x

    The following libraries can be installed via pip:
    Bash

    pip install pandas numpy scikit-learn

    Ensure the following data files are in the same directory:

        train_split.csv

        val_split.csv

        test_split.csv (or adjust the test.py script to use val_split.csv if that is your test set).

Usage

1. Training the Model

To train the neural network, run the Soft_alone_train.py script. You can customize the training process using command-line arguments.
Bash

# Example command for training a binary classifier
python Soft_alone_train.py --layers 64 64 --learning_rate 0.02 --epochs 50 --batch_size 16 --loss binaryCrossentropy

# For a multi-class problem (if applicable)
python Soft_alone_train.py --layers 64 64 --learning_rate 0.02 --epochs 50 --batch_size 16 --loss categoricalCrossentropy

Argument	Description	Default Value
--layers	A list of integers defining the number of neurons in each hidden layer.	64 64
--learning_rate	The learning rate for the gradient descent optimizer.	0.01
--epochs	The number of complete passes over the training dataset.	100
--batch_size	The number of samples per mini-batch.	32
--loss	The loss function to use ('binaryCrossentropy' or 'categoricalCrossentropy').	binaryCrossentropy

2. Testing the Model

After training, a model.pkl file will be generated. To evaluate this saved model on the test data, use the test.py script.
Bash

python test.py --model model.pkl

This script will load the model, perform the necessary preprocessing on the test data, and print the final test accuracy.