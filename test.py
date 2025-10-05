import pandas as pd
import pickle
import argparse 
import numpy as np 


class NeuralNetwork:
    """A simple fully connected neural network."""
    
    def __init__(self, layer_sizes, learning_rate, loss_function):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.weights = []
        self.biases = []
        self.activations = []
        
        # Initialize weights and biases
        # This loop creates the weights and biases for each layer transition
        for i in range(len(layer_sizes) - 1):
            # The weights for layer i to i+1 will have dimensions
            # (size_of_layer_i+1, size_of_layer_i)
            # We use a small random number to initialize to prevent all neurons
            # from learning the same things (symmetry breaking)
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def relu(self, z):  # Rectified linear Unit
        return np.maximum(0, z)
    
    def softmax(self, z):
        """
        Softmax activation function.
        Used for the output in multi-class classification.
        """
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        
    def forward_pass(self, X):
        """
        Perform a forward pass through the neural network.

        Args:
            X: The input data, a numpy array of shape(NR of features, NR of samples)
            Dataset has 31 features, X shape will be (31, num_samples).
        Returns :
                 The final outpuut, a numpy array.
        """
        # Initialize the activations list for this pass
        self.activations = []

        # The first activation is the input itself
        A_prev = X
        self.activations.append(A_prev)

        # Loop through hidden layers
        for i in range(len(self.weights) - 1):
            # Retreive the weights and biases for the current layer
            W = self.weights[i]
            b = self.biases[i]

            # Calculate the weighted sum of inputs (Z)
            # Z = W * A_prev + b
            Z = np.dot(W, A_prev) + b

            # Apply the ReLu(Rectified linear unit) activation function
            A_curr = self.relu(Z)

            # Store the current activation and update  A_prev for the next loop
            self.activations.append(A_curr)
            A_prev = A_curr

        # For the output layer, choose activation based on loss function
        W_out = self.weights[-1]
        b_out = self.biases[-1]
        Z_out = np.dot(W_out, A_prev) + b_out
        
        if self.loss_function == 'binaryCrossentropy':
            A_out = self.sigmoid(Z_out)
        elif self.loss_function == 'categoricalCrossentropy':
            A_out = self.softmax(Z_out)
        else:
            raise ValueError("Unsupported loss function.")
        
        self.activations.append(A_out)
        return A_out

    def compute_loss(self, y_true, y_pred):
        # Number of samples
        m = y_true.shape[1]

        # Avoid log(0) which can lead to errors.
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if self.loss_function == 'binaryCrossentropy':
            loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_function == 'categoricalCrossentropy':
            # For binary case, this is equivalent to binary cross-entropy.
            # In a true multi-class scenario, y_true would be one-hot encoded.
            loss = - (1/m) * np.sum(y_true * np.log(y_pred))
        else:
            raise ValueError("Unsupported loss function.")

        return loss
        
    def backward_pass(self, X, y_true, y_pred):
        """
        Performs the backward pass to compute the gradients.

        Args:
            X: input data, a numpy array of shape(Number_of_features, Number_of_samples)
            y_true: The true labels, a numpy of shape (1, number_of_samples).
            y_pred: The predicted probabilities, shape (1, number_of_samples)

            Returns:
                    A tuple of (gradient_weights, gradients_biases).
        """
        m = X.shape[1]  # Number of samples
        gradients_weights = [None] * len(self.weights)
        gradients_biases = [None] * len(self.biases)

        # Step 1: Backpropagate from the ouput layer
        if self.loss_function == 'binaryCrossentropy':
            dZ_out = y_pred - y_true
        elif self.loss_function == 'categoricalCrossentropy':
            # For softmax and categorical cross entropy returns simple derivative.
            dZ_out = y_pred - y_true
        else:
            raise ValueError("Unsupported loss function.")

        A_prev = self.activations[-2]  # The activation of the last hidden layer
        dW_out = (1/m) * np.dot(dZ_out, A_prev.T)
        db_out = (1/m) * np.sum(dZ_out, axis=1, keepdims=True)

        gradients_weights[-1] = dW_out
        gradients_biases[-1] = db_out

        # Step 2: Backpropagate through the hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            W_next = self.weights[i+1]
            dZ_next = dZ_out

            # Derivative of the ReLU activation function
            d_relu = np.where(self.activations[i+1] > 0, 1, 0)
            dZ_curr = np.dot(W_next.T, dZ_next) * d_relu
        
            # Calculate gradients for the current layer's weights and biases
            A_prev = self.activations[i]  # Activation of the previous layer
            dW_curr = (1/m) * np.dot(dZ_curr, A_prev.T)
            db_curr = (1/m) * np.sum(dZ_curr, axis=1, keepdims=True)
        
            # Store the gradients
            gradients_weights[i] = dW_curr
            gradients_biases[i] = db_curr
        
            # Update dZ_out for the next iteration
            dZ_out = dZ_curr
        
        return (gradients_weights, gradients_biases)

    def update_params(self, gradients):
        """
        Updates the weights and biases using the calculated gradients
        and learning rate

        Args:
            Gradients are the tuple of (gradients_weights, gradients_biases)
            from the backward pass.
        """
        gradients_weights, gradients_biases = gradients

        # Loop through each layer to update its weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_weights[i]
            self.biases[i] -= self.learning_rate * gradients_biases[i]
        
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the neural network using the Gradient descent.

        Args:
            X_train: The training data,
                    a numpy array of shape (num_features, num_samples.)
            y_train: The true labels of the traaining data,
                     shape(1, num_samples).
            epochs: The number of times to iterate the entire training dataset
        """
        num_samples = X_train.shape[1]

        # The term "Stochastic Gradient descent" in modern ML, refers 
        # Mini Batch Gradient Descent. It processes small and random subset of 
        # data to compute stable gradients.
        print("Training with Stochastic Gradient descent...")

        # Lists to store metrics for visualization
        training_loss_history = []
        training_accuracy_history = []
        
        # Loop for a specified number of epochs
        for epoch in range(epochs):
            # Shuffle the data for each epoch to ensure batches are different
            shuffled_indices = np.random.permutation(num_samples)
            X_shuffled = X_train[:, shuffled_indices]
            y_shuffled = y_train[:, shuffled_indices]

            total_loss = 0
            num_batches = int(np.ceil(num_samples / batch_size))

            for i in range(num_batches):
                # Define the start and end indeces for the current batch
                start = i * batch_size
                end = min((i + 1) * batch_size, num_samples)

                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]

                # 1. Forward Pass
                y_pred = self.forward_pass(X_batch)
                # 2. Compute Loss
                loss = self.compute_loss(y_batch, y_pred)
                total_loss += loss * (end - start)  # Weighted loss by batch size
                # 3. Backward Pass
                gradients = self.backward_pass(X_batch, y_batch, y_pred)
                # 4. Update parameters
                self.update_params(gradients)

            # Compute average loss for the epoch
            avg_loss = total_loss / num_samples
            training_loss_history.append(avg_loss)
            # if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

            # Compute training accuracy for the epoch
            y_pred_train = self.forward_pass(X_train)
            if self.loss_function == 'binaryCrossentropy':
                y_pred_labels = (y_pred_train >= 0.5).astype(int).flatten()
                y_true_labels = y_train.flatten()
            elif self.loss_function == 'categoricalCrossentropy':
                y_pred_labels = np.argmax(y_pred_train, axis=0)
                y_true_labels = np.argmax(y_train, axis=0)

            correct_predictions = np.sum(y_pred_labels == y_true_labels)
            accuracy = correct_predictions / num_samples
            training_accuracy_history.append(accuracy)

            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, Training Accuracy: {accuracy*100:.2f}%")

        print("\nTraining complete.")
        return training_loss_history, training_accuracy_history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on the test data.
        """
        # Perform a forward pass on the test data
        y_pred_probs = self.forward_pass(X_test)

        # Determine the number of samples
        num_samples = y_test.shape[1]

        # Convert predicted probabilities to class labels
        if self.loss_function == 'binaryCrossentropy':
            # For the songle neuron, threshold at 0.5
            y_pred_labels = (y_pred_probs >= 0.5).astype(int).flatten()
            y_true_labels = y_test.flatten()
        elif self.loss_function == 'categoricalCrossentropy':
            # For softmax output, find the index of the max probability
            y_pred_labels = np.argmax(y_pred_probs, axis=0)
            # u_test is already one_hot, so find the index of the 1
            y_true_labels = np.argmax(y_test, axis=0)
        else:
            raise ValueError("Unsupported Loss function.")
        
        # Compare predicted labels to true labels and calculate accuracy
        corrected_predictions = np.sum(y_pred_labels == y_true_labels)
        accuracy = corrected_predictions / num_samples

        return accuracy


def preprocess_data():
    """
    Loads, preprocess the data for the neural network
    """
    # Load the data
    try:
        train_df = pd.read_csv('train_split.csv')
        valid_df = pd.read_csv('val_split.csv')
    except (pd.errors.ParserError, ValueError):
        print("Assuming Csv files have no headers.")
        train_df = pd.read_csv('train_split.csv', header=None)
        valid_df = pd.read_csv('val_split.csv', header=None)

    print("Initial dataframes head:")
    print(train_df.head())

    # There are 31 features and one label, so index 31 should be the label (0-indexed)
    train_df.rename(columns={31: 'diagnosis'}, inplace=True)
    valid_df.rename(columns={31: 'diagnosis'}, inplace=True)

    print("Initial dataframes head:")
    print(train_df.head())

    # Separate features (X) and target (y)
    X_train = train_df.drop('diagnosis', axis=1)
    y_train = train_df['diagnosis']
    X_test = valid_df.drop('diagnosis', axis=1)
    y_test = valid_df['diagnosis']

    # --- Step 1: Label Encoding (M -> 1, B -> 0)
    y_train = y_train.apply(lambda x: 1 if x == 'M' else 0)
    y_test = y_test.apply(lambda x: 1 if x == 'M' else 0)

    # --- Step 2: Data Standardization (Manual Implementation)
    # Convert pandas DataFrames to NumPy arrays for calculation
    X_train_np = X_train.values
    X_test_np = X_test.values

    # Calculate mean and standard deviation from the TRAINING data ONLY
    mean = np.mean(X_train_np, axis=0)
    std = np.std(X_train_np, axis=0)

    # Avoid division by zero for features with zero standard deviation
    std[std == 0] = 1e-15

    # Standardize both training and test data
    X_train_scaled = (X_train_np - mean) / std
    X_test_scaled = (X_test_np - mean) / std

    # --- Step 3: Reshaping for the Neural Network
    # Transpose to (num_features, num_samples)
    X_train_final = X_train_scaled.T
    y_train_final = np.array(y_train).reshape(1, -1)
    X_test_final = X_test_scaled.T
    y_test_final = np.array(y_test).reshape(1, -1)
    
    return X_train_final, y_train_final, X_test_final, y_test_final


def test_model(model_path, data_path):
    """
    Load a saved model and evaluates it on test data.
    """ 
    try:
        with open(model_path, 'rb') as f:
            # Load the entire dictionary from the pickle file
            model_state = pickle.load(f)

        print("Model loaded successfully.")

        # Extract the model object and preprocessing parameters
        model = model_state['model']   
        mean = model_state['mean']
        std = model_state['std']

        # Load and preprocess the test data
        try:
            # Use the data_path variable instead of hardcoding 'val_split.csv'
            test_df = pd.read_csv(data_path)
        except (pd.errors.ParserError, ValueError):
            print("Assuming CSV file has no headers.")
            test_df = pd.read_csv(data_path, header=None)

        print("Initial dataframe head:")
        print(test_df.head())

        # Need to rename the column to 'diagnosis' if test file doen't have it
        test_df.rename(columns={31: 'diagnosis'}, inplace=True)
        print("Initial dataframes head:")
        print(test_df.head())

        # Seperate features (X) and target (y)
        X_test = test_df.drop('diagnosis', axis=1)
        y_test = test_df['diagnosis']
    
        # Data Preprocessing
        # Label Encoding
        y_test = y_test.apply(lambda x: 1 if x == 'M' else 0)

        # standardization using the mean/std from the training data
        X_test_scaled = (X_test.values - mean) / std
        X_test_final = X_test_scaled.T
        y_test_final = np.array(y_test).flatten()  # Start with 1D labels
        loss_function = model_state['loss_function']  # retrieve loss function used for training

        if loss_function == 'categoricalCrossentropy':
            num_classes = model.layer_sizes[-1]
            # One hot encode the test labels
            y_test_one_hot = np.zeros((num_classes, y_test_final.shape[0]))
            # Map binary labels (0 or 1) to the correct row index
            y_test_one_hot[y_test_final, np.arange(y_test_final.shape[0])] = 1
            y_test_final = y_test_one_hot  # Update one-hot version
        else:
            # If binaryCrossentropy, reshape to (1, N) as done previously
            y_test_final = np.array(y_test).reshape(1, -1)

        # Call the evaluate method on the extraxted model object
        accuracy = model.evaluate(X_test_final, y_test_final)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    except FileNotFoundError:
        print(f"Error: Model file '{model_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a pretrained neural network.")
    parser.add_argument("--model", type=str, default="model.pkl",
                        help="Path to the trained model file.")
    parser.add_argument("--data", type=str, default="val_split.csv",
                        help="Path to the unknown or test dataset file.")
    args = parser.parse_args()

    test_model(args.model.strip(), args.data.strip())  # When new data, use the argument.
