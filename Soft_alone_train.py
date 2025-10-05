import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import pickle


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
        
    def train(self, X_train, y_train, epochs, batch_size, X_val, y_val):
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
        training_loss_history = []  # <---Initialization of loss_history
        training_accuracy_history = []
        validation_loss_history = []
        validation_accuracy_history = []
        
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
            training_loss_history.append(avg_loss)  # <---Population of loss history
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

            # Validation step
            # 1. Validation Forward Pass
            y_pred_val = self.forward_pass(X_val)

            # 2. Compute Validation Loss
            val_loss = self.compute_loss(y_val, y_pred_val)
            validation_loss_history.append(val_loss)

            # 3. Compute Validation Accuracy
            if self.loss_function == 'binaryCrossentropy':
                y_pred_labels_val = (y_pred_val >= 0.5).astype(int).flatten()
                y_true_labels_val = y_val.flatten()
            elif self.loss_function == 'categoricalCrossentropy':
                # The shape of y_pred_val : (2, N_val)
                # Assignment for y_pred_labels_val
                # Get predicted class labels
                y_pred_labels_val = np.argmax(y_pred_val, axis=0)
                y_true_labels_val = np.argmax(y_val, axis=0)
            
            # Now we have y_pred_labels_val, y_true_labels_val
            correct_predictions_val = np.sum(y_pred_labels_val == y_true_labels_val)
            val_accuracy = correct_predictions_val / X_val.shape[1]
            validation_accuracy_history.append(val_accuracy)

            print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val_loss: {val_loss:.4f}, Train Acc: {accuracy*100:.2f}%, Val Acc: {val_accuracy*100:.2f}%")

        print("\nTraining complete.")
        return training_loss_history, training_accuracy_history, validation_loss_history, validation_accuracy_history

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
    X_test = valid_df.drop('diagnosis', axis=1)  # Unsealed DataFrame
    y_test = valid_df['diagnosis']

    # --- Step 1: Label Encoding (M -> 1, B -> 0)
    y_train = y_train.apply(lambda x: 1 if x == 'M' else 0)
    y_test = y_test.apply(lambda x: 1 if x == 'M' else 0)

    # --- Step 2: Data Standardization (Manual Implementation)
    # Convert pandas DataFrames to NumPy arrays for calculation
    X_train_np = X_train.values
    X_test_np = X_test.values

    # Calculate mean and standard deviation from the TRAINING data ONLY
    mean = np.mean(X_train.values, axis=0)
    std = np.std(X_train.values, axis=0)

    # Avoid division by zero for features with zero standard deviation
    std[std == 0] = 1e-15

    # Standardize both training and test data
    X_train_scaled = (X_train_np - mean) / std
    X_test_scaled = (X_test_np - mean) / std

    # Step 3: Reshaping for neural Network keep Binary Labels
    # Transpose to feature to (num_features, num_samples)
    X_train_final = X_train_scaled.T
    # test_final = X_test_scaled.T

    # Reshape the binary labels (0 or 1) to (1, num_sample)
    # Use .astype(int) to ensure it's a Numpy array, and .astype(int) for safety.
    y_train_final = y_train.values.astype(int).reshape(1, -1)
    y_test_final = y_test.values.astype(int).reshape(1, -1)
    
    # Return values : X_scaled(F, N), y_binary(1, N), 
    # X_unscaled(DF), y_binary(1, N), stats
    return X_train_final, y_train_final, X_test, y_test_final, mean, std


def plot_metrics(train_loss, train_accuracy, val_loss, val_accuracy):
    epochs = range(len(train_loss))

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label="Training Loss")
    plt.plot(epochs, val_loss, 'r', label="Validation Loss")
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plt accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'm', label='Validation accuracy')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train a custom neural network with command line arguments.")
    parser.add_argument("--layers", nargs='+', type=int, default=[32, 64], 
                        help="Sizes of the hidden layers. E.G., --layer 24 24.")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs for train for.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of minibatches.Use 1 for SGD, total samples for full batch")
    parser.add_argument("--loss", type=str, default="binaryCrossentropy",
                        choices=["binaryCrossentropy", "categoricalCrossentropy"],
                        help="Loss function to use. Note : CategoricalCrossentropy is for demonstration")
    
    args = parser.parse_args()

    # Validation check for the hidden layers concerning( at least two)
    if len(args.layers) < 2:
        raise ValueError(
            f"The model must have at least two hidden layers."
            f"You provided {len(args.layers)} layer size(s): {args.layers}. "
            f"Please provide at least two sizes, e.g., --layers 32 64."
        )

    # Load and preprocess data capturing the mean and std
    X_train, y_train, X_test, y_test_final, mean, std = preprocess_data()

    # Calculate scaled X_test, similar to how it's done before evaluation
    std[std == 0] = 1e-15
    X_val_final = ((X_test.values - mean) / std).T
    
    # Step 1: Adjust Network Architecture based on Loss function
    num_features = X_train.shape[0]
    loss_function = args.loss

    if loss_function == 'categoricalCrossentropy':
        # For categorical, output layer must be size 2 (for B and M)
        num_classes = 2
        layer_sizes = [num_features] + args.layers + [num_classes]
        
        # 1. FLATTEN: Convert the (1, N) binary label array 
        # to a 1D array (N,) for indexing
        y_train_labels = y_train.flatten()
        y_test_labels = y_test_final.flatten()

        # 2. One-Hot Encode since the array is correctly shaped)
        # One-Hot Encode fro training labels
        y_train_one_hot = np.zeros((num_classes, y_train_labels.shape[0]))  
        y_train_one_hot[y_train_labels, np.arange(y_train_labels.shape[0])] = 1

        # One-hot encode the validation/test labels
        y_test_one_hot = np.zeros((num_classes, y_test_labels.shape[0]))
        y_test_one_hot[y_test_labels, np.arange(y_test_labels.shape[0])] = 1

        # Reassign y_train and y_test to one-hot encoded versions
        y_train = y_train_one_hot
        y_test_final = y_test_one_hot  # Update the variable used for evaluation
    else:
        # For binary cross-entropy, output layer is size 1
        layer_sizes = [num_features] + args.layers + [1]

    # All other hyperparameters
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    
    # print the configuration to confirm
    print("Network Configuration")
    print(f" Layer Sizes: {layer_sizes}")
    print(f" Learning Rate: {learning_rate}")
    print(f" Epochs: {epochs}")
    print(f" Batch Size: {batch_size}")
    print(f" Loss Function: {loss_function}")
    print("- * 30")

    # Step 2: Create, Train and evaluate in this order
    model = NeuralNetwork(layer_sizes=layer_sizes, 
                          learning_rate=learning_rate,
                          loss_function=loss_function)
    loss_history, accuracy_history, val_loss_history, val_accuracy_history = model.train(
        X_train, y_train, args.epochs, args.batch_size,
        X_val_final, y_test_final)

    # Save the trained model and preprocessing parameters in one file
    model_state = {
        'model': model,
        'mean': mean,
        'std': std,
        'layer_sizes': layer_sizes, 
        'loss_function': loss_function
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_state, f)
    print("Model and preprocessing parameters saved to model.pkl")

    # Now evaluate the trained model on the correctly standardized test data
    # Standardize the test data using the saved mean/std
    std[std == 0] = 1e-15
    X_test_scaled = (X_test.values - mean) / std
    X_test_final = X_test_scaled.T

    accuracy = model.evaluate(X_test_final, y_test_final)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plot_metrics(loss_history, accuracy_history, val_loss_history, val_accuracy_history)

    # Good combination of hyper parameters1:
    # Network Configuration
    # Layer Sizes: [31, 64, 64, 1]
    # Learning Rate: 0.02
    # Epochs: 70
    # Batch Size: 16
    # Loss Function: binaryCrossentropy

    # Good combination of hyper paraters2
    # Network Configuration
    # Layer Sizes: [31, 64, 64, 2]
    # Learning Rate: 0.02
    # Epochs: 50
    # Batch Size: 16
    # Loss Function: categoricalCrossentropy


if __name__ == "__main__":
    main()
