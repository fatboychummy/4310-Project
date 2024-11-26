
from MLPNN.util import sigmoid, sigmoid_derivative
import numpy as np

class Network(object):
    def __init__(self, sizes : list[int]):
        """ Constructs a neural network with the given sizes

        Args:
            sizes (list[int]): The number of neurons in each layer
        """

        # Set the number of layers
        self.num_layers = len(sizes)
        
        # Set the number of neurons in each layer
        self.sizes = sizes
        
        # Initialize the weights and biases
        self.init_weights()
        
        # Initialize the biases
        self.init_biases()
        
        # The learning rate decay. Values less than 1 reduce the learning rate each epoch (i.e: 0.95 = 5% reduction). Values greater than 1 increase the learning rate each epoch.
        self.learn_rate_decay = 1



    def init_biases(self, random_seed : int = None) -> None:
        """ Initialize the biases of the neural network.

        Args:
            random_seed (int, optional): The random seed to use. If not provided, does not change the seed.
        """

        if random_seed:
            np.random.seed(random_seed)
        
        # Initialize the biases
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]



    def init_weights(self, random_seed : int = None) -> None:
        """ Initialize the weights of the neural network.

        Args:
            random_seed (int, optional): The random seed to use. If not provided, does not change the seed.
        """

        if random_seed:
            np.random.seed(random_seed)
        
        # Initialize the weights
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]



    def feed_forward(self, a : np.ndarray) -> np.ndarray:
        """ Feed forward an input through the network

        Args:
            a (np.ndarray): The input data

        Returns:
            np.ndarray: The output data
        """

        # For each layer, calculate the activation
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        
        return a



    def stochastic_gradient_descent(self, training_data : list[tuple[np.ndarray, np.ndarray]], epochs : int, mini_batch_size : int, learning_rate : float, test_data : list[tuple[np.ndarray, np.ndarray]] = None) -> None:
        """ Perform stochastic gradient descent to train the network

        Args:
            training_data (list[tuple[np.ndarray, np.ndarray]]): The training data
            epochs (int): The number of epochs to train
            mini_batch_size (int): The size of the mini-batches
            learning_rate (float): The learning rate
            test_data (list[tuple[np.ndarray, np.ndarray]], optional): The test data to evaluate the network
        """

        # If test data is provided, set the number of test samples
        if test_data:
            n_test = len(test_data)
        
        # Set the number of training samples
        n = len(training_data)
        
        # For each epoch
        for j in range(epochs):
            # Shuffle the training data, so that the mini-batches are random
            np.random.shuffle(training_data)
            
            # Create the mini-batches
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            # For each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            # Print the progress, if test data is provided
            if test_data:
                print(f"Epoch {j}: {self.evaluate_accuracy(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
            
            # Decay the learning rate
            learning_rate *= self.learn_rate_decay



    def update_mini_batch(self, mini_batch : list[tuple[np.ndarray, np.ndarray]], learning_rate : float) -> None:
        """ Updates the network's weights/biases via gradient descent+backprop, on a single mini-batch

        Args:
            mini_batch (list[tuple[np.ndarray, np.ndarray]]): The mini-batch of training data
            learning_rate (float): The learning rate
        """

        # Initialize the gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # For each training example in the mini-batch
        for x, y in mini_batch:
            # Calculate the gradient for the training example
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update the weights and biases
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]



    def backpropagate(self, x : np.ndarray, y : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Perform backpropagation to calculate the gradient of the cost function

        Args:
            x (np.ndarray): The input data
            y (np.ndarray): The expected output

        Returns:
            tuple[np.ndarray, np.ndarray]: The gradients of the cost function
        """

        # Initialize the gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feed forward
        activation = x.reshape(-1, 1) # Ensure column vector
        activations = [activation]
        zs = []
        
        # For each layer, calculate the activation and z vecs
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # z = w * a + b
            zs.append(z)
            activation = sigmoid(z) # The important part
            activations.append(activation) # Also important, otherwise our "brain" has a bit of amnesia
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1]) # Calculate the error
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # The gradient of the cost function
        
        # For each layer, calculate the gradient
        for l in range(2, self.num_layers):
            z = zs[-l] # Select the z vector
            sd = sigmoid_derivative(z) # Calculate the sigmoid derivative
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sd # Calculate the error
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose()) # The gradient of the cost function
        
        return nabla_b, nabla_w # AND WE'RE DONE



    def cost_derivative(self, output_activations : np.ndarray, y : np.ndarray) -> np.ndarray:
        """ Partial derivatives C_x / a for output activations

        Args:
            output_activations (np.ndarray): The output activations
            y (np.ndarray): The expected output

        Returns:
            np.ndarray: The derivative of the cost function
        """

        return output_activations - y



    def evaluate_accuracy(self, test_data : list[tuple[np.ndarray, np.ndarray]]) -> int:
        """ Evaluate the network on the test data

        Args:
            test_data (list[tuple[np.ndarray, np.ndarray]]): The test data

        Returns:
            int: The number of correct predictions
        """

        # Get the predictions
        test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for (x, y) in test_data]
        
        # Return the number of correct predictions
        return sum(int(x == y) for (x, y) in test_results)
    
    
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        """ Predict the class for the input data

        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted classes
        """

        # Predict the class for each data point
        y_pred = [np.argmax(self.feed_forward(x)) for x in X]
        
        # Return the predicted classes
        return np.array(y_pred)



    def bruh():
        print("bruh")
    