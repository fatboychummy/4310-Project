""" Naive Bayes Classifier
"""

import numpy as np

class NaiveBayes:
    def __init__(self):
        """ Constructor for NaiveBayes class
        """



    def fit(self, X : np.ndarray, y : np.ndarray):
        """ Fit the Naive Bayes classifier to the data

        Args:
            X (np.ndarray): The training data
            y (np.ndarray): The labels
        """
        y = y.ravel()
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        
        # Calculate mean, variance, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self._classes):
            X_c = X[y == c] # Get the data points with the current class
            self._mean[idx, :] = X_c.mean(axis=0)
            self._variance[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)



    def predict(self, X : np.ndarray) -> np.ndarray:
        """ Predict the class for the input data

        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted classes
        """
        # Predict the class for each data point
        y_pred = [self._predict(x) for x in X]
        
        # Return the predicted classes
        return np.array(y_pred)



    def _predict(self, x : np.ndarray) -> int:
        """ Predict the class for the input data

        Args:
            x (np.ndarray): The input data

        Returns:
            int: The predicted class
        """
        posteriors = []
        
        # Calculate the posterior for each class
        for idx, _ in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Return the class with the highest posterior
        return self._classes[np.argmax(posteriors)]



    def _pdf(self, class_idx : int, x : np.ndarray) -> np.ndarray:
        """ Calculate the probability density function for the given class

        Args:
            class_idx (int): The index of the class
            x (np.ndarray): The input data
            
        Returns:
            np.ndarray: The probability density function
        """
        mean = self._mean[class_idx]
        variance = self._variance[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
