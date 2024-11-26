""" LabelEncoder class for encoding labels to integers and vice versa.
"""

from typing import Any, List
import numpy as np

class BasicLabelEncoder:
    def __init__(self):
        """ Constructor for LabelEncoder class
        """
        self.label_to_int : List[dict[Any, int]] = []
        self.int_to_label : List[dict[int, Any]] = []
        self.num_classes : int = 0



    def fit(self, y : np.ndarray, continuous : List[int] = []) -> np.ndarray:
        """ Fit the encoder to the labels in y

        Args:
            y (numpy.ndarray): Labels to encode
            continuous (List[int]): List of indices/columns of continuous features. If present, the encoder will not encode these features.
            
        Returns:
            numpy.ndarray: Unique labels in y
        """
        
        # all_unique_labels = []
        self.label_to_int = []
        self.int_to_label = []
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 1: Iterate over each column, and check if it is continuous
        for i in range(y.shape[1]):
            self.label_to_int.append({})
            self.int_to_label.append({})

            # 2: If it is continuous, skip encoding
            if i in continuous:
                continue

            # 3: If it is not continuous, encode the column
            unique_labels = np.unique(y[:, i])
            for j, label in enumerate(unique_labels):
                self.label_to_int[i][label] = int(j)
                self.int_to_label[i][int(j)] = label
            
            # 4: Append unique labels to all_unique_labels
            # all_unique_labels.append(unique_labels)
            
        # 5: Collect all unique labels
        # unique_labels = np.unique(np.concatenate(all_unique_labels))
        
        # 5.5: Set the number of classes
        # self.num_classes = len(unique_labels)
        
        # 6: Return unique labels in y
        # return unique_labels



    def transform(self, y : np.ndarray) -> np.ndarray:
        """ Transform labels to integers

        Args:
            y (numpy.ndarray): Labels to encode

        Returns:
            numpy.ndarray: The transformed array
        """
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        new = np.ndarray(y.shape)
        
        # For each column, transform the labels to integers.
        for i in range(y.shape[1]):
            for j, label in enumerate(y[:, i]):
                # If the value doesn't exist in the dictionary, then assume continuous
                if label in self.label_to_int[i]:
                    new[j, i] = self.label_to_int[i][label]
                else:
                    new[j, i] = label
        
        return np.array(new)



    def inverse_transform(self, y : np.ndarray) -> np.ndarray:
        """ Transform integers to labels

        Args:
            y (numpy.ndarray): Encoded labels

        Returns:
            numpy.ndarray: The transformed array
        """
        
        new = np.ndarray(y.shape)
        
        # For each column, transform the integers to labels.
        for i in range(y.shape[1]):
            for j, label in enumerate(y[:, i]):
                # If the value doesn't exist in the dictionary, then assume continuous
                if label in self.int_to_label[i]:
                    new[j, i] = self.int_to_label[i][label]
                else:
                    new[j, i] = label
        
        return np.array(new)



    def fit_transform(self, y : np.ndarray, continuous : List[int] = []) -> np.ndarray:
        """ Fit the encoder to the labels in y and transform them to integers

        Args:
            y (numpy.ndarray): Labels to encode

        Returns:
            numpy.ndarray: The transformed array
        """
        self.fit(y, continuous)
        return self.transform(y)
    