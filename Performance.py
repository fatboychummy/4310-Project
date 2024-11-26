from typing import List
import numpy as np

def average(lst : List[float|int]) -> float:
    """ Calculate the average of a list of numbers
    
    Args:
        lst (List[float|int]): The list of numbers
        
    Returns:
        float: The average of the list
    """
    return sum(lst) / len(lst)



def accuracy(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the accuracy of the classifier
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The accuracy of the classifier
    """
    return np.sum(y_true == y_pred) / len(y_true)



def TPR(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the true positive rate of the classifier
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The true positive rate of the classifier
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives



def FPR(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the false positive rate of the classifier
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The false positive rate of the classifier
    """
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    actual_negatives = np.sum(y_true == 0)
    return false_positives / actual_negatives



def TNR(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the true negative rate of the classifier
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The true negative rate of the classifier
    """
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    actual_negatives = np.sum(y_true == 0)
    return true_negatives / actual_negatives



def FNR(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the false negative rate of the classifier
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The false negative rate of the classifier
    """
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    actual_positives = np.sum(y_true == 1)
    return false_negatives / actual_positives



def recall(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the recall of the classifier.
        The recall is the model's ability to correctly identify positive instances,
        with a focus on minimal false negatives.
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The recall of the classifier
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    return true_positives / (true_positives + false_negatives)



def ROC(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the ROC of the classifier.
        The ROC is a graphical representation of the true positive rate (TPR)
        against the false positive rate (FPR).
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The ROC of the classifier
    """
    return TPR(y_true, y_pred) / FPR(y_true, y_pred)



def sensitivity(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the sensitivity of the classifier.
        The sensitivity is the model's ability to correctly identify positive instances.
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The sensitivity of the classifier
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    return true_positives / (true_positives + false_negatives)



def specificity(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Calculate the specificity of the classifier.
        The specificity is the model's ability to correctly identify negative instances.
    
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        
    Returns:
        float: The specificity of the classifier
    """
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_negatives / (true_negatives + false_positives)



class Performance:
    """ Class to store the performance metrics of a model.
    """
    
    def __init__(self, iteration : int, y_true : np.ndarray, y_pred : np.ndarray, train_time, run_time : float):
        self.iteration = iteration
        self.accuracy = accuracy(y_true, y_pred)
        self.TPR = TPR(y_true, y_pred)
        self.FPR = FPR(y_true, y_pred)
        self.TNR = TNR(y_true, y_pred)
        self.FNR = FNR(y_true, y_pred)
        self.recall = recall(y_true, y_pred)
        self.ROC = ROC(y_true, y_pred)
        self.sensitivity = sensitivity(y_true, y_pred)
        self.specificity = specificity(y_true, y_pred)
        self.train_time = train_time
        self.run_time = run_time



    def __str__(self):
        return f"########## {self.iteration} ##########\nAccuracy   : {self.accuracy * 100: 8.4f}\nRecall     : {self.recall * 100: 8.4f}\nSensitivity: {self.sensitivity * 100: 8.4f}\nSpecificity: {self.specificity: 8.4f}\nTPR        : {self.TPR * 100: 8.4f}\nFPR        : {self.FPR * 100: 8.4f}\nTNR        : {self.TNR * 100: 8.4f}\nFNR        : {self.FNR * 100: 8.4f}\nROC        : {self.ROC: 8.4f}\nT-Time     : {self.train_time: 8.4f} s\nR-Time     : {self.run_time: 8.4f} s"



    def __repr__(self):
        return str(self)



class PerformanceList:
    """ Class to store a list of performance metrics.
    """
    
    def __init__(self):
        self.performances : Performance = []



    def add(self, performance : Performance) -> None:
        """ Add a performance to the list.
        
        Args:
            performance (Performance): The performance to add
        """
        self.performances.append(performance)
        
        
        
    def as_csv(self) -> str:
        """ Return individual performance values as a csv, making it easier to import into a spreadsheet program.

        Returns:
            str: The performance values as a csv, seperated by newlines
        """
        
        out = "Accuracy,Recall,Sensitivity,Specificity,TPR,FPR,TNR,FNR,ROC,Train Time,Run Time\n"
        
        for performance in self.performances:
            out += f"{performance.accuracy},{performance.recall},{performance.sensitivity},{performance.specificity},{performance.TPR},{performance.FPR},{performance.TNR},{performance.FNR},{performance.ROC},{performance.train_time},{performance.run_time}\n"

        return out



    def __str__(self):
        accuracies = [value.accuracy for value in self.performances]
        TPRs = [value.TPR for value in self.performances]
        FPRs = [value.FPR for value in self.performances]
        TNRs = [value.TNR for value in self.performances]
        FNRs = [value.FNR for value in self.performances]
        recalls = [value.recall for value in self.performances]
        ROCs = [value.ROC for value in self.performances]
        train_times = [value.train_time for value in self.performances]
        run_times = [value.run_time for value in self.performances]

        out = ""
        out += f"Average accuracy: {np.mean(accuracies) * 100: 8.4f}%"
        out += f"\nAverage TPR     : {np.mean(TPRs) * 100: 8.4f}%"
        out += f"\nAverage FPR     : {np.mean(FPRs) * 100: 8.4f}%"
        out += f"\nAverage TNR     : {np.mean(TNRs) * 100: 8.4f}%"
        out += f"\nAverage FNR     : {np.mean(FNRs) * 100: 8.4f}%"
        out += f"\nAverage recall  : {np.mean(recalls) * 100: 8.4f}%"
        out += f"\nAverage ROC     : {np.mean(ROCs): 8.4f}"
        out += f"\nAverage T-Time  : {np.mean(train_times): 8.4f} s"
        out += f"\nAverage R-Time  : {np.mean(run_times): 8.4f} s"
        out += f"\nOver {len(self.performances)} repetitions"
        
        return out
        
        
    