import numpy as np
from time import time
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from display import Display
from typing import List
from LabelEncoder import BasicLabelEncoder
import keyboard
import NaiveBayes as NB
import MLPNN
import Performance

# The dataset root directory
dataset_root : str = "dataset/"

# Define the filenames for each dataset
filenames : List[str] = [
    "bank-additional-full.csv",
    "bank-full.csv",
]

# Define the continuous features for each dataset
continuous_features : List[List[int]] = [
    [0,10,12,13,15,16,17,18,19],
    [0,5,11,13,14]
]



def get_dataset() -> tuple[int, DataFrame]:
    selection =  Display.menu("Select a dataset to use", filenames)
    filename = dataset_root + filenames[selection]
    
    return selection, read_csv(filename, sep=";")



def get_process() -> int:
    return Display.menu("Select a process to run", ["Naive Bayes", "Multi-Level Perceptron Neural Network"])



def get_repetitions() -> int:
    x = -1
    
    while x < 1: x = Display.get_int("Enter the number of repetitions to run")
    
    return x



def get_layer_count() -> int:
    x = -1
    
    while x < 2: x = Display.get_int("Enter the number of layers in the network")
    
    return x



def get_layer_size(layer : int) -> int:
    x = -1
    
    while x < 1: x = Display.get_int(f"Enter the number of nodes in hidden layer {layer + 1}")
    
    return x



def get_learn_rate_decay() -> float:
    x = -1.0
    
    while x < 0.0: x = Display.get_float("Enter the learning rate decay")
    
    return x



def get_epochs() -> int:
    x = -1
    
    while x < 1: x = Display.get_int("Enter the number of epochs to run")
    
    return x



def get_learn_rate() -> float:
    x = -1.0
    
    while x < 0.0: x = Display.get_float("Enter the learning rate")
    
    return x



def get_layer_sizes(layer_count : int) -> List[int]:
    # Ignore the first layer, as it is the input layer
    # Also ignore the last layer, as it is the output layer
    return [get_layer_size(i) for i in range(1, layer_count - 1)]



def NaiveBayes(iter : int, selection : int, dataset : DataFrame) -> Performance.Performance:
    data = dataset.to_numpy()
    
    train_time_start = time()
    
    # Create the label encoder
    ble = BasicLabelEncoder()
    
    # Fit the label encoders
    ble.fit(data, continuous_features[selection])
    data = ble.transform(data)
    
    # Split the data into values and results
    X = data[:, :-1]
    y = data[:, -1]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and fit the Naive Bayes classifier
    nb = NB.NaiveBayes()
    nb.fit(X_train, y_train)
    
    train_time_end = time()
    
    run_time_start = time()
    
    # Predict the classes
    y_pred = nb.predict(X_test)
    
    run_time_end = time()
    
    perf = Performance.Performance(
        iter,
        y_test,
        y_pred,
        train_time_end - train_time_start,
        run_time_end - run_time_start
    )
    
    print("##############################################")
    print(perf)
    print("##############################################")
    
    return perf



def MultiLevelPerceptronNeuralNetwork(iter : int, selection : int, dataset : DataFrame, layer_sizes : List[int], epochs : int, learn_rate : float, learn_rate_decay : float) -> Performance.Performance:
    train_time_start = time()
    
    # Network sizes : [input, hidden, hidden, ..., output]
    net = MLPNN.Network([dataset.shape[1] - 1] + layer_sizes + [1])
    
    # Encode the data
    data = dataset.to_numpy()
    ble = BasicLabelEncoder()
    ble.fit(data, continuous_features[selection])
    data = ble.transform(data)
    
    # Split the data into values and results
    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(np.float64)
    
    # Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train the network
    training_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))
    net.learn_rate_decay = learn_rate_decay
    net.stochastic_gradient_descent(training_data, epochs, 10, learn_rate, test_data)
    
    train_time_end = time()
    
    run_time_start = time()
    
    # Predict the classes
    y_pred = net.predict(X_test)
    
    # Convert the predictions to binary
    y_pred = np.where(y_pred > 0.5, 1, 0)

    run_time_end = time()
    
    perf = Performance.Performance(
        iter,
        y_test,
        y_pred,
        train_time_end - train_time_start,
        run_time_end - run_time_start
    )
    
    print("##############################################")
    print(perf)
    print("##############################################")
    
    return perf



def main():
    # Select the dataset to use and read it in
    dataset_selection, dataset = get_dataset()
    
    process_selection = get_process()
    
    n_layers = get_layer_count() if process_selection == 1 else 0
    layer_sizes = get_layer_sizes(n_layers) if process_selection == 1 else []
    epochs = get_epochs() if process_selection == 1 else 0
    learn_rate = get_learn_rate() if process_selection == 1 else 0.0
    learn_rate_decay = get_learn_rate_decay() if process_selection == 1 else 0.0
    
    values = Performance.PerformanceList()
    
    for i in range(get_repetitions()):
        if process_selection == 0:
            values.add(NaiveBayes(i + 1, dataset_selection, dataset))
        elif process_selection == 1:
            values.add(MultiLevelPerceptronNeuralNetwork(i + 1, dataset_selection, dataset, layer_sizes, epochs, learn_rate, learn_rate_decay))
        else:
            print("Invalid process selection")
    
    print(values)
    
    print("##############################################")
    print("Press C to save the results as a CSV")
    print("Press Q to exit")
    print("##############################################")
    
    while True:
        key = keyboard.read_event()
        
        if key.event_type == keyboard.KEY_DOWN:
            if key.name.lower() == "c":
                with open("results.csv", "w") as f:
                    f.write(values.as_csv())
                print("Results saved as results.csv")
                break
            elif key.name.lower() == "q":
                break
    
    print("Goodbye!")



if __name__ == "__main__":
    main()