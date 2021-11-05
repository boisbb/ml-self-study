from csv import reader
import random

# https://github.com/amindazad/Naive_Bayes_Classifier

def load_data():
    def load_csv(filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    # Split data into train and test
    def split_data(data, weight):
        """
        Random split of a data set into training and test data sets
        
        Parameters:
        -----------
        data: array-like, dataset
        weight: float, percentage of data to be used as training
        
        Returns:
        List of two datasets
        """
        train_length = int(len(data) * weight)
        train = []
        for i in range(train_length):
            idx = random.randrange(len(data))
            train.append(data[idx])
            data.pop(idx)
        return [train, data]

    df = load_csv('wine.data')


    # Converting strings to floats
    for i in range(len(df)):
        for j in range(len(df[i])):
            df[i][j]=float(df[i][j])

    train, test = split_data(df, 0.8)

    # Define target and features
    # Target is the first column in the wine dataset

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(train)):
        y_train.append(train[i][0])
        X_train.append(train[i][1:])
        
    for i in range(len(test)):
        y_test.append(test[i][0])
        X_test.append(test[i][1:])
    
    return X_train, X_test, y_train, y_test
    