import numpy as np

# https://towardsdatascience.com/classification-model-from-scratch-49f24bdd0636
# https://github.com/amindazad/Naive_Bayes_Classifier

class NaiveBayesClassifier:
    
    def __init__(self) -> None:
        pass
    
    def separate_classes(self, X, y):
        """Separates individual classes from training data

        Parameters:
        -----------
        X: ndarray
            Training data.
        y: ndarray
            True data - results.
        
        Returns:
        --------
        Dictionary for classes.
        """
        separated_classes = {}

        for i in range(len(X)):
            feature_values = X[i]
            class_name = y[i]

            if class_name not in separated_classes:
                separated_classes[class_name] = []
            separated_classes[class_name].append(feature_values)
        
        return separated_classes

    def stat_info(self, X):
        """Creates Standard Deviation and Mean for each class

        Parameters:
        -----------
        X: ndarray
            Training data.        
        """

        for feature in zip(*X):
            yield{
                # std is standard deviation - směrodatná odchylka
                'std': np.std(feature),
                'mean': np.mean(feature)
            }
    
    def gaussian_distribution(self, x, mean, std):
        """Calculates the Gaussian distribution for the training data.

        Parameters:
        -----------
        x: ndarray
            Data.
        mean: float
            Mean for the data.
        std: float
            Standard deviation for the data.
        """
        exponent = np.exp(-((x - mean) ** 2 / (2*std**2)))

        return exponent / (np.sqrt(2 * np.pi) * std)

    def fit(self, X, y):
        """Trains the model.
        
        Parameters:
        -----------
        X: ndarray
            Training data.
        y: ndarray
            Result data.
        
        Returns:
        Probabilities for each class with saved training data.
        """
        separated_classes = self.separate_classes(X, y)
        self.class_summary = {}

        for class_name, feature_values in separated_classes.items():
            self.class_summary[class_name] = {
                # prior probability - apriorní
                'prior_prob': len(feature_values)/len(X),
                'summary': [i for i in self.stat_info(feature_values)],
            }
        
        return self.class_summary
    
    def accuracy(self, y, y_pred):
        """Accuracy of the results.
        
        Parameters:
        -----------
        y: ndarray
            Actual true data.
        y_pred: ndarray
            Predicted results by the classifier.
        
        Returns:
        --------
        Accuracy of the model.
        """
        true_true = 0
        for y_t, y_p in zip(y, y_pred):
            if y_t == y_p:
                true_true += 1
        return true_true / len(y)
    
    def predict(self, X):
        """Predicts to which class do the values from test data X belong.
        
        Parameters:
        -----------
        X: ndarray
            Test data.
        
        Returns:
        --------
        Array of classes to which each row belongs.
        """

        MAPs = []

        for row in X:
            # also referred to as "posterior" - aposteriorní
            joint_probability = {}

            for class_name, features in self.class_summary.items():
                total_features = len(features['summary'])
                likelihood = 1

                for idx in range(total_features):
                    feature = row[idx]
                    mean = features['summary'][idx]['mean']
                    std = features['summary'][idx]['std']
                    normal_probability = self.gaussian_distribution(feature, mean, std)
                    likelihood *= normal_probability
                
                prior_probability = features['prior_prob']
                joint_probability[class_name] = prior_probability * likelihood
            
            MAP = max(joint_probability, key=joint_probability.get)
            MAPs.append(MAP)
        
        return MAPs
