import numpy as np

# https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2
# https://medium.com/analytics-vidhya/linear-regression-from-scratch-in-python-b6501f91c82d

class LinearRegressionGD:
    """Linear Regression - Gradient Descent
    
    Parameters:
    -------
    eta: float
        Learning rate (alpha in the web article).
    iter_cnt: int
        Number of iterations for fiting the model.
    """
    def __init__(self, eta=0.05, iter_cnt=1000) -> None:
        self.eta = eta
        self.iter_cnt = iter_cnt
    
    def fit(self, x, y):
        """Fit the training data and aproximate the weights - slope and intercept of the line.

        Parameters
        ----------
        x: ndarray, shape = [n_samples, n_features]
            Training samples
        y: ndarray, shape = [n_samples, n_target_features]
            Target values
        """

        # Based on the first tutorial, the output w_ already contains the bias (intercept) and the w0
        # some tutorials compute the bias seperately.

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.iter_cnt):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        
        self.coef_ = self.w_[1,0]
        self.intercept_ = self.w_[0,0]


    def predict(self, x):
        """Predict the outcome.

        Parameters:
        -----------
        x: ndarray, shape = [n_samples, n_features]
            Test samples
        
        Returns:
        --------
        ndarray of predictet values.
        """
        return np.dot(x, self.w_)
