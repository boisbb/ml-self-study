import numpy as np
from .utils import gradients, x_transform, calculate_mse

# https://towardsdatascience.com/polynomial-regression-in-python-b69ab7df6105

class PolynomialLinearRegressionGD:

    def __init__(self, eta=0.01, epochs=1000, batch_size=100) -> None:
        self.eta = 0.01
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self, X, y, degrees=[]):
        
        x = x_transform(X, degrees)
        m, n = x.shape

        self.w_ = np.zeros((n,1))
        self.bias_ = 0

        y = y.reshape(m,1)

        self.loss = []

        for epoch in range(self.epochs):
            for i in range((m-1)//self.batch_size + 1):
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                x_batch = x[start_i:end_i]
                y_batch = y[start_i:end_i]

                y_pred = np.dot(x_batch, self.w_) + self.bias_

                # dw and db are gradients/derivatives of weights and bias respectively
                dw, db = gradients(x_batch, y_batch, y_pred)

                self.w_ -= self.eta*dw
                self.bias_ -= self.eta*db
            
            loss = calculate_mse(y, np.dot(x, self.w_) + self.bias_)
            self.loss.append(loss)
    
    def predict(self, X, degrees=[]):
        x1 = x_transform(X, degrees)
        return np.dot(x1, self.w_) + self.bias_




