import numpy as np
import matplotlib.pyplot as plt

def abline(slope, intercept, color='red'):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-', color=color)

def calculate_mse(y, y_pred):
    """Calculate mean squared error.

    Parameters:
    -----------
    y: ndarray
        Array of target values.
    y_pred: ndarray
        Array of predicted values.
    """

    return np.mean((y_pred - y) ** 2)

def calculate_rmse(y, y_pred):
    """Calculate root mean squared error.

    Parameters:
    -----------
    y: ndarray
        Array of target values.
    y_pred: ndarray
        Array of predicted values.
    """

    return np.sqrt(calculate_mse(y, y_pred))

def calculate_ssr(y_pred, y):
    """Calculate total sum of residuals.

    Parameters:
    -----------
    y: ndarray
        Array of target values.
    y_pred: ndarray
        Array of predicted values.
    """
    return np.sum((y_pred - y) ** 2)

def calculate_sst(y):
    """Calculate total sum of squares.

    Parameters:
    -----------
    y: ndarray
        Array of target values.
    """
    return np.sum((y - np.mean(y)) ** 2)

def calculate_r_squared(y_pred, y):
    """Calculate coefficient of determination

    Parameters:
    -----------
    y: ndarray
        Array of target values.
    y_pred: ndarray
        Array of predicted values.
    """
    sst = calculate_sst(y)
    ssr = calculate_ssr(y_pred, y)

    return 1 - (ssr/sst)

def calculate_slope_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sample_covariance = np.sum((x - x_mean) * (y - y_mean))
    sample_variance = np.sum((x - x_mean) ** 2)

    slope = sample_covariance / sample_variance
    intercept = y_mean - slope * x_mean
    
    return slope, intercept


def gradients(X, y, y_pred):
    m = X.shape[0]
    residual = y_pred - y

    # Gradient of loss for weights
    dw = np.dot(X.T, residual) / m

    # Gradient of loss for bias
    db = np.sum(residual) / m

    return dw, db

def x_transform(X, degrees):
    t = X.copy()

    for i in degrees:
        X = np.append(X, t**i, axis=1)
    
    return X