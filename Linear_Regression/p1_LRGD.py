import numpy as np
import matplotlib.pyplot as plt
from src.linearRegressionGD import LinearRegressionGD

from src.utils import *

# https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2


np.random.seed(0)
x = np.random.rand(100,1)
x_train = np.c_[np.ones((x.shape[0], 1)), x]
y = 2 + 3 * x + np.random.rand(100,1)

model = LinearRegressionGD()
model.fit(x_train, y)

y_pred = model.predict(x_train)
rmse = calculate_rmse(y, y_pred)
r_squared = calculate_r_squared(y_pred, y)

print(f'Slope is: {model.coef_}')
print(f'Intercept is: {model.intercept_}')
print(f'RMSE is: {rmse}')
print(f'R**2 is: {r_squared}')


abline(model.coef_, model.intercept_)
plt.scatter(x, y_pred, s=10, color='g')
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
