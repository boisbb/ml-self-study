import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
x = np.random.rand(100,1)
x_train = np.c_[np.ones((x.shape[0], 1)), x]
y = 2 + 3 * x + np.random.rand(100,1)

model = LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

rmse = mean_squared_error(y, y_pred)
r_squared = r2_score(y, y_pred)

print(f'Slope is: {model.coef_}')
print(f'Intercept is: {model.intercept_}')
print(f'RMSE is: {rmse}')
print(f'R**2 is: {r_squared}')

# data points
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

# predicted values
plt.plot(x, y_pred, color='r')
plt.show()