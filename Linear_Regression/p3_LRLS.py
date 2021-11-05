import numpy as np
import matplotlib.pyplot as plt
from utils.utils import calculate_slope_intercept, abline

np.random.seed(0)
x = np.random.rand(100,1)
x_train = np.c_[np.ones((x.shape[0], 1)), x]
y = 2 + 3 * x + np.random.rand(100,1)

slope, intercept = calculate_slope_intercept(x, y)

print(f'Slope is: {slope}')
print(f'Intercept is: {intercept}')

abline(slope, intercept)
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()