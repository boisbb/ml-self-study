import numpy as np
import matplotlib.pyplot as plt
from src.polynomialLinearRegressionGD import PolynomialLinearRegressionGD
from src.utils import calculate_r_squared


np.random.seed(42)
X = np.random.rand(1000,1)
y = 5*((X) ** (2)) + np.random.rand(1000,1)

model = PolynomialLinearRegressionGD()
model.fit(X, y, degrees=[2])

y_pred = model.predict(X, [2])

r_squared = calculate_r_squared(y_pred, y)

w1, w2 = model.w_

print(f'Weight 1 is: {w1[0]}')
print(f'Weight 2 is: {w2[0]}')
print(f'R**2 is: {r_squared}')

# loss = MSE

plt.scatter(X, y, s=10)
plt.scatter(X, y_pred, s=10, c='pink')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
