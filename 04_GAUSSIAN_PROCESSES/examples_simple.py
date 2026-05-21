import numpy as np
import matplotlib.pyplot as plt
from gp import GaussianProcessRegressor
from kernels import RBFKernel, PeriodicKernel

np.random.seed(42)
print("Generating GP examples...")

# Example 1
X_train = np.sort(np.random.uniform(0, 10, 15)).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + 0.15 * np.random.randn(15)
X_test = np.linspace(0, 10, 200).reshape(-1, 1)

gp = GaussianProcessRegressor(kernel=RBFKernel(1.0), alpha=0.02)
gp.fit(X_train, y_train)
y_mean, y_std = gp.predict(X_test, return_std=True)

plt.figure(figsize=(12, 6))
plt.plot(X_test, np.sin(X_test), 'k--', label='True')
plt.scatter(X_train, y_train, c='red', s=100, label='Train', zorder=5)
plt.plot(X_test, y_mean, 'b-', linewidth=2, label='GP Mean')
plt.fill_between(X_test.ravel(), y_mean-1.96*y_std, y_mean+1.96*y_std, alpha=0.2)
plt.legend()
plt.title('GP Regression with Uncertainty')
plt.savefig('images/01_gp_basic.png', dpi=150)
plt.close()
print("Saved 01_gp_basic.png")

print("Done!")
