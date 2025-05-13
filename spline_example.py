import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def f(x):
    return np.sin(2 * np.pi * x**3)**3

np.random.seed(0)

x_dense = np.linspace(0, 1, 500)
x_data = np.random.rand(500)
x_data.sort()
y_true = f(x_data)
y_noisy = y_true + np.random.normal(0, 0.1, size=x_data.shape)

num_intervals = 6
x_knots = np.linspace(0, 1, num_intervals + 1)
y_knots = f(x_knots) + np.random.normal(0, 0.1, size=x_knots.shape)

cs = CubicSpline(x_knots, y_knots, bc_type='natural')

y_exact = f(x_dense)
y_spline = cs(x_dense)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].plot(x_dense, y_exact, 'k', linewidth=1.5, label='Original Function')
axs[0].set_title(r'Original Function: $f(x) = \sin^3(2\pi x^3)$')
axs[0].set_xlabel('x')
axs[0].set_ylabel('f(x)')
axs[0].grid(True)
axs[0].set_ylim(-1.5, 1.5)

axs[1].plot(x_dense, y_exact, 'k', linewidth=2, label='Original Function')
axs[1].plot(x_dense, y_spline, 'r-', linewidth=1.5, label='Spline Fit')
axs[1].scatter(x_data, y_noisy, color='blue', marker='*', s=10, alpha=0.8, label='Noisy Data')

for xk in x_knots:
    axs[1].axvline(xk, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

axs[1].set_title('Original Function, Noisy Data, and Spline Fit')
axs[1].set_xlabel('x')
axs[1].set_ylabel('f(x)')
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
