import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def f(x):
    return np.sin(2 * np.pi * x**3)**3

x_dense = np.linspace(0, 1, 500)
x_data = np.random.rand(500)
x_data.sort()
y_true = f(x_data)
y_noisy = y_true + np.random.normal(0, 0.1, size=x_data.shape)

x_knots_linear = np.linspace(0, 1, 7)
y_knots_linear = f(x_knots_linear) + np.random.normal(0, 0.1, size=x_knots_linear.shape)
cs_linear = CubicSpline(x_knots_linear, y_knots_linear, bc_type='natural')
y_spline_linear = cs_linear(x_dense)

x_knots_custom = np.array([0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0])
y_knots_custom = f(x_knots_custom) + np.random.normal(0, 0.1, size=x_knots_custom.shape)
cs_custom = CubicSpline(x_knots_custom, y_knots_custom, bc_type='natural')
y_spline_custom = cs_custom(x_dense)

y_exact = f(x_dense)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(x_dense, y_exact, 'k', linewidth=1.5)
axs[0, 0].set_title(r'Original Function')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('f(x)')
axs[0, 0].grid(True)
axs[0, 0].set_ylim(-1.5, 1.5)

axs[0, 1].plot(x_dense, y_exact, 'k', linewidth=2, label='Original Function')
axs[0, 1].plot(x_dense, y_spline_linear, 'r-', linewidth=1.5, label='Spline Fit')
axs[0, 1].scatter(x_data, y_noisy, color='blue', marker='.', s=10, alpha=0.8, label='Noisy Data')
for xk in x_knots_linear:
    axs[0, 1].axvline(xk, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
axs[0, 1].set_title('Spline Fit (Linear Knots)')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('f(x)')
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].set_ylim(-1.5, 1.5)

axs[1, 0].plot(x_dense, y_exact, 'k', linewidth=1.5)
axs[1, 0].set_title(r'Original Function')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('f(x)')
axs[1, 0].grid(True)
axs[1, 0].set_ylim(-1.5, 1.5)

axs[1, 1].plot(x_dense, y_exact, 'k', linewidth=2, label='Original Function')
axs[1, 1].plot(x_dense, y_spline_custom, 'r-', linewidth=1.5, label='Spline Fit')
axs[1, 1].scatter(x_data, y_noisy, color='blue', marker='.', s=10, alpha=0.8, label='Noisy Data')
for xk in x_knots_custom:
    axs[1, 1].axvline(xk, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
axs[1, 1].set_title('Spline Fit (Custom Knots)')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('f(x)')
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
