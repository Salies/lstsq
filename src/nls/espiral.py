import matplotlib.pyplot as plt
import numpy as np
from nls import levenberg_marquardt, gauss_newton, newton

# TODO: código um pouco bagunçado, melhorar depois

f_x = lambda x, theta: theta[0] * np.cos(x) * np.exp(x * theta[1])
f_y = lambda x, theta: theta[0] * np.sin(x) * np.exp(x * theta[1])

Df_x = lambda x, theta: np.hstack([
    np.exp(theta[1] * x) * np.cos(x),
    theta[0] * x * np.exp(theta[1] * x) * np.cos(x)
])

Df_y = lambda x, theta: np.hstack([
    np.exp(theta[1] * x) * np.sin(x),
    theta[0] * x * np.exp(theta[1] * x) * np.sin(x)
])

x_og = np.linspace(-3, 8 * np.pi, 100)
theta_og = np.vstack([1, .2])

# Aproximando o resultado x primeiro
y_a_og = f_x(x_og, theta_og)
x_noisy = x_og + np.random.normal(0, 0.1, 100) * 0.75
xf = np.vstack(x_noisy)
yaf = np.vstack(y_a_og)

ff_a = lambda theta: f_x(xf, theta) - yaf
Dff_a = lambda theta: Df_x(xf, theta)

theta_res_a = levenberg_marquardt(ff_a, Dff_a, np.vstack([1, 0]), 1.0)
print(theta_res_a)

# O mesmo para y
y_b_og = f_y(x_og, theta_og)
ybf = np.vstack(y_b_og)

ff_b = lambda theta: f_y(xf, theta) - ybf
Dff_b = lambda theta: Df_y(xf, theta)

theta_res_b = levenberg_marquardt(ff_b, Dff_b, np.vstack([1, 0]), 1.0)
print(theta_res_b)

plt.plot(f_x(x_og, theta_og), f_y(x_og, theta_og), color="blue")
plt.plot(f_x(x_og, theta_res_a), f_y(x_og, theta_res_b), color="red")
plt.show()