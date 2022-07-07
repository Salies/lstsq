# Recriação aproximada do caso apresentando na figura 18.13 do VMLS.

import numpy as np
from nls import gauss_newton, levenberg_marquardt
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)

f = lambda x, theta: np.exp(theta[0] * x) * np.cos(theta[1] * x)
Df = lambda x, theta: np.column_stack([
    np.cos(x * theta[1]) * np.exp(x * theta[0]) * x,
    (-1) * x * np.exp(x * theta[0]) * np.sin(x * theta[1])
])

y = np.vstack(f(x, np.array([2, -0.5])) + np.random.normal(0, 0.25, 100) * 0.75)

ff = lambda theta: f(np.vstack(x), theta) - y
Dff = lambda theta: Df(np.vstack(x), theta)
#theta = levenberg_marquardt(ff, Dff, np.vstack([1, 0]), 1.0)
theta = gauss_newton(ff, Dff, np.vstack([1.0, 0.0]))

print("resultado: ", theta['x'])
print("número de iterações: ", theta['it'])
plt.scatter(np.vstack(x), y, facecolors='none', edgecolors='#00ff00')
plt.plot(np.vstack(x), f(x, theta['x']))
plt.show()
