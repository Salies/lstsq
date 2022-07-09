import matplotlib.pyplot as plt
import numpy as np
from nls import levenberg_marquardt, gauss_newton, newton
import scipy.optimize as sciop

# Para 500 pontos entre -3 e 8pi
const = 200
x = np.linspace(-3, 8 * np.pi, const)

# Função
f = lambda theta: np.array([theta[0] * np.cos(x) * np.exp(x * theta[1]), 
                            theta[0] * np.sin(x) * np.exp(x * theta[1])])
'''df = lambda theta: np.matrix([
    [np.exp(theta[1] * x) * np.cos(x), theta[0] * x * np.exp(theta[1] * x) * np.cos(x)],
    [np.exp(theta[1] * x) * np.sin(x), theta[0] * x * np.exp(theta[1] * x) * np.sin(x)]
])'''
xy = f([1, .2])
xy_noise = [xy[0] + np.random.normal(0, 12, const), xy[1] + np.random.normal(0, 12, const)]
ff = lambda theta: (f(theta) - xy_noise).flatten()

# Resposta original
xy_aprox = sciop.least_squares(ff, [0, 0])
faprox = f(xy_aprox.x)

plt.scatter(xy_noise[0], xy_noise[1], facecolors='none', edgecolors='#00ff00', label='Pontos perturbados da função original')
plt.plot(faprox[0], faprox[1], label='Regressão não linear')
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
plt.legend(prop={'size': 14})
plt.show()