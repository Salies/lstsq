# Recriação aproximada do caso apresentando na figura 18.13
# do VMLS. Modelo para geração de dados e matriz jacobiana extraídos
# do Python companion: https://github.com/vbartle/VMLS-Companions

import numpy as np
from nls import levenberg_marquardt, gauss_newton, newton
import matplotlib.pyplot as plt

# Gerando dados
theta_ex = np.vstack([1, -.2, 1*np.pi/5, np.pi/3])
M = 30
xd = np.vstack(np.array([5*np.random.rand(M),
                         5 + 15*np.random.rand(M)]).ravel())
yd = np.vstack(theta_ex[0] * np.exp(theta_ex[1]*xd) *
               np.cos(theta_ex[2] * xd + theta_ex[3]))
N = len(xd.ravel())
yd = np.vstack(yd.ravel()) * np.vstack((1 + .2 * np.random.randn(N)))+.015 * np.vstack(np.random.randn(N))

print(len(xd))
print(len(yd))

# Preparando as funções
def f(theta): return theta[0] * np.exp(theta[1] * xd) * np.cos(theta[2] * xd + theta[3]) - yd

def Df(theta): return np.hstack([
    np.exp(theta[1]*xd) * np.cos(theta[2] * xd + theta[3]),
    theta[0] * (xd * np.exp(theta[1] * xd) * np.cos(theta[2] * xd + theta[3])),
    -theta[0] * (np.exp(theta[1] * xd) * xd *
                 np.sin(theta[2] * xd + theta[3])),
    -theta[0] * (np.exp(theta[1]*xd) * np.sin(theta[2] * xd + theta[3]))
])

# Resolvendo
#theta = levenberg_marquardt(f, Df, np.vstack([1, 0, 1, 0]), 1.0)
theta = gauss_newton(f, Df, np.vstack([1, 0, 1, 0]))

# Plotando
x = np.linspace(0, 20, 500)
y = theta[0] * np.exp(theta[1] * x) * np.cos(theta[2] * x + theta[3])

plt.scatter(xd, yd)
plt.plot(x, y, color="red")
plt.show()
