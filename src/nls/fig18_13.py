# Recriação aproximada do caso apresentando na figura 18.13 do VMLS.

import numpy as np
from nls import levenberg_marquardt
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 300)

# Definindo a função
f = lambda x, theta: theta[0] * np.exp(theta[1] * x) * np.cos(theta[2] * x + theta[3])

# Definindo sua matriz jacobiana
# (mostrada na explicação)
Df = lambda x, theta: np.column_stack([
    (np.exp(theta[1] * x) * np.cos((theta[2] * x) + theta[3])),
    (theta[0] * x * np.exp(theta[1] * x) * np.cos((theta[2] * x) + theta[3])),
    -1.0 * (theta[0] * x * np.exp(theta[1] * x) * np.sin((theta[2] * x) + theta[3])),
    -1.0 * (theta[0] * np.exp(theta[1] * x) * np.sin((theta[2] * x) + theta[3])),
])

# Definindo um primeiro valor theta = [2, -0.5, 7, -1]
y = f(x, np.array([2, -0.5, 7, -1]))

# Criando pontos distorcidos, a serem aproximados
# Os aditivos em x são pequenos pois senão a distorção seria muito grande,
# descaracterizando a função
x_noisy = np.vstack(x + np.random.normal(0, 0.1, 300) * 0.1)
y_noisy = np.vstack(y + np.random.normal(0, 0.25, 300) * 0.75)

# Definindo a função para passar aos métodos nls
# y deve ser subtraído (fórmula (18.4), pág. 386 do VMLS)
ff = lambda theta: f(x_noisy, theta) - y_noisy
Dff = lambda theta: Df(x_noisy, theta)
theta = levenberg_marquardt(ff, Dff, np.vstack([1, 0, 3, 0]), 1.0)

print(theta)
plt.scatter(x_noisy, y_noisy, facecolors='none', edgecolors='#00ff00')
plt.plot(x, f(x, theta['x']))
plt.show()