import numpy as np
from nls import levenberg_marquardt, gauss_newton, newton
import matplotlib.pyplot as plt

# Definindo a função
f = lambda theta, x : theta[0] * np.exp(theta[1] * x) * np.sin(theta[2] * x)

# Definindo sua matriz jacobiana
# (mostrada no documento)
Df = lambda theta, x : np.hstack([
    (np.exp(theta[1] * x) * np.sin(theta[2] * x)),
    (theta[0] * x * np.exp(theta[1] * x) * np.sin(theta[2] * x)),
    (theta[0] * x * np.exp(theta[1] * x) * np.cos(theta[2] * x))
])

# Define alguns pontos "originais" da função,
# com 60 valores x igualmente espaçados entre 0 e 5, e theta = [0.3, 0.4, 7]
x_og = np.linspace(0, 5, 60)
theta_og = np.array([0.3, 0.4, 7])
y_og = f(theta_og, x_og)
#plt.scatter(x_og, y_og, facecolors='none', edgecolors='r')
# Adicionando 75% de distorção aleatória ao eixo x
# mantenho o eixo y, para que a distorção não seja muito grande
x_noisy = x_og + np.random.normal(0, 0.1, 60) * 0.75
#plt.scatter(x_og, y_og, facecolors='none', edgecolors='green')
#plt.show()

xf = np.vstack(x_noisy)
yf = np.vstack(y_og)

# Definindo as funções já com os valores, para serem passadas
# às abordagens de mín. quadrados não lineares
#print(x_noisy)
#ff = lambda theta: np.vstack(f(theta, x_noisy) - y_og)
#Dff = lambda theta: Df(theta, np.vstack(x_noisy))
ff = lambda theta : np.vstack(theta[0] * np.exp(theta[1] * xf) * np.sin(theta[2] * xf) - yf)
Dff = lambda theta : np.hstack([
    (np.exp(theta[1] * xf) * np.sin(theta[2] * xf)),
    (theta[0] * xf * np.exp(theta[1] * xf) * np.sin(theta[2] * xf)),
    (theta[0] * xf * np.exp(theta[1] * xf) * np.cos(theta[2] * xf))
])


# Resolvendo usando as abordagens disponíveis
theta_res = gauss_newton(ff, Dff, np.vstack([0, 0, 5]), 0)

# Plotando
x = np.linspace(0, 5, 500)
y = f(theta_res, x)
plt.plot(x, y, color="red")
plt.scatter(x_noisy, y_og, facecolors='none', edgecolors='green')
plt.show()