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
res1 = f([1, .2])
res_noise = [res1[0] + np.random.normal(0, 12, const), res1[1] + np.random.normal(0, 12, const)]
ff = lambda theta: (f(theta) - res_noise).flatten()

# Resposta original
res1 = f([1, .2])
# Número 12 obtido empiricamente
#ff = lambda theta: f(theta) #+ np.random.normal(0, 10, const)
#res1 = ff([1, .2])
res_aprox = sciop.least_squares(ff, [0, 0])
print(res_aprox)
faprox = f(res_aprox.x)


plt.scatter(res_noise[0], res_noise[1], color="blue")
plt.plot(faprox[0], faprox[1], color="red")
plt.show()