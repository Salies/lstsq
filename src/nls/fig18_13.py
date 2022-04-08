# Recriação aproximada do caso apresentando na figura 18.13 do VMLS.

import numpy as np
from nls import levenberg_marquardt, gauss_newton, newton
import matplotlib.pyplot as plt

# Definindo a função
f = lambda x, theta: theta[0] * np.exp(theta[1] * x) * np.cos(theta[2] * x + theta[3])

# Definindo sua matriz jacobiana
# (mostrada na explicação)
Df = lambda x, theta: np.hstack([
    (np.exp(theta[1] * x) * np.cos((theta[2] * x) + theta[3])),
    (theta[0] * x * np.exp(theta[1] * x) * np.cos((theta[2] * x) + theta[3])),
    -1.0 * (theta[0] * x * np.exp(theta[1] * x) * np.sin((theta[2] * x) + theta[3])),
    -1.0 * (theta[0] * np.exp(theta[1] * x) * np.sin((theta[2] * x) + theta[3])),
])

# Define alguns pontos "originais" da função,
# com 60 valores x igualmente espaçados
x_og = np.linspace(0.2, 5.75, 60)
# Tentando pegar valores para theta que recriem o gráfico visto no livro
# apesar disso não ter nenhuma diferença na testagem dos algoritmos em si
theta_og = np.vstack([2, -0.5, 7, -1])
y_og = f(x_og, theta_og)
# Adicionando 75% de distorção aleatória ao eixo x
# mantém-se o eixo y, para que a distorção não seja muito grande
x_noisy = x_og + np.random.normal(0, 0.1, 60) * 0.75
# Ajustando aos padrões dos algoritmos
xf = np.vstack(x_noisy)
yf = np.vstack(y_og)
# Definindo funções, já com os valores,
# a serem passadas aos algoritmos de resolução
ff = lambda theta: f(xf, theta) - yf
Dff = lambda theta: Df(xf, theta)

# Calculando
# (usa-se Levenberg-Marquardt para garantir que não haja erros,
# visto que a distorção das entradas é aleatória)
theta_res = levenberg_marquardt(ff, Dff, np.vstack([1, 0, 1, 0]), 1.0)

# Plotando
x = np.linspace(0.2, 5.75, 500)
y = f(x, theta_res)
# Plano cartesiano
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
# Mostrando
plt.scatter(x_noisy, y_og, facecolors='none', edgecolors='#00ff00', label="Pontos distorcidos")
plt.plot(x, y, color="blue", label = "Função aproximada")
# Aqui não se usa x_og e y_og simplesmente porque são poucos pontos (60),
# logo o gráfico não ficaria "suave" como um senoide deve ser
x_og_plt = np.linspace(0.2, 5.75, 500)
y_og_plt = np.vstack(f(x_og_plt, theta_og))
plt.plot(np.vstack(x_og_plt), y_og_plt, color="#db4d4d", label="Função original")
plt.legend(loc="upper right")
plt.show()