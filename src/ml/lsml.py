import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Erro soma dos quadrados, de Fujii
def lse(fx, y):
    return sum(((y - fx) ** 2)) / 2

# Função de predição
# afim: f(x) = a * x + b
def f(x, a, b):
    return (a * x) + b

# Preparando o dataset
iris = np.array(load_iris()['data'])
# Da documentação do sklearn (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html):
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.
x = iris[:, 0]
y = iris[:, 2]

# Como a convergência independe de w e b
# definem-se arbitrariamente como 0
a = 0.0
b = 0.0

# Calculando a matriz A como em (Fujii)
z = np.sum(x ** 2)
o = np.sum(x)
n = len(x)
A = [
    [z, o],
    [o, n]
]
A = np.array(A)

# Pegando os autovalores de A
eig = np.linalg.eigvals(A)

# 0 < learning_rate < 2 / (autovalor máx. de A)
# pego o "meio termo"
learning_rate = 1.0 / np.max(eig)

print(f"Estado inicial:: err: {lse(f(x, a, b), y)}, a: {a}, b: {b}")

# Sem condição de terminação - faça 10000
for k in range(10000):
    # Atualizando por gradiente descendente, de acordo com Fujii
    prediction_err = y - f(x, a, b)
    a += learning_rate * np.sum(prediction_err * x)
    b += learning_rate * np.sum(prediction_err)

    # Relata a cada 100 iterações
    if k % 100 == 0:
        print(f"k {k}:, err: {lse(f(x, a, b), y)}, a: {a}, b: {b}")
            
print(f"Estado final:: err: {lse(f(x, a, b), y)}, a: {a}, b: {b}")

# A partir da Sepal Length, tentar prever a Petal Length
plt.scatter(iris[:, 0], iris[:, 2], color='none', edgecolors='violet', label='Flor')
plt.plot(iris[:, 0], a * iris[:, 0] + b, color='red', label='Regressão linear')
plt.xlabel("Comprimento da sépala", fontsize=16)
plt.ylabel("Comprimento da pétala", fontsize=16)
plt.legend(prop={'size': 14})
plt.show()
