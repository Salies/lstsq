import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Erro soma dos quadrados, de Fuiji
def lse(predictions, targets):
    return sum(((predictions - targets) ** 2)) / 2

# Função de predição
# afim: f(x) = w * x + b
def f(x, w, b):
    return (w * x) + b

# Preparando o dataset
iris = np.array(load_iris()['data'])
# Da documentação do sklearn (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html):
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.
x = iris[:, 0]
y = iris[:, 2]

# Como a convergência independe de w e b
# definem-se arbitrariamente como 0
w = 0.0
b = 0.0

# Calculando a matriz A como em (Fuiji)
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

print(f"Estado inicial:\nerro acumulado: {lse(f(x, w, b), y)}\nw: {w}, b: {b}")

for i in range(10000):
    prediction_err = f(x, w, b) - y
    acc_grad_w0 = np.sum(prediction_err * x)
    acc_grad_b = np.sum(prediction_err)

    # Atualiza o peso e o bias
    w -= learning_rate * acc_grad_w0
    b -= learning_rate * acc_grad_b

    # Relata a cada 1000 iterações
    if i % 1000 == 0:
        print(f"\nIteração {i}:\nerro acumulado: {lse(f(x, w, b), y)}\nw: {w}, b: {b}")
            
print(f"\nEstado final:\nerro acumulado: {lse(f(x, w, b), y)}\nw: {w}, b: {b}")

# A partir da Sepal Length, tentar prever a Petal Length
plt.scatter(iris[:, 0], iris[:, 2], color='none', edgecolors='violet')
plt.plot(iris[:, 0], w * iris[:, 0] + b, color='red')
plt.show()
