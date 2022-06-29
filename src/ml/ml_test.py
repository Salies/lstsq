import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from sklearn.datasets import load_iris

# Inicializa os pesos e a bias, pois estamos trabalhando com uam função afim
# Como explicado em texto, os valores iniciais aqui em nada afetarão a convergência
# do modelo, portanto incializamos tudo com 0.0.
def params_init(n):
    return {"weight": np.zeros(n), "bias": 0.0}

# Erro soma dos quadrados, do artigo japonês
def lse(predictions, targets):
    return sum(((predictions - targets) ** 2)) / 2

def f(x, w, b):
    return (w * x) + b

iris = np.array(load_iris()['data'])
# Da documentação do sklearn (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html):
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.
x = iris[:, 0]
y = iris[:, 2]

w = 0.0
b = 0.0
# Tem que tomar muito cuidado na hora de escolher o learning rate!
# TODO: tentar automatizar isso
learning_rate = 5*10e-6

predictions = [f(_x, w, b) for _x in x]

# Calcula o custo inicial (soma dos quadrados / 2)
initial_error = lse(predictions, y)
    
print("Estado inicial:")
print("erro acumulado: " + str(initial_error))
print("w: " + str(w) + ", " + str(b))

# Otimiza por gradiente descendente
# TODO: Peguei o algoritmo de gradiente descendente de algum lugar, verificar e corrigir depois
# Extendi o algoritmo pra suportar o b (bias) também - 
for i in range(20000):
    r = 1.0
    acc_grad_w0 = 0
    acc_grad_b = 0
    for _x, y_target in zip(x, y):
        acc_grad_w0 += (f(_x, w, b) - y_target)*_x
        acc_grad_b += (f(_x, w, b) - y_target)

    w -= learning_rate * acc_grad_w0 * r
    b -= learning_rate * acc_grad_b * r

    if i % 1000 == 0:
        print("\nIteração " + str(i))
        print("erro acumulado: " + str(lse(f(x, w, b), y)))
        print("w: " + str(w) + ", " + str(b))
            
print("\nEstado final:")
print("erro acumulado: " + str(lse(f(x, w, b), y)))
print("w: " + str(w) + ", " + str(b))

# A partir da Sepal Length, tentar prever a Petal Length
# TODO: Daria pra só treinar parte do conjunto e tentar prever o resto?
plt.scatter(iris[:, 0], iris[:, 2], color='none', edgecolors='violet')
plt.plot(iris[:, 0], w * iris[:, 0] + b, color='red')
plt.show()
