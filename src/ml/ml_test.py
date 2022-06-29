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
learning_rate = 0.0005

predictions = [f(_x, w, b) for _x in x]

# Calcula o custo inicial (soma dos quadrados / 2)
initial_error = lse(predictions, y)
    
print("Estado inicial:")
print("erro acumulado: " + str(initial_error))
print("w: " + str(w) + ", " + str(b))

# Otimiza por gradiente descendente
# TODO: Peguei o algoritmo de gradiente descendente de algum lugar, verificar e corrigir depois
for i in range(20000):
    accumulated_grad_w0 = 0
    accumulated_grad_b = 0   
    for _x, y_target in zip(x, y):
        accumulated_grad_w0 += (f(_x, w, b) - y_target)*_x
        accumulated_grad_b += (f(_x, w, b) - y_target)   

    w_grad = (1.0/len(x)) * accumulated_grad_w0
    b_grad = (1.0/len(x)) * accumulated_grad_b 

    w = w - learning_rate * w_grad
    b = b - learning_rate * b_grad

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
