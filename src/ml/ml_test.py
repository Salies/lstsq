import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from sklearn.datasets import load_iris

# Inicializa os pesos e a bias, pois estamos trabalhando com uam função afim
# Como explicado em texto, os valores iniciais aqui em nada afetarão a convergência
# do modelo, portanto incializamos tudo com 0.0.
def params_init(n):
    return {"weight": np.zeros(n), "bias": 0.0}

# Erro mínimos quadrados, do artigo japonês
def lse(predictions, targets):
    return sum(((predictions - targets) ** 2) / 2.0)

iris = np.array(load_iris()['data'])
# Da documentação do sklearn (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html):
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.
x = iris[:, 0]
y = iris[:, 2]

w = 0.0
b = 0.0

preds = [_x * w + b for _x in x]

print(preds)

# A partir da Sepal Length, tentar prever a Petal Length
#plt.scatter(iris[:, 0], iris[:, 2], color='none', edgecolors='violet')
#plt.show()
