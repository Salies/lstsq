import leastsquares as ls
#import atime as t
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

# Gerando o conjunto de dados
data = make_circles(100, noise=0.1)
A = [el for el in data[0]]
A = np.asarray(A)
x = A[:,0]
y = A[:,1]

# Preparando a matriz e os vetores
A = np.hstack((2*A,np.ones((len(x), 1))))
z = np.add(x**2, y**2)

# Calculando a aprox. por mínimos quadrados
theta = np.linalg.lstsq(A, z, rcond=None)[0]
#theta = ls.qrp(A, z) # via QR (por Gram-Schmidt)
#theta = ls.qrnp(A, z) # via QR (pelo numpy)
#theta = ls.neq(A, z) # via equações normais
#theta = ls.svd(A, z) # via SVD

# Avaliando o tempo de execução
#t.calc_time(A, z)

# Montando a função
a = theta[0]
b = theta[1]
R = np.sqrt(theta[2] + a**2 + b**2)
t = np.linspace(0, 2*np.pi, 100)
# Plotando o gráfico
plt.figure(figsize=(6,6))
plt.title('Aproximação circular')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y,'o', fillstyle='none', color='blue')
# Para o plot, é mais fácil a utilização
# das equações polares da circunferência.
# Ref.: https://stackoverflow.com/a/32097654 .
plt.plot(a + R*np.cos(t), b + R*np.sin(t), color='red')
plt.show()
