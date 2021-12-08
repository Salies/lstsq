# Resolução do exercício 13.3 do VMLS.
# Dados disponibilizados pelo próprio livro.

import numpy as np
import matplotlib.pyplot as plt
#import leastsquares as ls
from data import moore

data = moore()

# do exercício
t = data[:, 0]
N = data[:, 1]
n = len(t)

# Construindo A e b
A = np.column_stack((np.ones(n, dtype=int), np.subtract(t, np.full(n, 1970)))) # para visualização, ver 13.3 em SM(1e)
b = np.log10(N)

# Resolvendo mínimos quadrados
x = np.linalg.lstsq(A, b, rcond=None)[0]
#x = ls.qrp(A, b) # via QR (por Gram-Schmidt)
#x = ls.qrnp(A, b) # via QR (pelo numpy)
#x = ls.neq(A, b) # via equações normais
#x = ls.svd(A, b) # via SVD

print("theta_1 = " + str(x[0]) + ", theta_2 = " + str(x[1]))
rms = np.linalg.norm(np.subtract(A @ x, b)) / np.sqrt(n)
print("Erro RMS: " + str(rms))

plt.yscale('log')
plt.ylabel('Transistores')
plt.xlabel('Ano')
plt.scatter(t, N, color='none', edgecolor='b')
p = np.power(np.full(n, 10), A @ x)
plt.plot(t, p, 'r')
plt.title('Lei de Moore')
plt.show()



