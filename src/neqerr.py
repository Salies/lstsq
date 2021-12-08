# Exemplo de como a solução via equações normais
# produz um resultado com erro grande quando a matriz
# A é mal condicionada.
# Exemplo adaptado de:
# https://andreask.cs.illinois.edu/cs357-s15/public/demos/09-svd-applications/Least%20Squares%20using%20the%20SVD.html

import numpy as np
import leastsquares as ls

np.random.seed(12)
A = np.random.randn(6, 4)
b = np.random.randn(6)
A[3] = A[4] + A[5]
A[1] = A[5] + A[1]
A[2] = A[3] + A[1]
A[0] = A[3] + A[1]

x_neq = ls.neq(A, b)
x_svd = ls.svd(A, b)

print("Norma do vetor residual (equações normais): " + str(np.linalg.norm(A @ x_neq - b)))
print("Norma do vetor residual (SVD): " + str(np.linalg.norm(A @ x_svd - b)))