# Um simples teste.
# Todos os casos devem retonrar
# (1/3, -1/3), aproximadamente.

import leastsquares as ls
import numpy as np

A = np.array([[2, 0], [-1, 1], [0, 2]])
b = np.array([1, 0, -1])

print(ls.neq(A, b))
print(ls.qrp(A, b))
print(ls.qrnp(A, b))
print(ls.svd(A, b))