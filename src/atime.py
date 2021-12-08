# Função para avaliar o tempo de resolução
# de mínimos quadrados, em diferentes abordagens.

import leastsquares as ls
from numpy.linalg import lstsq
from time import time

def calc_time(A, b):
    ti = time()
    for _ in range(0, 10000):
        ls.qrp(A, b) # via QR (por Gram-Schmidt)
    tqr = time()
    for _ in range(0, 10000):
        ls.qrnp(A, b) # via QR (pelo numpy)
    tqrnp = time()
    for _ in range(0, 10000):
        ls.neq(A, b) # via equações normais
    tne = time()
    for _ in range(0, 10000):
        ls.svd(A, b) # via SVD
    tsvd = time()
    for _ in range(0, 10000):
        lstsq(A, b, rcond = None)
    tnp = time()

    print('Tempo via QR (por Gram-Schmidt): ' + str(tqr - ti))
    print('Tempo via QR (numpy): ' + str(tqrnp - tqr))
    print('Tempo via equações normais: ' + str(tne - tqrnp))
    print('Tempo via SVD: ' + str(tsvd - tne))
    print('Tempo do numpy: ' + str(tnp - tsvd))