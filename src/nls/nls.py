# Implemetações de algoritmos p/ resolver mínimos quadrados
# não lineares. Aqui, onde é necessária alguma resolução via
# mínimos quadrados lineares, usa-se o método do numpy (np.linalg.lstsq)
# visto que o objetivo não é testar os métodos produzidos para
# a versão linear, e sim para a versão não linear.
# Implementações baseadas nos pseudoalgoritmos do VMLS.

import numpy as np

# Suporte para manipulação de escalares
def lstsq(A, b):
    if(np.isscalar(A)):
        return np.divide(b, A)
    return np.linalg.lstsq(A, b, rcond=None)[0]

def solve(A, b):
    if(np.isscalar(A)):
        # Fórmula (18.9) VMLS
        return np.divide(b, A)
    return np.linalg.solve(A, b)

def lm_opt(A, b):
    if(np.isscalar(A)):
        return 2 * A * b
    return 2 * A.T @ b

def lm_update(A, b, l, shape):
    if(np.isscalar(A)):
        # Fórmula (18.13) do VMLS
        return (A / (l + (A ** 2))) * b
    return np.linalg.inv(A.T @ A + l * np.identity(shape)) @ A.T @ b

# Métodos para resolução de mínimos quadrados não-lineares
def gauss_newton(f, Df, x1, k_max = 100, tol = 1e-6):
    xk = x1
    for k in range(k_max):
        if(np.linalg.norm(f(xk)) < tol):
            break
        # Iteração do método de Gauss-Newton
        # Fórmula (18.6) do VMLS
        xk -= lstsq(Df(xk), f(xk))
    return {'x': xk, 'it': k}

def newton(f, Df, x1, k_max = 100, tol = 1e-6):
    xk = x1
    for k in range(k_max):
        if(np.linalg.norm(f(xk)) < tol):
            break
        # Iteração do método de Newton
        # Desenvolvimento da (18.6), pág. 388 do VMLS.
        # Basicamente, como sabe-se que f: R^n -> R^n, pode-se
        # apenas resolver um sistema linear.
        xk -= solve(Df(xk), f(xk))
    return {'x': xk, 'it': k}

# Não suporta escalarares
def levenberg_marquardt(f, Df, x1, lambda1, k_max = 100, tol = 1e-6):
    xk = x1
    l = lambda1 # variável l pois lambda é uma palavra reservada em Python
    shape = None
    if(not np.isscalar(x1)):
        shape = x1.shape[0]
    for k in range(k_max):
        # pág. 393 VMLS
        # Stopping criteria - Small residual
        # or
        # Stopping criteria - Small optimality condition residual
        if(np.linalg.norm(f(xk)) < tol or np.linalg.norm(lm_opt(Df(xk), f(xk))) < tol):
            break
        A = Df(xk)
        x_next = xk - lm_update(A, f(xk), l, shape)
        # Passo 3 do pseudo-algoritmo do VMLS
        # Definindo lambda e o valor de x
        if(np.linalg.norm(f(x_next)) < np.linalg.norm(f(xk))):
            l *= 0.8
            xk = x_next
        else:
            l *= 2.0
    return {'x': xk, 'it': k}