# Implemetações de algoritmos p/ resolver mínimos quadrados
# não lineares. Aqui, onde é necessária alguma resolução via
# mínimos quadrados lineares, usa-se o método do numpy (np.linalg.lstsq)
# visto que o objetivo não é testar os métodos produzidos para
# a versão linear, e sim para a versão não linear.
# Implementações baseadas nos pseudoalgoritmos do VMLS.

import numpy as np
import sympy

# Suporte para manipulação de escalares
def lstsq(A, b):
    if(np.isscalar(A)):
        return np.divide(b, A)
    return np.linalg.inv((A.T @ A)) @ A.T @ b

def solve(A, b):
    if(np.isscalar(A)):
        return np.divide(b, A)
    return np.linalg.inv(A) @ b

# Para lm
# Stopping criteria - Small optimality condition residual
def lm_op_res(A, b):
    if(np.isscalar(A)):
        return 2 * A * b
    return 2 * A.T @ b

def lm_iteration(A, b, l, n):
    if(np.isscalar(A)):
        return np.divide(b, A) + (1.0 / l)
    return np.linalg.inv(A.T @ A + l * np.eye(n)) @ A.T @ b

def gauss_newton(f, Df, x1, k_max = 100, tol = 1e-6):
    xk = x1
    for _ in range(k_max):
        if(np.linalg.norm(f(xk)) < tol):
            break
        # Iteração do método de Gauss-Newton
        # Fórmula (18.6) do VMLS
        xk -= lstsq(Df(xk), f(xk))
    return xk

def newton(f, Df, x1, k_max = 100, tol = 1e-6):
    xk = x1
    for _ in range(k_max):
        if(np.linalg.norm(f(xk)) < tol):
            break
        # Iteração do método de Newton
        # Desenvolvimento da (18.6), pág. 388 do VMLS.
        # Basicamente, como sabe-se que f: R^n -> R^n, pode-se
        # apenas resolver um sistema linear.
        xk -= solve(Df(xk), f(xk))
    return xk  

def levenberg_marquardt(f, Df, x1, lambda1, k_max = 100, tol = 1e-6):
    xk = x1
    x_shape = None if np.isscalar(x1) else x1.shape[0]
    l = lambda1 # variável l pois lambda é uma palavra reservada em Python
    for _ in range(k_max):
        # pág. 393 VMLS
        # Stopping criteria - Small residual
        if(np.linalg.norm(f(xk)) < tol):
            break
        # Stopping criteria - Small optimality condition residual
        if(np.linalg.norm(lm_op_res(Df(xk), f(xk))) < tol):
            break
        x_next = xk - lm_iteration(Df(xk), f(xk), l, x_shape)
        # Passo 3 do pseudo-algoritmo do VMLS
        # Definindo lambda e o valor de x
        if(np.linalg.norm(f(x_next)) < np.linalg.norm(f(xk))):
            l *= 0.8
            xk = x_next
        else:
            l *= 2.0
    return xk