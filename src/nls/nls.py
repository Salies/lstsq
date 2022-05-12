# Implemetações de algoritmos p/ resolver mínimos quadrados
# não lineares. Aqui, onde é necessária alguma resolução via
# mínimos quadrados lineares, usa-se o método do numpy (np.linalg.lstsq)
# visto que o objetivo não é testar os métodos produzidos para
# a versão linear, e sim para a versão não linear.
# Implementações baseadas nos pseudoalgoritmos do VMLS.

import numpy as np
import numpy.linalg as npl
import sympy

# Funções para resolver as equações lineares,
# lidando com "inversões" de matrizes 1x1.
def safe_solve(fx, Dfx):
    if(isinstance(Dfx, float)):
        return np.divide(fx, Dfx)
    return np.linalg.inv(Dfx) @ fx

def safe_lstsq(A, b):
    if(isinstance(A, float)):
        return np.divide(b, A)
    return np.linalg.lstsq(A, b, rcond = None)[0]

def safe_matrix(A):
    if(isinstance(A, float)):
        return np.vstack([A])
    return A

def gauss_newton(f, Df, x1, k_max = 10, err = 1e-6):
    # Define o x a ser calculado, valor inicial em x1
    xk = x1
    for _ in range(k_max):
        fxk = f(xk)
        #print("printei!")
        #print(fxk)
        # Condição de terminação precoce (VMLS, p. 387).
        if(np.linalg.norm(fxk) <= err):
            break
        safe_df = safe_matrix(Df(xk))
        # Condição de terminação precoce (ld em Df).
        inds = sympy.Matrix(safe_df).rref()[1]
        if(safe_df.shape[1] != len(inds)):
            raise ValueError("As colunas de Df(xk) não são linearmente independentes.")
        xk = xk - safe_lstsq(Df(xk), fxk)
    return xk

def newton(f, Df, x1, k_max = 10, err = 1e-6):
    xk = x1
    for _ in range(k_max):
        fxk = f(xk)
        if(np.linalg.norm(fxk) <= err):
            break
        safe_df = safe_matrix(Df(xk))
        inds = sympy.Matrix(safe_df).rref()[1]
        if(safe_df.shape[1] != len(inds)):
            raise ValueError("As colunas de Df(xk) não são linearmente independentes.")
        xk = xk - safe_solve(fxk, Df(xk))
    return xk

'''def levenberg_marquardt(f, Df, x1, lambda1, k_max = 100, err = 1e-6):
    xk = x1
    l = lambda1 # lambda é palavra reservada em Python
    N = len(x1)
    for _ in range(k_max):
        fxk = f(xk)
        if(np.linalg.norm(fxk) <= err):
            break
        Dfxk = Df(xk)
        # Passos 1 e 2 do VMLS
        # (minimização de x_(k+1) em função do termo apresentado, usando mínimos quadrados linear)
        # Linha baseada na solução do Python Companion: https://github.com/vbartle/VMLS-Companions
        xkp1 = xk - np.linalg.lstsq(np.vstack([Dfxk, np.sqrt(l) * np.eye(N)]), np.vstack([fxk, np.zeros((N, 1))]), rcond = None)[0]
        # Passo 3 do VMLS
        if(np.linalg.norm(f(xkp1)) < np.linalg.norm(fxk)):
            l *= 0.8
            # Atualiza o x
            xk = xkp1
            continue
        # Não atualiza o x
        l *= 2.0
    return xk'''

def levenberg_marquardt(f, Df, x1, lambda1, kmax = 100, tol = 1e-6):
    n = len(x1)
    x = x1
    lambd = lambda1
    objectives = [] 
    residuals = []
    for k in range(kmax):
        fk = f(x)
        Dfk = Df(x)
        objectives.append(npl.norm(fk)**2)
        residuals.append(npl.norm(2*np.matmul(Dfk.T,fk)))
        if npl.norm(2*np.matmul(Dfk.T,fk)) < tol:
            break
        xt = x - npl.lstsq(np.vstack([Dfk,np.sqrt(lambd)*np.eye(n)]),np.vstack([fk,np.zeros((n,1))]))[0]
        if npl.norm(f(xt)) < npl.norm(fk):
            lambd = .8*lambd
            x = xt
        else:
            lambd = 2.0*lambd
    return x, dict([("objectives", objectives),("residuals",residuals)])

