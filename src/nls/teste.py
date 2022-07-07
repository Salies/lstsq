import numpy as np
import numpy.linalg as npl
from nls import gauss_newton, newton

# Teste do exemplo (18.10) do VMLS
# A função f converge muito rapidamente para 0
# Logo, espera-se que o resultado obtido seja muito próximo de 0

f = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# Derivada de f
Df = lambda x: (4 * np.exp(2 * x)) / ((np.exp(2 * x) + 1)**2)

# FUNÇÃO EXTRAÍDA DO VMLS PYTHON COMPANION
# POR VBARTLE
# Fonte: https://github.com/vbartle/VMLS-Companions
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
        xt = x - npl.lstsq(np.vstack([Dfk,np.sqrt(lambd)*np.eye(n)]),np.vstack([fk,np.zeros((n,1))]), rcond=None)[0]
        if npl.norm(f(xt)) < npl.norm(fk):
            lambd = .8*lambd
            x = xt
        else:
            lambd = 2.0*lambd
    return {'x': x, 'it': k}

# Este mesmo teste com x1 = 1.15 leva a um erro por overflow
# (não convergência)
# Apenas LM converge independentemente do caso
t1 = newton(f, Df, 0.95, 20, 1e-6)
t2 = gauss_newton(f, Df, 0.95, 20, 1e-6)
t3 = levenberg_marquardt(f,Df, np.array([0.95]), 1.0, 20)

print(t1['x'], t2['x'], t3['x'][0][0])
print(t1['it'], t2['it'], t3['it'])

# Como pode-se observar, lm leva mais iterações para convergir
# e converge mais lentamente, dada a existência do atributo lambda 
# e a diferença no método de aproximação.
# Logo, é necessária uma engenharia maior para uma rápida convergência mas,
# dos métodos analisados, o de Levenberg-MArquardt é o mais robusto.