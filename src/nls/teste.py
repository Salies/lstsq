import numpy as np
import numpy.linalg as npl
from nls import gauss_newton, newton, levenberg_marquardt

# Teste do exemplo (18.10) do VMLS
# A função f converge muito rapidamente para 0
# Logo, espera-se que o resultado obtido seja muito próximo de 0

f = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# Derivada de f
Df = lambda x: (4 * np.exp(2 * x)) / ((np.exp(2 * x) + 1)**2)

# Este mesmo teste com x1 = 1.15 leva a um erro por overflow
# (não convergência)
# Apenas LM converge independentemente do caso
t1 = newton(f, Df, 0.95, 20)
t2 = gauss_newton(f, Df, 0.95, 20)
t3 = levenberg_marquardt(f,Df, 0.95, 1.0, 20)

print(t1['x'], t2['x'], t3['x'])
print(t1['it'], t2['it'], t3['it'])

# Como pode-se observar, lm leva mais iterações para convergir
# e converge mais lentamente, dada a existência do atributo lambda 
# e a diferença no método de aproximação.
# Logo, é necessária uma engenharia maior para uma rápida convergência mas,
# dos métodos analisados, o de Levenberg-MArquardt é o mais robusto.