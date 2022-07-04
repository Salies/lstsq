import numpy as np
from nls import gauss_newton, newton, levenberg_marquardt

# Teste do exemplo (18.10) do VMLS
# A função f converge muito rapidamente para 0
# Logo, espera-se que o resultado obtido seja muito próximo de 0

f = lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
# Derivada de f
Df = lambda x: (4 * np.exp(2 * x)) / ((np.exp(2 * x) + 1)**2)

print(newton(f, Df, 0.95, 20, 1e-6))
print(gauss_newton(f, Df, 0.95, 20, 1e-6))
print(levenberg_marquardt(f, Df, 0.95, 1.0, 20, 1e-6))