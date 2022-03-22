import numpy as np
from nls import gauss_newton, newton, levenberg_marquardt

# Testes baseados no VMLS Python Companion
# https://github.com/vbartle/VMLS-Companions

f = lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
Df = lambda x: 4 / (np.exp(x) + np.exp(-x))**2

print(newton(f, Df, 0.95, 20, 1e-6))
print(gauss_newton(f, Df, 0.95, 20, 1e-6))
print(levenberg_marquardt(f, Df, np.array([0.95]), 1.0, 20, 1e-6)[0])