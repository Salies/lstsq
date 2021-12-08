# Implementação prática do gráfico
# mostrado na Figura 13.3 do VMLS.
# Os dados foram extraídos de
# https://www.worldometers.info/oil/#oil-consumption .

import numpy as np
import leastsquares as ls
import matplotlib.pyplot as plt
import atime as t
from data import world_oil_consumption_yearly

# Construção de A e b
b = np.array(world_oil_consumption_yearly())
A = np.column_stack((np.ones(b.size), np.arange(b.size)))

# Avaliando o tempo de execução
#t.calc_time(A, b)

# Calculando para o plot
x = np.linalg.lstsq(A, b, rcond = None)[0]
# x = ls.qrp(A, b) # via QR (por Gram-Schmidt)
# x = ls.qrnp(A, b) # via QR (pelo numpy)
# x = ls.neq(A, b) # via equações normais
# x = ls.svd(A, b) # via SVD

# Plotando o gráfico
plt.figure(figsize=(7,6))
plt.title('Consumo mundial de petróleo, por ano (1980 a 2016)')
plt.xlabel('Ano')
plt.ylabel('Consumo de petróleo\n(em dezenas de bilhões de barris)')
plt.plot(np.arange(1980, 2017), b, 'o', fillstyle='none', color='blue')
plt.plot(np.arange(1980, 2017), A @ x, color='red')   
plt.show()  