import leastsquares as ls
import numpy as np
import matplotlib.pyplot as plt
from data import parabola

# 
data_x, b = parabola()

A = np.column_stack((
    np.ones(data_x.size),
    data_x,
    np.square(data_x)
))

x = np.linalg.lstsq(A, b, rcond = None)[0]
#x = ls.qrp(A, b) # via QR (por Gram-Schmidt)
#x = ls.qrnp(A, b) # via QR (pelo numpy)
#x = ls.neq(A, b) # via equações normais
#x = ls.svd(A, b) # via SVD

t = np.linspace(-0.5, 2, 1000)
y_plot = x[2]*(t**2) + x[1]*t + x[0]

plt.title('Parábola aproximada')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(data_x, b, 'o', fillstyle='none', color='blue')
plt.plot(t, y_plot, color='red')
plt.show()