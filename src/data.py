from random import randrange
from numpy import array

# Consumo mundial anual de petróleo de 1980 a 2016.
# Extraído de: https://www.worldometers.info/oil/#oil-consumption .
# A ideia de usar este conjunto de dados veio 
def world_oil_consumption_yearly():
    return array([
    23036343915,
    22245594410,
    21760832800,
    21479673300,
    21766998343.5,
    21868336651.5,
    22555450746.55,
    23017505290.45,
    23708687500.3,
    24106653672.5,
    24330680124,
    24438716087.1,
    24541129714.3,
    24456570124.15,
    24973805908.25,
    25495306556.75,
    26245859819.9,
    26691609507.2,
    26992142028.25,
    27595453659.95,
    28210206640.05,
    28507128825.65,
    28776432479.6,
    29373227824.7,
    30522016049.7,
    30995659967.75,
    31436547114.1,
    31955661472.9,
    31769649658.25,
    31415881905.45,
    32430488261.05,
    32722326463.85,
    33217305522.5,
    33728506498.8,
    34324483477.4,
    35062349891.7,
    35442913090.2
    ])

# Dados do exercício 13.3 do VMLS.
def moore():
    return array([[1971, 2250], [1972, 2500], [1974, 5000], [1978, 29000], [1982, 120000], 
[1985, 275000], [1989, 1180000], [1993, 3100000], [1997, 7500000], [1999, 24000000], 
[2000, 42000000], [2002, 220000000], [2003, 410000000]])

# Gera 19 pontos, [-2, 1.7] de uma parábola 2x² - 3x + 1
# com deduções ou adições aleatórias.
def parabola():
    a = []
    y = []
    for i in range(-2, 18):
        x = i / 10
        a.append(x)
        y.append((2*(x**2) - 3*x + 1) - (randrange(-3, 4) / 10))
    return array(a), array(y)