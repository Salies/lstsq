import numpy as np

# Em todos os casos, A e b são arrays do numpy
  # Solução via equações normais, sem qualquer otimização
def neq(A, b):
    return np.linalg.inv(A.T @ A) @ A.T @ b

# Algoritmo de Gram-Schmidt, usado para a implementação
# de fatoração QR
def gs(a):
    # coleção de vetores q, a ser construída
    q = []
    for i in range(len(a)):
        # montando q~ (aqui chamado de qt, "q til")
        qt = a[i]
        for qi in q:
            qt = np.subtract(qt, np.dot(qi, a[i]) * qi)
        # se q~ = 0, sair
        if np.linalg.norm(qt) == 0:
            return q
        # senão, normalize
        q.append(qt / np.linalg.norm(qt))
    return q

# Solução via decomposição QR. É mais seguro utilizar qrnp, contudo,
# pois essa implementação carece de algumas verificações. Para os efeitos
# do projeto, contudo, o resultado é análogo.
def qrp(A, b):
    Qt = gs(A.T)
    # Como sabe-se que A = QR ...
    R_inv = np.linalg.inv(Qt @ A)
    return R_inv @ Qt @ b

# Solução via decomposição QR, baseada em numpy
def qrnp(A, b):
    Q, R = np.linalg.qr(A)
    return np.linalg.inv(R) @ Q.T @ b

# Solução via SVD. Por ser muito complexa, a decomposição SVD não foi implementada,
# usando-se a do numpy em seu lugar. Criar uma solução baseada em outros métodos
# do numpy (como para obter vetores e valores singulares) também não seria adequado,
# do ponto de vista de eficiência.
def svd(A, b):
    # s = sigma minúsculo = vetor ranqueado sigma_1, ..., sigma_p
    U, s, Vt = np.linalg.svd(A)
    # Montando a matriz Sigma, e sua pseudo-inversa
    S = np.zeros(A.shape)
    np.fill_diagonal(S, s)
    return Vt.T @ np.linalg.pinv(S) @ U.T @ b