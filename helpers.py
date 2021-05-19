import numpy as np
import numba as nb
from numba import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@nb.njit(fastmath=True)
def l2_norm(true_state, predicted_state):
    return np.sum(np.power(( true_state - predicted_state ), 2 ))

@nb.njit(fastmath=True) #, parallel=True)
def nb_einsum(A, B):
    assert A.shape == B.shape
    res = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res += A[i,j]*B[i,j]
    return res

@nb.njit(fastmath=True)
def dpsi(X, nablaPsi, nabla2Psi, k, l, t=1):
    difference = X[:, l+1] - X[:, l]
    term_1 = (1/t) * (difference)
    term_2 = nablaPsi[k, :, l]
    term_3 = (1/(2*t)) * np.outer(difference, difference)
    term_4 = nabla2Psi[k, :, :, l]
    return np.dot(term_1, term_2) + nb_einsum(term_3, term_4)

@nb.njit(fastmath=True)
def dPsiMatrix(X, nablaPsi, nabla2Psi, k, m, t=1):
    dPsi_X = np.zeros((k, m))
    for row in range(k):
        for column in range(m-1):
            dPsi_X[row, column] = dpsi(
                X, nablaPsi, nabla2Psi, row, column, t=t
            )
    return dPsi_X

@nb.njit(fastmath=True)
def SVD(X, full_matrices=True):
    # return u, s, vh
    return np.linalg.svd(X, full_matrices=full_matrices)