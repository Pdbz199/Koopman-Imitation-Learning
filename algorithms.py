import numpy as np
import scipy as sp
import numba as nb

@nb.njit(fastmath=True)
def gedmd(X, Y, rank=8):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    U_tilde = U[:, :rank]
    Sigma_tilde = np.diag(Sigma[:rank])
    VT_tilde = VT[:rank]

    M_tilde = np.linalg.solve(Sigma_tilde.T, (U_tilde.T @ Y @ VT_tilde.T).T).T
    L = M_tilde.T # estimate of Koopman generator
    return L