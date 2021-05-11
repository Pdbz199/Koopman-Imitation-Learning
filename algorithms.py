import numpy as np
import scipy as sp
import numba as nb

#%% (Theta=Psi_X_T, dXdt=dPsi_X_T, lamb=0.05, n=d)
def SINDy(Theta, dXdt, d, lamb=0.05):
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0] # Initial guess: Least-squares
    
    for k in range(10):
        smallinds = np.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                          # and threshold
        for ind in range(d):                       # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]
            
    L = Xi
    return L

# conjugate gradient?
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
@nb.njit(fastmath=True)
def ols(X, Y, pinv=True):
    if pinv:
        return np.linalg.pinv(X.T @ X) @ X.T @ Y
    return np.linalg.inv(X.T @ X) @ X.T @ Y

#%% (X=Psi_X_T, Y=dPsi_X_T, rank=8)
@nb.njit(fastmath=True)
def rrr(X, Y, rank=8):
    B_ols = ols(X, Y) # if infeasible use GD (numpy CG)
    U, S, V = np.linalg.svd(Y.T @ X @ B_ols)
    W = V[0:rank].T

    B_rr = B_ols @ W @ W.T
    L = B_rr#.T
    return L

# L = gedmd(Psi_X_tilde.T, dPsi_X_tilde.T)
@nb.njit(fastmath=True)
def gedmd(X, Y, rank=8):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    U_tilde = U[:, :rank]
    Sigma_tilde = np.diag(Sigma[:rank])
    VT_tilde = VT[:rank]

    # Sigma_tilde.T @ (U_tilde.T @ Y @ VT_tilde.T).T = x
    # 8 x 8                8 x 20000 x 20000 x 21 x 21 x 8
    M_tilde = np.linalg.solve(Sigma_tilde.T, (U_tilde.T @ Y @ VT_tilde.T).T).T
    L = M_tilde.T # estimate of Koopman generator
    return L