import os
import numpy as np
import torch

import time


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def truncated_svd(data, truncate_threshold):
    U, Sigma, V = torch.linalg.svd(data, full_matrices=False)

    # weight_ridge = U @ torch.div(Ut_cov_HY.T, torch.pow(Sigma, 2) + ridge).T

    idx = Sigma > truncate_threshold
    U = U[:, idx]
    Sigma = Sigma[idx]
    V = V[:, idx]


    return U, Sigma, V


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy, for initial classes
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    # for incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)



# taken from: https://github.com/sbarratt/torch_cg
def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves a batch of matrix linear systems of the form

        A_i X_i = B_i,  i=1,...,K,

    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


class CG(torch.autograd.Function):

    def __init__(self, A_bmm, M_bmm=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        self.A_bmm = A_bmm
        self.M_bmm = M_bmm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose

    def forward(self, B, X0=None):
        X, _ = cg_batch(self.A_bmm, B, M_bmm=self.M_bmm, X0=X0, rtol=self.rtol,
                     atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return X

    def backward(self, dX):
        dB, _ = cg_batch(self.A_bmm, dX, M_bmm=self.M_bmm, rtol=self.rtol,
                      atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return dB
    
def cholesky_update(L, x, add=True):
    """
    Update the Cholesky decomposition L of matrix A, where A = L @ L.T

    When adding a rank-1 update: A' = A + x @ x.T (if add=True)
    When subtracting a rank-1 update: A' = A - x @ x.T (if add=False)

    Returns the updated Cholesky factor L'

    Parameters:
    -----------
    L : torch.Tensor
    Lower triangular Cholesky factor of shape (n, n)
    x : torch.Tensor
    Vector for rank-1 update of shape (n,)
    add : bool
    If True, perform rank-1 update; if False, perform rank-1 downdate

    Returns:
    --------
    L_new : torch.Tensor
    Updated Cholesky factor
    """
    if not add:
        # For downdate, we need to ensure the result will still be positive definite
        v = torch.triangular_solve(x.unsqueeze(1), L, upper=False)[0].squeeze()
        v_norm_sq = torch.sum(v**2)
        if v_norm_sq >= 1.0:
            raise ValueError("Cholesky downdate would result in a non-positive definite matrix")

    n = L.shape[0]
    L_new = L.clone()

    if add:
        # Rank-1 update: A + x @ x.T
        for k in range(n):
            # Compute the rank-1 update for the k-th row
            r = torch.sqrt(L_new[k, k]**2 + (x[k]**2 if add else -x[k]**2))
            c = r / L_new[k, k]
            s = x[k] / L_new[k, k]
            L_new[k, k] = r

            if k < n - 1:
                # Update the remaining elements in the k-th column
                L_new[k+1:, k] = (L_new[k+1:, k] + s * x[k+1:]) / c
                # Update x for the next iteration
                x[k+1:] = x[k+1:] - s * L_new[k+1:, k]
    else:
        # Rank-1 downdate: A - x @ x.T
        for k in range(n):
            # Solve the linear system to get the effect on this column
            p = x[k] / L_new[k, k]
            r = torch.sqrt(L_new[k, k]**2 - x[k]**2)
            c = r / L_new[k, k]
            s = p / L_new[k, k]
            L_new[k, k] = r

            if k < n - 1:
                # Update the remaining rows
                x[k+1:] = x[k+1:] - p * L_new[k+1:, k]
                L_new[k+1:, k] = c * L_new[k+1:, k] - s * x[k+1:]

    return L_new


def cholesky_update_batch(L, X, add=True):
    """
    Update the Cholesky decomposition L for multiple rank-1 updates at once.
    This is more efficient than applying individual updates sequentially.

    Parameters:
    -----------
    L : torch.Tensor
    Lower triangular Cholesky factor of shape (n, n)
    X : torch.Tensor
    Matrix of shape (n, m) where each column is a rank-1 update vector
    add : bool
    If True, perform rank-1 updates; if False, perform rank-1 downdates

    Returns:
    --------
    L_new : torch.Tensor
    Updated Cholesky factor after all updates
    """
    L_current = L.clone()
    for i in range(X.shape[1]):
        L_current = cholesky_update(L_current, X[:, i], add=add)

    return L_current


def test_cholesky_update():
    """
    Test function to verify the correctness of the implementation
    """
    # Create a positive definite matrix A = B @ B.T + diagonal for stability
    n = 5
    torch.manual_seed(0)
    B = torch.randn(n, n)
    A = B @ B.T + torch.eye(n) * 10

    # Compute original Cholesky decomposition
    L_original = torch.linalg.cholesky(A)

    # Create a rank-1 update vector
    x = torch.randn(n)

    # Create the updated matrix directly
    A_updated = A + torch.outer(x, x)
    L_true = torch.linalg.cholesky(A_updated)

    # Update using our function
    L_updated = cholesky_update(L_original, x, add=True)

    # Compare the results
    diff = torch.norm(L_updated - L_true)
    print(f"Difference between direct computation and update: {diff:.8f}")

    # Test downdate
    A_downdated = A - torch.outer(x, x)
    L_downdated_true = torch.linalg.cholesky(A_downdated)
    L_downdated = cholesky_update(L_original, x, add=False)
    diff_down = torch.norm(L_downdated - L_downdated_true)
    print(f"Difference for downdate: {diff_down:.8f}")

def cholesky_update_vectorized(L, X, add=True):
    """
    Fully vectorized implementation of Cholesky decomposition update
    for low-rank updates without explicit for loops.

    This implementation uses PyTorch's batched matrix operations to
    compute the updated Cholesky factor in parallel.

    Parameters:
    -----------
    L : torch.Tensor
    Lower triangular Cholesky factor of shape (n, n)
    X : torch.Tensor
    Matrix of shape (n, k) where k is the rank of the update
    add : bool
    If True, perform low-rank update (A + X@X.T)
    If False, perform low-rank downdate (A - X@X.T)

    Returns:
    --------
    L_new : torch.Tensor
    Updated Cholesky factor
    """
    n = L.shape[0]
    k = X.shape[1]
    device = L.device

    # For downdates, check positive definiteness
    if not add:
        # Solve L @ Z = X
        Z = torch.triangular_solve(X, L, upper=False)[0]
        # Check if I - Z.T @ Z is positive definite
        eigenvalues = torch.linalg.eigvalsh(torch.eye(k, device=device) - Z.T @ Z)
        if torch.any(eigenvalues <= 0):
            raise ValueError("Cholesky downdate would result in a non-positive definite matrix")

    # We'll use a different approach based on the low-rank update formula
    # For A' = A + X@X.T, we can derive L' directly

    # 1. Solve L @ Z = X to get Z
    Z = torch.triangular_solve(X, L, upper=False)[0] # Shape: (n, k)

    # 2. Compute the QR decomposition of Z
    if add:
        # For updates: we augment Z with an identity matrix
        # [Z] [Q1]
        # [I] = Q[Q2] @ R
        I_k = torch.eye(k, device=device)
        Z_augmented = torch.cat([Z, I_k], dim=0) # Shape: (n+k, k)
        Q, R = torch.linalg.qr(Z_augmented, mode='complete')
        Q1 = Q[:n, :k] # Shape: (n, k)
        Q2 = Q[n:, :k] # Shape: (k, k)

        # 3. Compute L' = L @ (I + Q1 @ Q2.T)
        # This is equivalent to L' = L + L @ Q1 @ Q2.T
        L_new = L + L @ Q1 @ Q2.T
    else:
        # For downdates: we need a slightly different approach
        # Compute the Cholesky decomposition of I - Z.T @ Z
        C = torch.linalg.cholesky(torch.eye(k, device=device) - Z.T @ Z)

        # Create a block matrix [Z @ C^-T, L]
        ZCinv = Z @ torch.triangular_solve(torch.eye(k, device=device), C.T, upper=True)[0]

        # QR decomposition of the block matrix
        Q, R = torch.linalg.qr(torch.cat([ZCinv, L], dim=1), mode='reduced')

        # Extract the updated L from R
        L_new = R[:, k:].triu(0).T

    return L_new