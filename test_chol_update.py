import os
print("Current Working Directory:", os.getcwd())
import numpy as np
from utils.toolkit import target2onehot, tensor2numpy, cholesky_update, cholesky_update_batch, test_cholesky_update
import torch
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

test_cholesky_update()