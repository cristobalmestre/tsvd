import torch

# Step 1: Define A and compute L
A = torch.tensor([[4.0, 2.0],
                  [2.0, 3.0]])

L = torch.linalg.cholesky(A)

# Step 2: Define z
z = torch.tensor([[1.0], [2.0]])  # Shape: (2, 1)

# Step 3: Manual Cholesky of updated A
A_updated = A + z @ z.T
L_updated_manual = torch.linalg.cholesky(A_updated)

# Step 4: Use your cholesky_update_vectorized function
def cholesky_update_vectorized(L, X, add=True):
    Z = torch.triangular_solve(X, L, upper=False)[0]
    if add:
        I_k = torch.eye(Z.shape[1])
        Z_aug = torch.cat([Z, I_k], dim=0)
        Q, R = torch.linalg.qr(Z_aug, mode='complete')
        Q1 = Q[:L.shape[0], :X.shape[1]]
        Q2 = Q[L.shape[0]:, :X.shape[1]]
        return L + L @ Q1 @ Q2.T
    else:
        raise NotImplementedError("Only update implemented")

L_updated_code = cholesky_update_vectorized(L, z)

# Compare results
print("Original L:\n", L)
print("Manual L Updated:\n", L_updated_manual)
print("Code L Updated:\n", L_updated_code)
print("Difference:\n", L_updated_manual - L_updated_code)