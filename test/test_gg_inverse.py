import torch
import pytest
import numpy as np

from heavyball.utils import _gg_inverse_via_newtonschulz, eye_like, to_triu, triu_to_line, promote

# Tolerances for the Frobenius norm error
TOLERANCES = {
    torch.float32: 1e-1,
    torch.float64: 1e-3,
    torch.bfloat16: 0.6, # bfloat16 can have higher error
    torch.float16: 0.8   # float16 also can have higher error
}

def generate_spd_matrix(size: int, dtype: torch.dtype, device: torch.device, condition_number: float = 10.0):
    """
    Creates a symmetric positive definite matrix G.
    """
    if condition_number < 1.0:
        condition_number = 1.0 # Ensure condition number is at least 1

    # Generate diagonal matrix D with specified condition number
    # Singular values range from x to x * condition_number.
    # To make it simple, let's make them range from 1 to condition_number.
    # If size is 1, linspace might not behave as expected for start=end, so handle it.
    if size == 1:
        D_diag = torch.tensor([1.0 * condition_number], device=device, dtype=dtype) # Or just [1.0]
    else:
        D_diag = torch.linspace(1.0, condition_number, size, device=device, dtype=dtype)

    D = torch.diag(D_diag)

    # Generate a random matrix A
    A = torch.randn(size, size, device=device, dtype=dtype)

    # Get an orthogonal matrix U from A using QR decomposition
    U, _ = torch.linalg.qr(A)

    # Construct G = U @ D @ U.T
    # Ensure G is symmetric: G = (G + G.T) / 2
    # Promoting to float32 for matrix multiplications if original dtype is lower precision, then casting back
    intermediate_dtype = dtype
    if dtype == torch.float16 or dtype == torch.bfloat16:
        intermediate_dtype = torch.float32
        U = U.to(intermediate_dtype)
        D = D.to(intermediate_dtype)

    G = U @ D @ U.T
    G = (G + G.T) / 2
    return G.to(dtype)


@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("condition_number", [10.0, 100.0])
@pytest.mark.parametrize("dtype_str", ["float32", "float64", "bfloat16"]) # Added bfloat16, float16 later if needed
@pytest.mark.parametrize("inverse_order", [5, 10, 15]) # Increased orders
@pytest.mark.parametrize("precond_lr_val", [0.1, 0.5])
@pytest.mark.parametrize("reg_lambda_val", [1e-6, 1e-4, 0.0]) # Added 0.0 for no regularization
def test_gg_inverse_iteration(size, condition_number, dtype_str, inverse_order, precond_lr_val, reg_lambda_val):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bfloat16 is poorly supported on CPU for this test's operations (e.g. QR)")
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU architecture")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = generate_spd_matrix(size, dtype=dtype, device=device, condition_number=condition_number)

    # Initialize Q as a list of tensors (as expected by _gg_inverse_via_newtonschulz)
    # Using eye_like(G) which should be a full matrix.
    # _gg_inverse_via_newtonschulz uses to_triu(oq, True) internally, so Q_initial_list
    # can be a list of full matrices or line representations.
    # For simplicity, starting with full matrices.
    Q_initial_list = [eye_like(G)]

    # Ensure precond_lr is a scalar tensor with the same dtype and device as G
    precond_lr_tensor = torch.tensor(precond_lr_val, dtype=G.dtype, device=G.device)

    # Call the function under test
    # G is cloned because some internal operations might modify it if not careful, though it's passed as G16.
    # Q_initial_list is modified in-place.
    _gg_inverse_via_newtonschulz(
        G.clone(),
        Q_initial_list,
        inverse_order=inverse_order,
        precond_lr=precond_lr_tensor,
        eps=1e-7, # Default from function if not specified, but good to be explicit
        norm_eps=1e-7, # Default
        min_update_step=1e-7, # Default
        svd_power_iter=1, # Default
        max_grad_norm=0.01, # Default
        regularization_lambda=reg_lambda_val
    )

    Q_final = Q_initial_list[0]

    # Compute G_inv_computed @ G
    # We expect Q_final to be an approximation of G_inv_sqrt, so Q_final @ Q_final.T ~ G_inv
    # The function _gg_inverse_via_newtonschulz is meant to return Q such that Q approximates G^(-1/2) or G^(-1)
    # based on its update rule. The Newton-Schulz for inverse is X_k+1 = X_k (2I - A X_k).
    # If Q is an estimate of G_inv, then Q_final @ G should be I.
    # Let's assume Q_final is G_inv.

    identity_computed = torch.matmul(Q_final.to(torch.float32), G.to(torch.float32)) # Promote for precision in check
    identity_target = torch.eye(size, dtype=torch.float32, device=device) # Compare in float32

    error = torch.norm(identity_computed - identity_target, p='fro')

    print(f"Test params: size={size}, cond={condition_number}, dtype={dtype_str}, order={inverse_order}, lr={precond_lr_val}, reg={reg_lambda_val}, Error: {error.item()}")

    # Adjust tolerance based on dtype
    tol = TOLERANCES.get(dtype, 1e-1) # Default tolerance if dtype not in map

    # Increase tolerance for high condition numbers or low precision
    if condition_number > 50:
        tol *= 2
    if dtype == torch.float16 or dtype == torch.bfloat16:
        tol *= 2 # Additional factor for low precision types

    assert error.item() < tol, f"Error {error.item()} exceeds tolerance {tol}"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

# Example of how to run this test:
# pytest test_gg_inverse.py -v -s
# -s is to show print statements
# Add -k "float32" to run only float32 tests initially
