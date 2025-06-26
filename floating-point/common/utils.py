from scipy.sparse import random as sparse_random
import grpc
import smpc_pb2
import numpy as np

def matrix_to_proto(matrix):
    """Convert numpy matrix to protobuf Matrix message."""
    if matrix is None:
        return smpc_pb2.Matrix(data=[], rows=0, cols=0)
    
    matrix_flat = matrix.flatten().astype(np.float64)
    return smpc_pb2.Matrix(
        data=matrix_flat.tolist(),
        rows=matrix.shape[0],
        cols=matrix.shape[1]
    )

def proto_to_matrix(proto_matrix):
    """Convert protobuf Matrix message to numpy matrix."""
    if proto_matrix.rows == 0 or proto_matrix.cols == 0:
        return None
    
    data = np.array(proto_matrix.data, dtype=np.float64)
    return data.reshape(proto_matrix.rows, proto_matrix.cols)

def is_invertible(matrix, tolerance=1e-10):
    """Check if a matrix is invertible using condition number."""
    matrix_float = matrix.astype(np.float64)
    cond = np.linalg.cond(matrix_float)
    return cond < 1 / np.finfo(np.float64).eps, cond

def create_shares_float(matrix, adaptive, config):
    """Create floating-point additive secret shares."""
    matrix_64 = matrix.astype(np.float64)
    
    if not adaptive:
        noise_scale = config.get('minimum_noise_range_val', 2.0)
        share_0 = np.random.uniform(-noise_scale, noise_scale, size=matrix_64.shape).astype(np.float64)
        share_1 = np.random.uniform(-noise_scale, noise_scale, size=matrix_64.shape).astype(np.float64)
    else:
        max_vals = np.max(np.abs(matrix_64), axis=0) + 1e-9
        obf_min = config.get('obfuscation_factor_min', 0.1)
        obf_max = config.get('obfuscation_factor_max', 0.5)
        obfuscation = 1 + np.random.uniform(obf_min, obf_max, size=max_vals.shape)
        scale = np.maximum(max_vals * obfuscation, config.get('minimum_noise_range_val', 2.0))
        share_0 = (np.random.rand(*matrix_64.shape) - 0.5) * 2 * scale
        share_1 = (np.random.rand(*matrix_64.shape) - 0.5) * 2 * scale
    
    share_2 = matrix_64 - share_0 - share_1
    return share_0.astype(np.float64), share_1.astype(np.float64), share_2.astype(np.float64)

def reconstruct_shares(shares):
    """Reconstruct the original matrix from additive shares."""
    return shares[0] + shares[1] + shares[2]

def secure_matmul_shares(a_shares, b_shares):
    """
    This function should NOT be used in the secure protocol!
    Each party computes their own portion based on their replicated shares.
    This is kept only for compatibility but should be avoided.
    """
    raise Exception("secure_matmul_shares should not be called - each party computes independently!")

def compute_party_matmul(party_id, a_replicated, b_replicated):
    """
    Compute the matrix multiplication for a specific party in the 3-party protocol.
    
    Args:
        party_id: 1, 2, or 3
        a_replicated: [share_i, share_j] for this party
        b_replicated: [share_i, share_j] for this party
    
    Returns:
        The party's contribution to the secure matrix multiplication
    """
    if party_id == 1:  # Party 0 in 0-indexed
        # Party 0 has (x0, x1) and (y0, y1)
        # Computes: x0⊗y0 + x0⊗y1 + x1⊗y0
        x0, x1 = a_replicated[0], a_replicated[1]
        y0, y1 = b_replicated[0], b_replicated[1]
        return x0 @ y0 + x0 @ y1 + x1 @ y0
        
    elif party_id == 2:  # Party 1 in 0-indexed
        # Party 1 has (x1, x2) and (y1, y2)
        # Computes: x1⊗y1 + x1⊗y2 + x2⊗y1
        x1, x2 = a_replicated[0], a_replicated[1]
        y1, y2 = b_replicated[0], b_replicated[1]
        return x1 @ y1 + x1 @ y2 + x2 @ y1
        
    else:  # Party 2 in 0-indexed (party_id == 3)
        # Party 2 has (x2, x0) and (y2, y0)
        # Computes: x2⊗y2 + x2⊗y0 + x0⊗y2
        x2, x0 = a_replicated[0], a_replicated[1]
        y2, y0 = b_replicated[0], b_replicated[1]
        return x2 @ y2 + x2 @ y0 + x0 @ y2


def calculate_error_metrics(computed, plaintext):
    """Calculate comprehensive error metrics for evaluation."""
    if computed is None:
        return {
            'norm_error': np.nan,
            'max_abs_error': np.nan,
            'mean_abs_error': np.nan,
            'max_rel_error': np.nan,
            'mean_rel_error': np.nan,
            'snr_db': np.nan
        }
    
    computed_64 = computed.astype(np.float64)
    plaintext_64 = plaintext.astype(np.float64)
    error = computed_64 - plaintext_64
    
    # Absolute error metrics
    norm_error = np.linalg.norm(error)
    max_abs_error = np.max(np.abs(error))
    mean_abs_error = np.mean(np.abs(error))
    
    # Relative error metrics with epsilon for numerical stability
    epsilon = 1e-5
    abs_error = np.abs(error)
    abs_plaintext = np.abs(plaintext_64)
    denominator = np.maximum(abs_plaintext, epsilon)
    relative_error = abs_error / denominator
    max_rel_error = np.max(relative_error)
    mean_rel_error = np.mean(relative_error)
    
    # Signal-to-noise ratio
    signal_power = np.mean(plaintext_64**2)
    noise_power = np.mean(error**2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-12 else float('inf')
    
    return {
        'norm_error': float(norm_error),
        'max_abs_error': float(max_abs_error),
        'mean_abs_error': float(mean_abs_error),
        'max_rel_error': float(max_rel_error),
        'mean_rel_error': float(mean_rel_error),
        'snr_db': float(snr_db)
    }

def generate_sparse_matrix(n, density, outlier_config=None):
    """Generate a sparse matrix with optional outliers."""
    # Generate base sparse matrix
    sparse_matrix = sparse_random(
        n, n, 
        density=density, 
        format='dok',
        data_rvs=lambda size: -np.random.rand(size)
    )
    matrix = sparse_matrix.toarray()
    
    # Add outliers if specified
    num_outliers = 0
    if outlier_config and outlier_config.get('introduce_outliers', False):
        probability = outlier_config.get('outlier_probability', 0.1)
        range_min = outlier_config.get('outlier_range_min', -5000)
        range_max = outlier_config.get('outlier_range_max', -2000)
        
        # Create mask for outliers (exclude diagonal)
        mask = (np.random.rand(n, n) < probability) & (np.eye(n) == 0)
        num_outliers = np.sum(mask)
        
        if num_outliers > 0:
            outlier_values = np.random.uniform(range_min, range_max, size=num_outliers)
            matrix[mask] = outlier_values
    
    # Ensure diagonal is 1 for invertibility
    np.fill_diagonal(matrix, 1)
    
    return matrix.astype(np.float64), num_outliers

def generate_orthogonal_matrix(n, scaling_factor=10000, max_condition=10):
    """Generate an orthogonal matrix with good conditioning."""
    max_attempts = 100
    
    for _ in range(max_attempts):
        # Generate random matrix and perform QR decomposition
        z = np.random.randn(n, n)
        q, _ = np.linalg.qr(z)
        
        # Scale and convert to integer, then back to float
        t_int = np.round(q * scaling_factor).astype(np.int64)
        t_float = t_int.astype(np.float64)
        
        # Check condition number
        condition_number = np.linalg.cond(t_float)
        
        if condition_number < 1 / np.finfo(np.float64).eps and condition_number < max_condition:
            return t_float, float(condition_number)
    
    raise Exception(f"Failed to generate well-conditioned orthogonal matrix after {max_attempts} attempts")

def validate_security_properties(party_shares_dict):
    """
    Validate that security properties are maintained.
    
    Args:
        party_shares_dict: Dictionary mapping party_id -> list of matrices they have
    
    Raises:
        Exception if security properties are violated
    """
    total_parties = len(party_shares_dict)
    
    if total_parties != 3:
        raise Exception(f"Expected 3 parties, got {total_parties}")
    
    # Check that no party has more than 2 shares of any secret
    for party_id, shares in party_shares_dict.items():
        if len(shares) > 2:
            raise Exception(f"Party {party_id} has {len(shares)} shares - security violation!")
    
    print("✅ Security properties validated: No party has more than 2 shares")

def check_reconstruction_correctness(original_matrix, reconstructed_shares):
    """
    Verify that the reconstruction is mathematically correct.
    
    Args:
        original_matrix: The original secret matrix
        reconstructed_shares: List of 3 additive shares
    
    Returns:
        bool: True if reconstruction is correct
    """
    if len(reconstructed_shares) != 3:
        raise Exception(f"Expected 3 shares for reconstruction, got {len(reconstructed_shares)}")
    
    reconstructed = reconstructed_shares[0] + reconstructed_shares[1] + reconstructed_shares[2]
    
    # Check if reconstruction matches original (within numerical precision)
    max_error = np.max(np.abs(original_matrix - reconstructed))
    
    if max_error < 1e-10:
        print("✅ Reconstruction is mathematically correct")
        return True
    else:
        print(f"❌ Reconstruction error: {max_error}")
        return False
   