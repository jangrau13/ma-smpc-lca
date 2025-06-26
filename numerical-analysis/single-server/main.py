#!/usr/bin/env python3
"""
Enhanced SMPC-LCA Simulation
with Automatic GPU Detection and CPU Fallback on a single server.

Features:
- Automatic NVIDIA GPU detection with CUDA version checking
- Automatic CuPy installation for GPU acceleration
- Seamless fallback to NumPy for CPU-only systems
- Fixed-point and floating-point secure computation protocols
- Comprehensive error handling and memory management
"""

import sys
import subprocess
import re
import importlib.util
import importlib
import numpy as np
import pandas as pd
from scipy.sparse import random as sparse_random
import time
from itertools import product
import os
import hashlib
import gc

print("=== Enhanced Multi-Party Computation Simulation with GPU Support ===")
print("--- Performing GPU capability assessment ---")

# =================================================================================
# --- GPU DETECTION AND ENVIRONMENT SETUP ---
# =================================================================================

def detect_gpu_vendor() -> str:
    """
    Detect the GPU vendor (NVIDIA or unknown) by checking for nvidia-smi.
    
    Returns:
        str: GPU vendor identifier ('nvidia' or 'unknown')
    """
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return "nvidia"
    except FileNotFoundError:
        pass
    return "unknown"

def get_cuda_version() -> str:
    """
    Retrieve the CUDA version from nvcc or nvidia-smi.
    
    Returns:
        str: CUDA version string or None if not found
    """
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
        
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass
    return None

def get_cupy_package_name(cuda_version: str) -> str:
    """
    Determine the appropriate CuPy package name based on CUDA version.
    
    Args:
        cuda_version: CUDA version string
        
    Returns:
        str: CuPy package name or None if unsupported
    """
    try:
        version_float = float(cuda_version)
        if version_float >= 12.0:
            return "cupy-cuda12x"
        elif version_float >= 11.0:
            return "cupy-cuda11x"
        else:
            return None
    except (ValueError, TypeError):
        return None

def install_package(package_name: str) -> bool:
    """
    Install a Python package using pip with comprehensive error handling.
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        bool: True if installation successful, False otherwise
    """
    print(f"Installing {package_name}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        print(f"{package_name} installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}.")
        print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("'pip' command not found. Please ensure Python and pip are correctly installed.")
        return False

def setup_computation_environment():
    """
    Setup computational environment with GPU detection and CPU fallback.
    
    Returns:
        Tuple[Any, bool]: (compute_module, is_gpu_mode)
    """
    # Try to setup GPU computation with CuPy
    gpu_vendor = detect_gpu_vendor()
    if gpu_vendor == "nvidia":
        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None:
            print("CuPy not found. Attempting installation...")
            cuda_version = get_cuda_version()
            if cuda_version:
                print(f"Found NVIDIA GPU with CUDA {cuda_version}")
                package_to_install = get_cupy_package_name(cuda_version)
                if package_to_install:
                    if install_package(package_to_install):
                        print("Invalidating import caches to load CuPy...")
                        importlib.invalidate_caches()
                    else:
                        print("CuPy installation failed. Falling back to CPU mode.")
                        return np, False
                else:
                    print("Unsupported CUDA version. Falling back to CPU mode.")
                    return np, False
            else:
                print("CUDA not detected. Falling back to CPU mode.")
                return np, False
        
        # Try to import CuPy
        try:
            cp = importlib.import_module('cupy')
            print("CuPy loaded successfully. GPU mode enabled.")
            return cp, True
        except ImportError as e:
            print(f"CuPy import failed: {e}. Falling back to CPU mode.")
            return np, False
    else:
        print("No NVIDIA GPU detected. Using CPU mode.")
        return np, False

# Setup computational environment
cp, IS_GPU_MODE = setup_computation_environment()

print(f"🔧 Computation mode: {'GPU (CuPy)' if IS_GPU_MODE else 'CPU (NumPy)'}")

def clear_memory():
    """Clear memory pools and perform garbage collection."""
    if IS_GPU_MODE:
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass
    gc.collect()

def sync_computation():
    """Synchronize computation (GPU streams or CPU no-op)."""
    if IS_GPU_MODE:
        cp.cuda.Stream.null.synchronize()

# =================================================================================
# --- EXPERIMENT CONFIGURATION ---
# =================================================================================

param_grid = {
    'N': [100, 250],                           # Matrix dimension
    'R': [100, 250],                                # Row dimension for the secret matrix B
    'INTRODUCE_OUTLIERS': [False, True],                  # Toggle for adding outliers
    'OUTLIER_PROBABILITY': [0.01],                        # Probability of an element being an outlier
    'OUTLIER_RANGE': [[-1000, -100], [-5000, -2000]],      # Outlier value range [min, max]
    'B_INT_RANGE_SCALE': [[1000, 5000], [10000, 50000]],   # Range for the integer secret B matrix [-x, x]
    'PRECISION': [20, 25, 30, 35],                        # Bits for fixed-point fractional part
    'USE_ADAPTIVE_SHARING': [True],                       # Toggle for adaptive noise in float method
    'MINIMUM_NOISE_RANGE_VAL': [2],                       # Minimum noise scale for security
    'OBFUSCATION_FACTOR_RANGE': [[0.1, 0.5]],              # Range for random scaling in adaptive mode
    'A_DENSITY': [0.01, 0.1, 0.5],                        # Matrix density for A
}

# --- Global Settings ---
CSV_RESULTS_FILE = f'{"gpu" if IS_GPU_MODE else "cpu"}-results.csv'
RUN_ADVANCED_ANALYSIS = True
T_SCALING_FACTOR = 10000
MAX_COND_T = 10

# =================================================================================
# --- HELPER & CORE LOGIC FUNCTIONS ---
# =================================================================================

def is_invertible(m):
    """Checks if a matrix is invertible using its condition number."""
    # Convert to NumPy for condition number calculation
    if IS_GPU_MODE and hasattr(m, 'get'):
        m_cpu = m.get().astype(np.float64)
    else:
        m_cpu = m.astype(np.float64)
    
    cond = np.linalg.cond(m_cpu)
    return cond < 1 / np.finfo(np.float64).eps, cond

def create_shares(matrix):
    """
    Creates three int64 additive shares for a given matrix using correct
    modular arithmetic over the Z_2^64 ring.
    """
    compute_lib = cp if IS_GPU_MODE else np
    
    ring_size = 2**64
    share0 = compute_lib.random.randint(0, ring_size, size=matrix.shape, dtype=compute_lib.uint64)
    share1 = compute_lib.random.randint(0, ring_size, size=matrix.shape, dtype=compute_lib.uint64)
    # Automatic modular arithmetic (wraps around)
    share2 = (matrix.astype(compute_lib.uint64) - share0 - share1)
    return share0.astype(compute_lib.int64), share1.astype(compute_lib.int64), share2.astype(compute_lib.int64)

def reconstruct(shares):
    """Reconstructs the original matrix from its three shares."""
    compute_lib = cp if IS_GPU_MODE else np
    return shares[0].astype(compute_lib.uint64) + shares[1].astype(compute_lib.uint64) + shares[2].astype(compute_lib.uint64)

def secure_matmul(a_shares, t_shares):
    """Performs matrix multiplication on shared matrices using the Beaver triplet strategy."""
    compute_lib = cp if IS_GPU_MODE else np
    
    a0, a1, a2 = [compute_lib.asarray(s) for s in a_shares]
    t0, t1, t2 = [compute_lib.asarray(s) for s in t_shares]
    p1 = a0 @ t0 + a0 @ t1 + a1 @ t0
    p2 = a1 @ t1 + a1 @ t2 + a2 @ t1
    p3 = a2 @ t2 + a2 @ t0 + a0 @ t2
    return p1, p2, p3

def secure_local_diag(vector_shares):
    """Each party locally turns its share of a vector into a diagonal matrix."""
    compute_lib = cp if IS_GPU_MODE else np
    
    s0, s1, s2 = [compute_lib.asarray(s) for s in vector_shares]
    return compute_lib.diag(s0.flatten()), compute_lib.diag(s1.flatten()), compute_lib.diag(s2.flatten())

def calculate_error_metrics(computed, plaintext):
    """Calculates and returns a dictionary of error metrics."""
    if computed is None:
        return {'norm_error': np.nan, 'max_abs_error': np.nan, 'mean_abs_error': np.nan, 'snr_db': np.nan}
    
    # Convert to NumPy for error calculations
    if IS_GPU_MODE:
        if hasattr(computed, 'get'):
            computed_cpu = computed.get().astype(np.float64)
        else:
            computed_cpu = computed.astype(np.float64)
        if hasattr(plaintext, 'get'):
            plaintext_cpu = plaintext.get().astype(np.float64)
        else:
            plaintext_cpu = plaintext.astype(np.float64)
    else:
        computed_cpu = computed.astype(np.float64)
        plaintext_cpu = plaintext.astype(np.float64)
    
    err = computed_cpu - plaintext_cpu
    max_abs_err = np.max(np.abs(err))
    mean_abs_err = np.mean(np.abs(err))
    p_sig = np.mean(plaintext_cpu**2)
    p_noise = np.mean(err**2)
    snr = 10 * np.log10(p_sig / p_noise) if p_noise > 1e-12 else float('inf')
    
    return {
        'max_abs_error': float(max_abs_err), 
        'mean_abs_error': float(mean_abs_err), 
        'snr_db': float(snr)
    }

def secure_randomize_shares(shares):
    """Randomizes existing shares without changing the secret."""
    compute_lib = cp if IS_GPU_MODE else np
    
    s0, s1, s2 = [compute_lib.asarray(s) for s in shares]
    r0_uint = compute_lib.random.randint(0, 2**62, size=s0.shape, dtype=compute_lib.uint64)
    r1_uint = compute_lib.random.randint(0, 2**62, size=s1.shape, dtype=compute_lib.uint64)
    s0_uint, s1_uint, s2_uint = [s.astype(compute_lib.uint64) for s in (s0, s1, s2)]
    new_s0 = s0_uint - r0_uint
    new_s1 = s1_uint - r1_uint
    new_s2 = s2_uint + r0_uint + r1_uint
    return new_s0.astype(compute_lib.int64), new_s1.astype(compute_lib.int64), new_s2.astype(compute_lib.int64)

def create_shares_float(matrix, adaptive, config):
    """Creates float shares."""
    compute_lib = cp if IS_GPU_MODE else np
    
    m64 = matrix.astype(compute_lib.float64)
    if not adaptive:
        noise_scale = config['MINIMUM_NOISE_RANGE_VAL']
        s0 = compute_lib.random.uniform(-noise_scale, noise_scale, size=m64.shape).astype(compute_lib.float64)
        s1 = compute_lib.random.uniform(-noise_scale, noise_scale, size=m64.shape).astype(compute_lib.float64)
    else:
        maxs = compute_lib.max(compute_lib.abs(m64), axis=0) + 1e-9
        obfs = 1 + compute_lib.random.uniform(*config['OBFUSCATION_FACTOR_RANGE'], size=maxs.shape)
        scale = compute_lib.maximum(maxs * obfs, config['MINIMUM_NOISE_RANGE_VAL'])
        s0 = (compute_lib.random.rand(*m64.shape) - 0.5) * 2 * scale
        s1 = (compute_lib.random.rand(*m64.shape) - 0.5) * 2 * scale
    return s0.astype(compute_lib.float64), s1.astype(compute_lib.float64), m64 - s0 - s1

def reconstruct_float(shares):
    """Reconstructs float shares."""
    return shares[0] + shares[1] + shares[2]

def secure_randomize_shares_float(shares, adaptive, config):
    """Randomizes float shares."""
    compute_lib = cp if IS_GPU_MODE else np
    
    s0, s1, s2 = [compute_lib.asarray(s) for s in shares]
    if adaptive:
        sc0 = (compute_lib.max(compute_lib.abs(s0), axis=0) + 1e-9) / 3.0
        sc1 = (compute_lib.max(compute_lib.abs(s1), axis=0) + 1e-9) / 3.0
        r0 = (compute_lib.random.rand(*s0.shape) - 0.5) * 2 * sc0
        r1 = (compute_lib.random.rand(*s1.shape) - 0.5) * 2 * sc1
    else:
        noise = config['MINIMUM_NOISE_RANGE_VAL'] / 3.0
        r0 = compute_lib.random.uniform(-noise, noise, size=s0.shape)
        r1 = compute_lib.random.uniform(-noise, noise, size=s1.shape)
    return s0 - r0.astype(compute_lib.float64), s1 - r1.astype(compute_lib.float64), s2 + r0.astype(compute_lib.float64) + r1.astype(compute_lib.float64)

# =================================================================================
# --- SIMULATION EXECUTION FUNCTIONS ---
# =================================================================================

def run_fixed_point_simulation(A_float, T_int, B_int, f_int, plaintext_result, config):
    """
    Runs the fixed-point simulation with automatic GPU/CPU library selection.
    """
    start_time = time.time()
    compute_lib = cp if IS_GPU_MODE else np
    scale_factor = 2**config['PRECISION']

    # Convert matrices to appropriate library arrays
    if IS_GPU_MODE:
        A_float = cp.asarray(A_float)
        T_int = cp.asarray(T_int)
        B_int = cp.asarray(B_int)
        f_int = cp.asarray(f_int)
        plaintext_result = cp.asarray(plaintext_result)

    A_fix = (A_float * scale_factor).astype(compute_lib.int64)
    a_shares = create_shares(A_fix)
    t_shares = create_shares(T_int)
    b_shares = create_shares(B_int)
    f_shares = create_shares(f_int)

    try:
        at_shares = secure_matmul(a_shares, t_shares)
        AT_reconstructed = reconstruct(at_shares)
        AT_recon_float = AT_reconstructed.astype(compute_lib.int64).astype(compute_lib.float64) / scale_factor
        
        # Matrix inversion - convert to CPU for NumPy if needed
        if IS_GPU_MODE:
            AT_inv_float = cp.linalg.inv(AT_recon_float)
        else:
            AT_inv_float = np.linalg.inv(AT_recon_float)

        at_inv_shares = create_shares((AT_inv_float * scale_factor).astype(compute_lib.int64))
        a_inv_shares = secure_matmul(t_shares, at_inv_shares)
        a_inv_rand_shares = secure_randomize_shares(a_inv_shares)

        s_shares = secure_matmul(a_inv_rand_shares, f_shares)
        s_rand_shares = secure_randomize_shares(s_shares)
        diag_s_shares = secure_local_diag(s_rand_shares)
        final_shares = secure_matmul(b_shares, diag_s_shares)

        final_computed_fixed = reconstruct(final_shares).astype(compute_lib.int64).astype(compute_lib.float64) / scale_factor

        sync_computation()

    except Exception as e:
        print(f"  [WARNING] Fixed-point simulation failed: {e}")
        final_computed_fixed = None
    finally:
        clear_memory()

    duration = time.time() - start_time
    results = {'fixed_point_time_s': duration}
    results.update({f'fixed_{k}': v for k, v in calculate_error_metrics(final_computed_fixed, plaintext_result).items()})
    return results

def run_floating_point_simulation(A_float, T_float, B_float, f_float, plaintext_result, inv_plaintext, config):
    """Run floating point simulation with automatic GPU/CPU library selection."""
    start_time = time.time()
    compute_lib = cp if IS_GPU_MODE else np
    
    # Convert matrices to appropriate library arrays
    if IS_GPU_MODE:
        A_float64 = cp.asarray(A_float).astype(cp.float64)
        T_float64 = cp.asarray(T_float).astype(cp.float64)
        B_float64 = cp.asarray(B_float).astype(cp.float64)
        f_float64 = cp.asarray(f_float).astype(cp.float64)
        plaintext_result64 = cp.asarray(plaintext_result).astype(cp.float64)
        inv_plaintext64 = cp.asarray(inv_plaintext).astype(cp.float64)
    else:
        A_float64 = A_float.astype(np.float64)
        T_float64 = T_float.astype(np.float64)
        B_float64 = B_float.astype(np.float64)
        f_float64 = f_float.astype(np.float64)
        plaintext_result64 = plaintext_result.astype(np.float64)
        inv_plaintext64 = inv_plaintext.astype(np.float64)
    
    share_creator = lambda m: create_shares_float(m, adaptive=config['USE_ADAPTIVE_SHARING'], config=config)
    leakage_metrics = {}

    try:
        a_shares = share_creator(A_float64)
        t_shares = share_creator(T_float64)

        at_shares = secure_matmul(a_shares, t_shares)
        AT_reconstructed = reconstruct_float(at_shares)
        
        # Matrix inversion
        if IS_GPU_MODE:
            AT_inv_computed = cp.linalg.inv(AT_reconstructed)
        else:
            AT_inv_computed = np.linalg.inv(AT_reconstructed)

        at_inv_shares = share_creator(AT_inv_computed)
        a_inv_shares = secure_matmul(t_shares, at_inv_shares)
        a_inv_rand_shares = secure_randomize_shares_float(a_inv_shares, adaptive=config['USE_ADAPTIVE_SHARING'], config=config)

        b_shares = share_creator(B_float64)
        f_shares = share_creator(f_float64)

        s_shares = secure_matmul(a_inv_rand_shares, f_shares)
        s_rand_shares = secure_randomize_shares_float(s_shares, adaptive=config['USE_ADAPTIVE_SHARING'], config=config)

        diag_s_shares = secure_local_diag(s_rand_shares)
        final_shares = secure_matmul(b_shares, diag_s_shares)

        final_computed_float = reconstruct_float(final_shares)

        sync_computation()

    except Exception as e:
        print(f"  [WARNING] Floating-point simulation failed: {e}")
        final_computed_float = None
    finally:
        clear_memory()

    duration = time.time() - start_time
    results = {'float_point_time_s': duration}
    results.update({f'float_{k}': v for k, v in calculate_error_metrics(final_computed_float, plaintext_result64).items()})
    results.update(leakage_metrics)
    return results

# =================================================================================
# --- MAIN EXECUTION DRIVER ---
# =================================================================================

def main():
    if os.path.exists(CSV_RESULTS_FILE):
        results_df = pd.read_csv(CSV_RESULTS_FILE)
    else:
        results_df = pd.DataFrame()

    keys, values = zip(*param_grid.items())
    experiment_configs = [dict(zip(keys, v)) for v in product(*values)]

    print(f"Found {len(experiment_configs)} total experiment configurations.")
    if not results_df.empty:
        print(f"Found {len(results_df)} existing results in {CSV_RESULTS_FILE}.")

    for i, config in enumerate(experiment_configs):
        config_str = str(sorted(config.items()))
        run_id = hashlib.md5(config_str.encode()).hexdigest()
        seed = int(run_id, 16) % (2**32)
        config['run_id'] = run_id
        config['seed'] = seed

        if not results_df.empty and run_id in results_df['run_id'].values:
            continue

        print(f"\n--- Running Experiment {i+1}/{len(experiment_configs)} (ID: {run_id[:8]}) ---")
        param_str = '; '.join([f"{k}: {v}" for k, v in config.items() if k not in ['run_id', 'seed', 'OBFUSCATION_FACTOR_RANGE', 'OUTLIER_RANGE']])
        print(f"  {param_str}")

        # Set seed for random number generation
        if IS_GPU_MODE:
            cp.random.seed(seed)
        np.random.seed(seed)

        N, R = config['N'], config['R']
        
        # Generate sparse matrix
        s = sparse_random(N, N, density=config['A_DENSITY'], format='dok', data_rvs=lambda size: -np.random.rand(size))
        A_float = s.toarray()

        num_outliers = 0
        if config['INTRODUCE_OUTLIERS']:
            mask = (np.random.rand(N, N) < config['OUTLIER_PROBABILITY']) & (np.eye(N) == 0)
            num_outliers = np.sum(mask)
            A_float[mask] = np.random.uniform(*config['OUTLIER_RANGE'], size=num_outliers)

        np.fill_diagonal(A_float, 1)

        is_inv, cond_A = is_invertible(A_float)
        if not is_inv:
            print("  [SKIPPING] Generated matrix A is not invertible.")
            continue

        # Generate orthogonal matrix T
        while True:
            z = np.random.randn(N, N)
            q, _ = np.linalg.qr(z)
            T_int = np.round(q * T_SCALING_FACTOR).astype(np.int64)
            cond_t = np.linalg.cond(T_int.astype(np.float64))
            
            if cond_t < 1 / np.finfo(np.float64).eps and cond_t < MAX_COND_T:
                break

        # Generate other matrices
        B_int = np.random.randint(*config['B_INT_RANGE_SCALE'], size=(R, N), dtype=np.int64)
        f_int = np.zeros((N, 1), dtype=np.int64)
        f_int[0, 0] = 1

        T_float = T_int.astype(np.float64)
        B_float = B_int.astype(np.float64)
        f_float = f_int.astype(np.float64)

        # Compute plaintext results
        A_inv_plaintext = np.linalg.inv(A_float)
        s_plaintext_vec = A_inv_plaintext @ f_float
        diag_s_plaintext = np.diag(s_plaintext_vec.flatten())
        final_plaintext_result = B_float @ diag_s_plaintext

        run_metrics = config.copy()
        run_metrics['computation_mode'] = 'GPU' if IS_GPU_MODE else 'CPU'
        run_metrics['matrix_cond_A'] = float(cond_A)
        run_metrics['matrix_cond_T'] = float(cond_t)
        run_metrics['num_outliers_injected'] = num_outliers

        run_metrics['obfuscation_range_min'] = config['OBFUSCATION_FACTOR_RANGE'][0]
        run_metrics['obfuscation_range_max'] = config['OBFUSCATION_FACTOR_RANGE'][1]
        run_metrics['outlier_range_min'] = config['OUTLIER_RANGE'][0]
        run_metrics['outlier_range_max'] = config['OUTLIER_RANGE'][1]
        del run_metrics['OBFUSCATION_FACTOR_RANGE']
        del run_metrics['OUTLIER_RANGE']

        # Run simulations
        fixed_results = run_fixed_point_simulation(A_float, T_int, B_int, f_int, final_plaintext_result, config)
        run_metrics.update(fixed_results)

        float_results = run_floating_point_simulation(A_float, T_float, B_float, f_float, final_plaintext_result, A_inv_plaintext, config)
        run_metrics.update(float_results)

        new_row = pd.DataFrame([run_metrics])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(CSV_RESULTS_FILE, index=False)
        print(f"  Experiment complete. Results saved to {CSV_RESULTS_FILE}.")

    print(f"\nAll experiments finished. Results saved to {CSV_RESULTS_FILE}")
    print(f"Computation mode used: {'GPU (CuPy)' if IS_GPU_MODE else 'CPU (NumPy)'}")

if __name__ == "__main__":
    main()