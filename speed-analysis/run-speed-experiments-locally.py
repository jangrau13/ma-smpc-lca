#!/usr/bin/env python3
"""
===============================================================================
Comprehensive Matrix Operations & Network Performance Benchmark Suite (CPU/GPU)
===============================================================================

A unified benchmarking framework for evaluating matrix operations performance
on both CPU (NumPy) and GPU (CuPy) platforms, plus network performance metrics.
This suite provides comprehensive performance analysis with automatic fallback 
to CPU when GPU acceleration is unavailable.

Features:
- Automatic GPU detection with CPU fallback
- Comprehensive operation coverage (multiplication, addition, inverse, random generation)
- Network performance testing (HTTP latency and download speed)
- Performance metrics calculation (GFLOPS, throughput, Mbps, response times)
- Stateful execution with progress saving/resuming
- Cross-platform compatibility with detailed system reporting
"""

import sys
import subprocess
import re
import platform
import importlib.util
import importlib
import os
import gc
import json
import time
import socket
import urllib.request
import urllib.error
import ssl
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings

print("=== Comprehensive Matrix Operations & Network Performance Benchmark Suite (CPU/GPU) ===")
print("--- Performing system capability assessment ---")

# ===============================================================================
# SECTION 1: SYSTEM DETECTION AND DEPENDENCY MANAGEMENT
# ===============================================================================

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

def get_cuda_version() -> Optional[str]:
    """
    Retrieve the CUDA version from nvcc or nvidia-smi.
    
    Returns:
        Optional[str]: CUDA version string or None if not found
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

def get_cupy_package_name(cuda_version: str) -> Optional[str]:
    """
    Determine the appropriate CuPy package name based on CUDA version.
    
    Args:
        cuda_version: CUDA version string
        
    Returns:
        Optional[str]: CuPy package name or None if unsupported
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
        print(f"✅ {package_name} installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}.")
        print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ 'pip' command not found. Please ensure Python and pip are correctly installed.")
        return False

def setup_computation_environment():
    """
    Setup computational environment with GPU detection and CPU fallback.
    
    Returns:
        Tuple[Any, Any, bool]: (compute_module, numpy_module, is_gpu_mode)
    """
    # Always ensure NumPy is available
    try:
        np = importlib.import_module('numpy')
        print("✅ NumPy loaded successfully.")
    except ImportError:
        print("❌ NumPy not found. Installing...")
        if install_package("numpy"):
            np = importlib.import_module('numpy')
        else:
            raise RuntimeError("Failed to install NumPy. Cannot continue.")
    
    # Try to setup GPU computation with CuPy
    gpu_vendor = detect_gpu_vendor()
    if gpu_vendor == "nvidia":
        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None:
            print("❌ CuPy not found. Attempting installation...")
            cuda_version = get_cuda_version()
            if cuda_version:
                print(f"Found NVIDIA GPU with CUDA {cuda_version}")
                package_to_install = get_cupy_package_name(cuda_version)
                if package_to_install:
                    if install_package(package_to_install):
                        print("Invalidating import caches to load CuPy...")
                        importlib.invalidate_caches()
                    else:
                        print("⚠️ CuPy installation failed. Falling back to CPU mode.")
                        return np, np, False
                else:
                    print("⚠️ Unsupported CUDA version. Falling back to CPU mode.")
                    return np, np, False
            else:
                print("⚠️ CUDA not detected. Falling back to CPU mode.")
                return np, np, False
        
        # Try to import CuPy
        try:
            cp = importlib.import_module('cupy')
            print("✅ CuPy loaded successfully. GPU mode enabled.")
            return cp, np, True
        except ImportError as e:
            print(f"⚠️ CuPy import failed: {e}. Falling back to CPU mode.")
            return np, np, False
    else:
        print("⚠️ No NVIDIA GPU detected. Using CPU mode.")
        return np, np, False

# Setup computational environment
cp, np, IS_GPU_MODE = setup_computation_environment()

print(f"🔧 Computation mode: {'GPU (CuPy)' if IS_GPU_MODE else 'CPU (NumPy)'}")

# ===============================================================================
# SECTION 2: MEMORY MANAGEMENT AND ERROR HANDLING
# ===============================================================================

def get_memory_info() -> Tuple[int, int]:
    """
    Retrieve current memory usage information (GPU or system RAM).
    
    Returns:
        Tuple[int, int]: (used_bytes, total_bytes)
    """
    if IS_GPU_MODE:
        try:
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            _, total = device.mem_info
            used = mempool.used_bytes()
            return used, total
        except Exception as e:
            print(f"  Warning: Could not retrieve GPU memory info: {e}")
            return 0, 16 * 1024**3
    else:
        # For CPU mode, we'll use a simple heuristic based on available system RAM
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.used, memory.total
        except ImportError:
            # Fallback estimate
            return 0, 16 * 1024**3

def clear_memory():
    """
    Clear memory pools and perform garbage collection.
    """
    if IS_GPU_MODE:
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass
    gc.collect()

def reconstruct_memory_pools() -> bool:
    """
    Reconstruct memory pools after out-of-memory errors (GPU only).
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not IS_GPU_MODE:
        return True
    
    try:
        clear_memory()
        print("  🔄 Reconstructing GPU memory pools after OOM...")
        mempool = cp.get_default_memory_pool()
        total_memory = cp.cuda.Device().mem_info[1]
        mempool.set_limit(size=int(total_memory * 0.9))
        print(f"  ✅ Memory pool reconstructed with limit: {total_memory * 0.9 / 1e9:.2f} GB")
        return True
    except Exception as e:
        print(f"  ❌ Failed to reconstruct memory pools: {e}")
        return False

def handle_fatal_error(error_msg: str, operation_desc: str, benchmark_data: Dict, json_filename: str) -> bool:
    """
    Handle fatal errors by saving progress and terminating safely.
    
    Args:
        error_msg: Error message text
        operation_desc: Description of the operation that failed
        benchmark_data: Current benchmark data
        json_filename: Output file for saving progress
        
    Returns:
        bool: False (always, as this indicates fatal error)
    """
    if IS_GPU_MODE:
        cuda_error_keywords = [
            "cudaErrorIllegalAddress", "illegal memory access",
            "cudaErrorLaunchFailure", "launch failure",
            "CUDARuntimeError", "cudaErrorHardware"
        ]
        is_fatal_error = any(keyword in error_msg for keyword in cuda_error_keywords)
    else:
        # For CPU mode, consider system-level memory errors as potentially fatal
        is_fatal_error = "MemoryError" in error_msg or "SystemError" in error_msg
    
    if is_fatal_error:
        print(f"\n{'!'*80}")
        print(f"  FATAL ERROR DETECTED in: {operation_desc}")
        print(f"  Error details: {error_msg}")
        print(f"{'!'*80}\n")
        
        print("  A non-recoverable error has occurred.")
        print("  Saving all progress made so far...")
        
        save_progress(benchmark_data, json_filename)
        
        mode_str = "RESTART THE JUPYTER KERNEL" if IS_GPU_MODE else "restart the Python process"
        print("\n" + "="*80)
        print("  🛑 ACTION REQUIRED: The benchmark cannot continue safely.")
        print(f"     Progress has been saved to '{json_filename}'.")
        print(f"     Please {mode_str} and run the script again.")
        print("     The benchmark will automatically resume from where it left off.")
        print("="*80 + "\n")
        
        sys.exit(1)
    
    return False

def sync_computation():
    """
    Synchronize computation (GPU streams or CPU no-op).
    """
    if IS_GPU_MODE:
        cp.cuda.Stream.null.synchronize()
    # No synchronization needed for CPU operations

# ===============================================================================
# SECTION 3: PROGRESS MANAGEMENT
# ===============================================================================

def save_progress(benchmark_data: Dict, filename: str):
    """
    Save benchmark progress to JSON file with error handling.
    
    Args:
        benchmark_data: Current benchmark data to save
        filename: Output filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
    except Exception as e:
        print(f"   ⚠️ CRITICAL: Failed to save progress to {filename}: {e}")

def load_progress(filename: str) -> Optional[Dict]:
    """
    Load benchmark progress from existing JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Optional[Dict]: Loaded benchmark data or None if not found/invalid
    """
    if os.path.exists(filename):
        print(f"📊 Found existing benchmark file: {filename}. Resuming session.")
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            backup_file = f"{filename}.{int(time.time())}.bak"
            print(f"⚠️ Could not load {filename}: {e}. Backing up to {backup_file}")
            os.rename(filename, backup_file)
            return None
    else:
        print("🚀 No previous benchmark file found. Starting new session.")
        return None

# ===============================================================================
# SECTION 4: NETWORK PERFORMANCE TESTING
# ===============================================================================



def measure_http_latency(url: str, num_runs: int = 5, timeout: float = 10.0) -> Optional[Dict]:
    """
    Measure HTTP request latency to a URL.
    
    Args:
        url: Target URL
        num_runs: Number of HTTP requests
        timeout: Request timeout in seconds
        
    Returns:
        Optional[Dict]: HTTP latency metrics or None if failed
    """
    try:
        times = []
        response_codes = []
        successful_requests = 0
        
        # Create SSL context that doesn't verify certificates (for benchmark purposes only)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            try:
                request = urllib.request.Request(url)
                request.add_header('User-Agent', 'Matrix-Benchmark-Suite/1.0')
                
                # Use custom SSL context for HTTPS URLs
                if url.startswith('https://'):
                    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                        # Read a small amount to ensure the response is complete
                        response.read(1024)
                        end_time = time.perf_counter()
                        
                        request_time = (end_time - start_time) * 1000  # Convert to ms
                        times.append(request_time)
                        response_codes.append(response.getcode())
                        successful_requests += 1
                else:
                    # For HTTP URLs, use normal request
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        # Read a small amount to ensure the response is complete
                        response.read(1024)
                        end_time = time.perf_counter()
                        
                        request_time = (end_time - start_time) * 1000  # Convert to ms
                        times.append(request_time)
                        response_codes.append(response.getcode())
                        successful_requests += 1
                        
            except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
                continue
        
        if not times:
            return None
        
        return {
            'url': url,
            'avg_latency_ms': np.mean(times),
            'min_latency_ms': min(times),
            'max_latency_ms': max(times),
            'std_latency_ms': np.std(times),
            'response_codes': response_codes,
            'successful_requests': successful_requests,
            'total_attempts': num_runs,
            'success_rate': (successful_requests / num_runs) * 100
        }
    except Exception as e:
        print(f"  ❌ HTTP latency error for {url}: {e}")
        return None

def measure_download_speed(url: str, expected_size: int, timeout: float = 30.0) -> Optional[Dict]:
    """
    Measure download speed from a URL.
    
    Args:
        url: Download URL
        expected_size: Expected file size in bytes
        timeout: Download timeout in seconds
        
    Returns:
        Optional[Dict]: Download speed metrics or None if failed
    """
    try:
        start_time = time.perf_counter()
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Matrix-Benchmark-Suite/1.0')
        
        # Create SSL context that doesn't verify certificates (for benchmark purposes only)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Use custom SSL context for HTTPS URLs
        if url.startswith('https://'):
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                downloaded_bytes = 0
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    downloaded_bytes += len(chunk)
        else:
            # For HTTP URLs, use normal request
            with urllib.request.urlopen(request, timeout=timeout) as response:
                downloaded_bytes = 0
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    downloaded_bytes += len(chunk)
        
        end_time = time.perf_counter()
        download_time = end_time - start_time
        
        if downloaded_bytes == 0:
            return None
        
        # Calculate speeds
        speed_bps = downloaded_bytes / download_time
        speed_mbps = (speed_bps * 8) / (1024 * 1024)  # Convert to Mbps
        
        return {
            'url': url,
            'downloaded_bytes': downloaded_bytes,
            'expected_bytes': expected_size,
            'download_time_seconds': download_time,
            'speed_bps': speed_bps,
            'speed_mbps': speed_mbps,
            'speed_accuracy': (downloaded_bytes / expected_size) * 100 if expected_size > 0 else 100
        }
    except Exception as e:
        print(f"  ❌ Download speed error for {url}: {e}")
        return None

def run_network_benchmarks(benchmark_data: Dict, json_filename: str) -> Dict:
    """
    Run network performance benchmarks (HTTP latency and download speed).
    
    Args:
        benchmark_data: Current benchmark data
        json_filename: Progress file
        
    Returns:
        Dict: Network benchmark results
    """
    print(f"\n{'='*60}\nNETWORK PERFORMANCE BENCHMARKS\n{'='*60}")
    
    network_results = benchmark_data.get("network_results", {})
    
    # HTTP latency tests
    if "http_latency" not in network_results:
        print("\n--- HTTP Request Latency ---")
        http_targets = [
            # Mix of reliable HTTP and HTTPS endpoints for latency testing
            "https://www.google.com/",                # Major site HTTPS
            "https://www.bing.com/",
            "https://www.yahoo.com/",
        ]
        
        http_results = []
        successful_tests = 0
        
        for url in http_targets:
            protocol = "HTTPS" if url.startswith("https://") else "HTTP"
            print(f"  Testing {protocol} latency to {url.split('/')[2]}...")
            result = measure_http_latency(url, num_runs=5)
            if result and result['success_rate'] > 80:  # Only accept if >80% success rate
                http_results.append(result)
                print(f"    ✅ Average: {result['avg_latency_ms']:.1f}ms "
                      f"(Success rate: {result['success_rate']:.0f}%)")
                successful_tests += 1
                
                # Stop after getting enough good measurements
                if successful_tests >= 4:
                    break
            else:
                if result:
                    print(f"    ⚠️ Poor connectivity: {result['avg_latency_ms']:.1f}ms "
                          f"(Success rate: {result['success_rate']:.0f}%)")
                else:
                    print(f"    ❌ Failed to reach {url}")
        
        network_results["http_latency"] = http_results
        save_progress(benchmark_data, json_filename)
    
    # Download speed tests
    if "download_speed" not in network_results:
        print("\n--- Download Speed Tests ---")
        # Using reliable high-speed test files from CDN providers
        download_targets = [
            # CacheFly CDN - Specifically designed for bandwidth testing
            ("http://cachefly.cachefly.net/10mb.test", 10485760),      # 10MB HTTP
            ("http://cachefly.cachefly.net/100mb.test", 104857600),    # 100MB HTTP
            ("http://cachefly.cachefly.net/200mb.test", 209715200),    # 200MB HTTP
        ]
        
        download_results = []
        successful_tests = 0
        
        for url, expected_size in download_targets:
            size_mb = expected_size / (1024 * 1024)
            print(f"  Testing download speed with {size_mb:.0f}MB file from {url.split('/')[2]}...")
            result = measure_download_speed(url, expected_size, timeout=60.0)  # Longer timeout for large files
            if result and result['speed_accuracy'] > 95:  # Only accept if we got >95% of expected data
                download_results.append(result)
                print(f"    ✅ Speed: {result['speed_mbps']:.1f} Mbps "
                      f"({result['downloaded_bytes'] / (1024*1024):.1f}MB in "
                      f"{result['download_time_seconds']:.2f}s, "
                      f"Accuracy: {result['speed_accuracy']:.1f}%)")
                successful_tests += 1
                
                # Stop after getting 2-3 good measurements to avoid excessive bandwidth usage
                if successful_tests >= 3:
                    print(f"  ✅ Sufficient bandwidth measurements collected.")
                    break
            else:
                if result:
                    print(f"    ⚠️ Partial download: {result['speed_mbps']:.1f} Mbps "
                          f"(Only {result['speed_accuracy']:.1f}% downloaded)")
                else:
                    print(f"    ❌ Failed to download from {url}")
        
        if not download_results:
            print("  ⚠️ All high-speed tests failed. Trying smaller test files...")
            # Fallback to smaller files if large ones fail
            fallback_targets = [
                ("http://cachefly.cachefly.net/5mb.test", 5242880),    # 5MB
                ("https://httpbin.org/bytes/1048576", 1048576),        # 1MB fallback
            ]
            
            for url, expected_size in fallback_targets:
                size_mb = expected_size / (1024 * 1024)
                print(f"    Fallback: Testing {size_mb:.1f}MB file...")
                result = measure_download_speed(url, expected_size)
                if result:
                    download_results.append(result)
                    print(f"      ✅ Speed: {result['speed_mbps']:.1f} Mbps "
                          f"(Accuracy: {result['speed_accuracy']:.1f}%)")
                    break
        
        network_results["download_speed"] = download_results
        save_progress(benchmark_data, json_filename)
    
    # Update benchmark data with network results
    benchmark_data["network_results"] = network_results
    
    return network_results

# ===============================================================================
# SECTION 5: SYSTEM INFORMATION COLLECTION
# ===============================================================================

def get_cpu_info() -> Dict[str, str]:
    """
    Retrieve detailed CPU information.
    
    Returns:
        Dict[str, str]: CPU information dictionary
    """
    cpu_info = {}
    
    try:
        if platform.system() == "Windows":
            cpu_info['name'] = platform.processor()
        elif platform.system() == "Linux":
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_info['name'] = line.split(':')[1].strip()
                        break
                    elif line.startswith('cpu cores'):
                        cpu_info['cores'] = line.split(':')[1].strip()
        elif platform.system() == "Darwin":
            cpu_info['name'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).strip().decode()
            cpu_info['cores'] = subprocess.check_output(['sysctl', '-n', 'hw.ncpu']).strip().decode()
    except Exception:
        pass
    
    cpu_info.setdefault('name', 'Unknown_CPU')
    cpu_info.setdefault('cores', 'Unknown')
    cpu_info['filename'] = re.sub(r'[^\w\-_]', '_', cpu_info['name'].replace(' ', '_'))[:50]
    
    return cpu_info

def get_comprehensive_system_info() -> Dict[str, Any]:
    """
    Collect comprehensive system information for both CPU and GPU modes.
    
    Returns:
        Dict[str, Any]: System information dictionary
    """
    info = {'computation_mode': 'GPU' if IS_GPU_MODE else 'CPU'}
    
    # CPU information (always collected)
    cpu_info = get_cpu_info()
    info.update({f'cpu_{k}': v for k, v in cpu_info.items()})
    
    # GPU information (only in GPU mode)
    if IS_GPU_MODE:
        info['gpu_vendor'] = detect_gpu_vendor()
        try:
            device = cp.cuda.Device()
            info['gpu_id'] = device.id
            if hasattr(device, 'name'):
                info['gpu_name'] = device.name.decode('utf-8')
            if hasattr(device, 'compute_capability'):
                info['compute_capability'] = f"{device.compute_capability[0]}.{device.compute_capability[1]}"
            _, total_mem = device.mem_info
            info['gpu_memory_gb'] = round(total_mem / (1024**3), 2)
            
            # Try to get detailed GPU info from nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    info['gpu_name'] = parts[0]
                    info['gpu_driver_version'] = parts[1]
            except Exception:
                info.setdefault('gpu_name', 'Unknown NVIDIA GPU')
            
            info['cuda_version'] = get_cuda_version()
            info['cupy_version'] = cp.__version__
        except Exception as e:
            info['gpu_error'] = str(e)
    else:
        # Get system RAM info for CPU mode
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['system_memory_gb'] = round(memory.total / (1024**3), 2)
            info['available_memory_gb'] = round(memory.available / (1024**3), 2)
        except ImportError:
            info['system_memory_gb'] = 'Unknown (psutil not available)'
    
    # Software versions
    info['numpy_version'] = np.__version__
    info['python_version'] = platform.python_version()
    info['platform'] = platform.platform()
    
    return info

def print_system_info(info: Dict[str, Any]):
    """
    Display system information in formatted output.
    
    Args:
        info: System information dictionary
    """
    print(f"\n--- System Information ({info['computation_mode']} Mode) ---")
    for key, value in info.items():
        if key != 'computation_mode':
            display_key = key.replace('_', ' ').title()
            print(f"  {display_key}: {value}")
    print("--" + "-" * (len(f"System Information ({info['computation_mode']} Mode)") + 2) + "\n")

# ===============================================================================
# SECTION 6: MATRIX OPERATION BENCHMARKS
# ===============================================================================

def create_random_matrix(shape: Tuple[int, ...], dtype) -> Union[Any, Any]:
    """
    Create a random matrix using the appropriate library (CuPy or NumPy).
    
    Args:
        shape: Matrix shape tuple
        dtype: Data type
        
    Returns:
        Matrix with random values
    """
    compute_lib = cp if IS_GPU_MODE else np
    
    if dtype in [np.int64, np.int32] or (IS_GPU_MODE and dtype in [cp.int64, cp.int32]):
        if IS_GPU_MODE:
            iinfo = np.iinfo(np.int64 if dtype == cp.int64 else np.int32)
            return cp.random.randint(iinfo.min, iinfo.max, shape, dtype=dtype)
        else:
            iinfo = np.iinfo(dtype)
            return np.random.randint(iinfo.min, iinfo.max, shape, dtype=dtype)
    else:
        return compute_lib.random.random(shape).astype(dtype)

def benchmark_matrix_multiplication(size: int, dtype, benchmark_data: Dict, json_filename: str,
                                  num_warmup: int = 3, num_runs: int = 5) -> Optional[float]:
    """
    Benchmark square matrix multiplication (N×N @ N×N).
    
    Args:
        size: Matrix dimension
        dtype: Data type for matrices
        benchmark_data: Current benchmark data
        json_filename: Progress file
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        
    Returns:
        Optional[float]: Average execution time or None if failed
    """
    mode_str = "GPU" if IS_GPU_MODE else "CPU"
    op_desc = f"{dtype.__name__} {size}×{size} matrix multiplication ({mode_str})"
    
    try:
        print(f"  Creating {dtype.__name__} matrices of size {size}×{size}...")
        
        A = create_random_matrix((size, size), dtype)
        B = create_random_matrix((size, size), dtype)
        
        # Warmup phase
        print(f"  Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            if IS_GPU_MODE:
                C = cp.matmul(A, B)
            else:
                C = np.matmul(A, B)
            sync_computation()
            del C
        
        # Benchmark phase
        print(f"  Performing {num_runs} benchmark runs...")
        times = []
        for i in range(num_runs):
            sync_computation()
            start_time = time.perf_counter()
            if IS_GPU_MODE:
                C = cp.matmul(A, B)
            else:
                C = np.matmul(A, B)
            sync_computation()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            times.append(execution_time)
            print(f"    Run {i+1}: {execution_time:.6f}s")
            del C
        
        del A, B
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average: {avg_time:.6f}s ± {std_time:.6f}s")
        return avg_time
        
    except Exception as e:
        if "memory" in str(e).lower() or "Memory" in str(type(e).__name__):
            print(f"  ❌ Memory Error for {op_desc}")
            if IS_GPU_MODE:
                reconstruct_memory_pools()
            return None
        else:
            error_msg = str(e)
            handle_fatal_error(error_msg, op_desc, benchmark_data, json_filename)
            print(f"  ❌ Unexpected error for {op_desc}: {error_msg}")
            return None
    finally:
        clear_memory()

def benchmark_matrix_inverse(size: int, dtype, benchmark_data: Dict, json_filename: str,
                           num_warmup: int = 3, num_runs: int = 5) -> Optional[float]:
    """
    Benchmark matrix inversion for square matrices (float types only).
    
    Args:
        size: Matrix dimension
        dtype: Data type (only float types supported)
        benchmark_data: Current benchmark data
        json_filename: Progress file
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        
    Returns:
        Optional[float]: Average execution time or None if failed
    """
    mode_str = "GPU" if IS_GPU_MODE else "CPU"
    op_desc = f"{dtype.__name__} {size}×{size} matrix inverse ({mode_str})"
    
    try:
        print(f"  Creating {dtype.__name__} matrix of size {size}×{size} for inverse...")
        
        # Create well-conditioned matrix for inversion
        compute_lib = cp if IS_GPU_MODE else np
        A = compute_lib.random.random((size, size)).astype(dtype) * 0.1
        A += compute_lib.eye(size, dtype=dtype) * 10  # Diagonal dominance for stability
        
        # Warmup phase
        print(f"  Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            A_inv = compute_lib.linalg.inv(A)
            sync_computation()
            del A_inv
        
        # Benchmark phase
        print(f"  Performing {num_runs} benchmark runs...")
        times = []
        for i in range(num_runs):
            sync_computation()
            start_time = time.perf_counter()
            A_inv = compute_lib.linalg.inv(A)
            sync_computation()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            times.append(execution_time)
            print(f"    Run {i+1}: {execution_time:.6f}s")
            del A_inv
        
        del A
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average: {avg_time:.6f}s ± {std_time:.6f}s")
        return avg_time
        
    except Exception as e:
        if "memory" in str(e).lower() or "Memory" in str(type(e).__name__):
            print(f"  ❌ Memory Error for {op_desc}")
            if IS_GPU_MODE:
                reconstruct_memory_pools()
            return None
        else:
            error_msg = str(e)
            handle_fatal_error(error_msg, op_desc, benchmark_data, json_filename)
            print(f"  ❌ Unexpected error for {op_desc}: {error_msg}")
            return None
    finally:
        clear_memory()

def benchmark_matrix_vector_multiplication(size: int, dtype, benchmark_data: Dict, json_filename: str,
                                         num_warmup: int = 3, num_runs: int = 5) -> Optional[float]:
    """
    Benchmark matrix-vector multiplication (N×N @ N×1).
    
    Args:
        size: Matrix dimension
        dtype: Data type for matrices
        benchmark_data: Current benchmark data
        json_filename: Progress file
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        
    Returns:
        Optional[float]: Average execution time or None if failed
    """
    mode_str = "GPU" if IS_GPU_MODE else "CPU"
    op_desc = f"{dtype.__name__} {size}×{size} @ {size}×1 matrix-vector multiplication ({mode_str})"
    
    try:
        print(f"  Creating {dtype.__name__} matrix {size}×{size} and vector {size}×1...")
        
        A = create_random_matrix((size, size), dtype)
        x = create_random_matrix((size, 1), dtype)
        
        # Warmup phase
        print(f"  Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            if IS_GPU_MODE:
                y = cp.matmul(A, x)
            else:
                y = np.matmul(A, x)
            sync_computation()
            del y
        
        # Benchmark phase
        print(f"  Performing {num_runs} benchmark runs...")
        times = []
        for i in range(num_runs):
            sync_computation()
            start_time = time.perf_counter()
            if IS_GPU_MODE:
                y = cp.matmul(A, x)
            else:
                y = np.matmul(A, x)
            sync_computation()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            del y
        
        del A, x
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average: {avg_time:.6f}s ± {std_time:.6f}s")
        return avg_time
        
    except Exception as e:
        if "memory" in str(e).lower() or "Memory" in str(type(e).__name__):
            print(f"  ❌ Memory Error for {op_desc}")
            if IS_GPU_MODE:
                reconstruct_memory_pools()
            return None
        else:
            error_msg = str(e)
            handle_fatal_error(error_msg, op_desc, benchmark_data, json_filename)
            print(f"  ❌ Unexpected error for {op_desc}: {error_msg}")
            return None
    finally:
        clear_memory()

def benchmark_matrix_addition(size: int, dtype, benchmark_data: Dict, json_filename: str,
                            num_warmup: int = 3, num_runs: int = 5) -> Optional[float]:
    """
    Benchmark square matrix addition (A + B + C).
    
    Args:
        size: Matrix dimension
        dtype: Data type for matrices
        benchmark_data: Current benchmark data
        json_filename: Progress file
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        
    Returns:
        Optional[float]: Average execution time or None if failed
    """
    mode_str = "GPU" if IS_GPU_MODE else "CPU"
    op_desc = f"{dtype.__name__} {size}×{size} matrix addition ({mode_str})"
    
    try:
        print(f"  Creating 3 {dtype.__name__} matrices of size {size}×{size} for addition...")
        
        A = create_random_matrix((size, size), dtype)
        B = create_random_matrix((size, size), dtype)
        C = create_random_matrix((size, size), dtype)
        
        # Warmup phase
        print(f"  Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            D = A + B + C
            sync_computation()
            del D
        
        # Benchmark phase
        print(f"  Performing {num_runs} benchmark runs...")
        times = []
        for i in range(num_runs):
            sync_computation()
            start_time = time.perf_counter()
            D = A + B + C
            sync_computation()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            del D
        
        del A, B, C
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average: {avg_time:.6f}s ± {std_time:.6f}s")
        return avg_time
        
    except Exception as e:
        if "memory" in str(e).lower() or "Memory" in str(type(e).__name__):
            print(f"  ❌ Memory Error for {op_desc}")
            if IS_GPU_MODE:
                reconstruct_memory_pools()
            return None
        else:
            error_msg = str(e)
            handle_fatal_error(error_msg, op_desc, benchmark_data, json_filename)
            print(f"  ❌ Unexpected error for {op_desc}: {error_msg}")
            return None
    finally:
        clear_memory()

def benchmark_random_generation(size: int, dtype, generation_type: str, benchmark_data: Dict, json_filename: str,
                               num_warmup: int = 3, num_runs: int = 10) -> Optional[float]:
    """
    Benchmark random matrix generation operations.
    
    Args:
        size: Matrix dimension
        dtype: Data type for matrices
        generation_type: Type of random generation ('rand' or 'randint')
        benchmark_data: Current benchmark data
        json_filename: Progress file
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        
    Returns:
        Optional[float]: Average execution time or None if failed
    """
    mode_str = "GPU" if IS_GPU_MODE else "CPU"
    op_desc = f"{dtype.__name__} {size}×{size} random {generation_type} generation ({mode_str})"
    
    try:
        print(f"  Benchmarking {generation_type}() for {size}×{size} {dtype.__name__} matrix...")
        compute_lib = cp if IS_GPU_MODE else np
        
        # Warmup phase
        print(f"  Performing {num_warmup} warmup runs...")
        for _ in range(num_warmup):
            if generation_type == "rand":
                A = compute_lib.random.rand(size, size).astype(dtype)
            elif generation_type == "randint":
                if IS_GPU_MODE:
                    iinfo = np.iinfo(np.int64 if dtype == cp.int64 else np.int32)
                    A = cp.random.randint(iinfo.min, iinfo.max, (size, size), dtype=dtype)
                else:
                    iinfo = np.iinfo(dtype)
                    A = np.random.randint(iinfo.min, iinfo.max, (size, size), dtype=dtype)
            sync_computation()
            del A
        
        # Benchmark phase
        print(f"  Performing {num_runs} benchmark runs...")
        times = []
        for i in range(num_runs):
            clear_memory()
            sync_computation()
            start_time = time.perf_counter()
            if generation_type == "rand":
                A = compute_lib.random.rand(size, size).astype(dtype)
            elif generation_type == "randint":
                if IS_GPU_MODE:
                    iinfo = np.iinfo(np.int64 if dtype == cp.int64 else np.int32)
                    A = cp.random.randint(iinfo.min, iinfo.max, (size, size), dtype=dtype)
                else:
                    iinfo = np.iinfo(dtype)
                    A = np.random.randint(iinfo.min, iinfo.max, (size, size), dtype=dtype)
            sync_computation()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            times.append(execution_time)
            print(f"    Run {i+1}: {execution_time:.6f}s")
            del A
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average: {avg_time:.6f}s ± {std_time:.6f}s")
        return avg_time
        
    except Exception as e:
        if "memory" in str(e).lower() or "Memory" in str(type(e).__name__):
            print(f"  ❌ Memory Error for {op_desc}")
            if IS_GPU_MODE:
                reconstruct_memory_pools()
            return None
        else:
            error_msg = str(e)
            handle_fatal_error(error_msg, op_desc, benchmark_data, json_filename)
            print(f"  ❌ Unexpected error for {op_desc}: {error_msg}")
            return None
    finally:
        clear_memory()

# ===============================================================================
# SECTION 7: MAIN BENCHMARK ORCHESTRATION
# ===============================================================================

def run_benchmark_suite(benchmark_configs: List[Dict], sizes: List[int], 
                       benchmark_data: Dict, json_filename: str, skip_list: Dict) -> Dict:
    """
    Execute the complete benchmark suite with all configured operations.
    
    Args:
        benchmark_configs: List of benchmark configuration dictionaries
        sizes: List of matrix sizes to test
        benchmark_data: Current benchmark data
        json_filename: Progress file
        skip_list: Dictionary of configurations to skip
        
    Returns:
        Dict: Updated benchmark results
    """
    all_results = benchmark_data["benchmark_results"]
    
    for config in benchmark_configs:
        operation_type = config['type']
        dtype = config['dtype']
        dtype_name = config['dtype_name']
        
        if operation_type == "matrix_multiplication":
            key = f"{dtype_name}_matrix_multiplication"
            print(f"\n{'='*60}\nBENCHMARKING {key.upper()}\n{'='*60}")
            all_results.setdefault(key, {'results': [], 'failed_sizes': []})
            existing_successes = {res[0] for res in all_results[key]['results']}
            
            for size in sizes:
                if key in skip_list and size in skip_list[key]:
                    print(f"\n🚫 Manually skipping {size}×{size} (blacklisted)...")
                    continue
                if size in existing_successes:
                    print(f"\n⏭️ Skipping {size}×{size} (already successful)...")
                    continue
                
                print(f"\nBenchmarking {size}×{size} {dtype_name} matrix multiplication...")
                avg_time = benchmark_matrix_multiplication(size, dtype, benchmark_data, json_filename)
                if avg_time is not None:
                    all_results[key]['results'].append([size, avg_time])
                    flops = 2 * size**3 - size**2
                    gflops = flops / (avg_time * 1e9)
                    print(f"  ✅ Performance: {gflops:.2f} GFLOPS")
                else:
                    all_results[key]['failed_sizes'].append(size)
                save_progress(benchmark_data, json_filename)
        
        elif operation_type == "matrix_inverse" and "int" not in dtype_name:  # Only for float types
            key = f"{dtype_name}_matrix_inverse"
            print(f"\n{'='*60}\nBENCHMARKING {key.upper()}\n{'='*60}")
            all_results.setdefault(key, {'results': [], 'failed_sizes': []})
            existing_successes = {res[0] for res in all_results[key]['results']}
            
            # Limit inverse to smaller sizes due to computational complexity
            inverse_sizes = [s for s in sizes if s <= 5000]
            for size in inverse_sizes:
                if key in skip_list and size in skip_list[key]:
                    print(f"\n🚫 Manually skipping {size}×{size} (blacklisted)...")
                    continue
                if size in existing_successes:
                    print(f"\n⏭️ Skipping {size}×{size} (already successful)...")
                    continue
                
                print(f"\nBenchmarking {size}×{size} {dtype_name} matrix inverse...")
                avg_time = benchmark_matrix_inverse(size, dtype, benchmark_data, json_filename)
                if avg_time is not None:
                    all_results[key]['results'].append([size, avg_time])
                    flops = (2.0/3.0) * size**3
                    gflops = flops / (avg_time * 1e9)
                    print(f"  ✅ Performance: {gflops:.2f} GFLOPS")
                else:
                    all_results[key]['failed_sizes'].append(size)
                save_progress(benchmark_data, json_filename)
        
        elif operation_type == "matrix_vector_multiplication":
            key = f"{dtype_name}_matrix_vector_multiplication"
            print(f"\n{'='*60}\nBENCHMARKING {key.upper()}\n{'='*60}")
            all_results.setdefault(key, {'results': [], 'failed_sizes': []})
            existing_successes = {res[0] for res in all_results[key]['results']}
            
            for size in sizes:
                if key in skip_list and size in skip_list[key]:
                    print(f"\n🚫 Manually skipping {size}×{size} (blacklisted)...")
                    continue
                if size in existing_successes:
                    print(f"\n⏭️ Skipping {size}×{size} (already successful)...")
                    continue
                
                print(f"\nBenchmarking {size}×{size} @ {size}×1 {dtype_name} matrix-vector multiplication...")
                avg_time = benchmark_matrix_vector_multiplication(size, dtype, benchmark_data, json_filename)
                if avg_time is not None:
                    all_results[key]['results'].append([size, avg_time])
                    flops = 2 * size * size
                    gflops = flops / (avg_time * 1e9)
                    print(f"  ✅ Performance: {gflops:.2f} GFLOPS")
                else:
                    all_results[key]['failed_sizes'].append(size)
                save_progress(benchmark_data, json_filename)
        
        elif operation_type == "matrix_addition":
            key = f"{dtype_name}_matrix_addition"
            print(f"\n{'='*60}\nBENCHMARKING {key.upper()}\n{'='*60}")
            all_results.setdefault(key, {'results': [], 'failed_sizes': []})
            existing_successes = {res[0] for res in all_results[key]['results']}
            
            for size in sizes:
                if key in skip_list and size in skip_list[key]:
                    print(f"\n🚫 Manually skipping {size}×{size} (blacklisted)...")
                    continue
                if size in existing_successes:
                    print(f"\n⏭️ Skipping {size}×{size} (already successful)...")
                    continue
                
                print(f"\nBenchmarking {size}×{size} {dtype_name} matrix addition...")
                avg_time = benchmark_matrix_addition(size, dtype, benchmark_data, json_filename)
                if avg_time is not None:
                    all_results[key]['results'].append([size, avg_time])
                    ops = 2 * size * size
                    gops = ops / (avg_time * 1e9)
                    print(f"  ✅ Performance: {gops:.2f} GOPS")
                else:
                    all_results[key]['failed_sizes'].append(size)
                save_progress(benchmark_data, json_filename)
        
        elif operation_type == "random_generation":
            for gen_type in ['rand', 'randint']:
                if gen_type == 'randint' and "int" not in dtype_name:
                    continue
                if gen_type == 'rand' and "int" in dtype_name:
                    continue
                
                key = f"{dtype_name}_random_{gen_type}"
                print(f"\n{'='*60}\nBENCHMARKING {key.upper()}\n{'='*60}")
                all_results.setdefault(key, {'results': [], 'failed_sizes': []})
                existing_successes = {res[0] for res in all_results[key]['results']}
                
                for size in sizes:
                    if key in skip_list and size in skip_list[key]:
                        print(f"\n🚫 Manually skipping {size}×{size} (blacklisted)...")
                        continue
                    if size in existing_successes:
                        print(f"\n⏭️ Skipping {size}×{size} (already successful)...")
                        continue
                    
                    print(f"\nBenchmarking {size}×{size} {dtype_name} random {gen_type} generation...")
                    avg_time = benchmark_random_generation(size, dtype, gen_type, benchmark_data, json_filename)
                    if avg_time is not None:
                        all_results[key]['results'].append([size, avg_time])
                        elements = size * size
                        throughput = elements / (avg_time * 1e6)  # Million elements per second
                        print(f"  ✅ Throughput: {throughput:.2f} M elements/sec")
                    else:
                        all_results[key]['failed_sizes'].append(size)
                    save_progress(benchmark_data, json_filename)
    
    return all_results

def main():
    """
    Main benchmark orchestration function.
    """
    mode_str = "GPU (CuPy)" if IS_GPU_MODE else "CPU (NumPy)"
    print(f"Comprehensive Matrix Operations & Network Performance Benchmark Suite")
    print(f"Running in {mode_str} mode")
    
    # Configuration - adjust sizes based on computation mode
    if IS_GPU_MODE:
        sizes = [100, 250, 500, 1000, 2500, 5000, 10000]
    else:
        # Smaller sizes for CPU to keep benchmarks reasonable
        sizes = [100, 250, 500]
        print("ℹ️ Using smaller matrix sizes for CPU benchmarking")
    
    # Data types - normalize between CuPy and NumPy
    if IS_GPU_MODE:
        dtypes_to_test = [(cp.float64, "float64"), (cp.int64, "int64")]
    else:
        dtypes_to_test = [(np.float64, "float64"), (np.int64, "int64")]
    
    # Operation configurations
    benchmark_configs = []
    for dtype, dtype_name in dtypes_to_test:
        benchmark_configs.extend([
            {'type': 'matrix_multiplication', 'dtype': dtype, 'dtype_name': dtype_name},
            {'type': 'matrix_inverse', 'dtype': dtype, 'dtype_name': dtype_name},
            {'type': 'matrix_vector_multiplication', 'dtype': dtype, 'dtype_name': dtype_name},
            {'type': 'matrix_addition', 'dtype': dtype, 'dtype_name': dtype_name},
            {'type': 'random_generation', 'dtype': dtype, 'dtype_name': dtype_name}
        ])
    
    # Skip list for problematic configurations
    skip_list = {
        # Example entries (uncomment and modify as needed):
        # 'int64_matrix_multiplication_gpu': [50000],
        # 'float64_matrix_addition_cpu': [5000],
    }
    
    print("\n--- Manual Skip List Configuration ---")
    if not skip_list:
        print("  No configurations are currently blacklisted.")
    else:
        for op, configs in skip_list.items():
            print(f"  Will skip {op} for configs: {configs}")
    
    # Generate session filename
    base_info = get_comprehensive_system_info()
    cpu_name = base_info.get('cpu_filename', 'Unknown_CPU')
    mode_suffix = "gpu" if IS_GPU_MODE else "cpu"
    if IS_GPU_MODE:
        gpu_name_safe = re.sub(r'[^\w\-_]', '_', base_info.get('gpu_name', 'Unknown_GPU'))
        json_filename = 'local-results/' + f"matrix_network_benchmark_{mode_suffix}_{cpu_name}_{gpu_name_safe}.json"
    else:
        json_filename = 'local-results/' +  f"matrix_network_benchmark_{mode_suffix}_{cpu_name}.json"
    
    # Load or initialize benchmark data
    benchmark_data = load_progress(json_filename)
    if benchmark_data is None:
        system_info = get_comprehensive_system_info()
        print_system_info(system_info)
        benchmark_data = {
            "system_info": system_info,
            "benchmark_results": {},
            "network_results": {},
            "benchmark_config": {
                "computation_mode": mode_str,
                "sizes": sizes,
                "data_types": [name for _, name in dtypes_to_test],
                "operations": ["matrix_multiplication", "matrix_inverse", "matrix_vector_multiplication", 
                             "matrix_addition", "random_generation"],
                "skip_list": skip_list
            }
        }
    else:
        print_system_info(benchmark_data["system_info"])
    
    # Run network benchmarks first (independent of matrix operations)
    network_results = run_network_benchmarks(benchmark_data, json_filename)
    
    # Execute matrix benchmark suite
    print(f"\n--- Starting/Resuming Matrix Benchmark ({mode_str}) ---")
    all_results = run_benchmark_suite(benchmark_configs, sizes, benchmark_data, json_filename, skip_list)
    
    # Final summary
    print(f"\n📊 Comprehensive benchmark results saved to {json_filename}")
    print(f"\n{'='*60}\nBENCHMARK SUMMARY ({mode_str})\n{'='*60}")
    
    # Network summary
    print("\n--- Network Performance Summary ---")
    if "http_latency" in network_results and network_results["http_latency"]:
        http_avg = np.mean([r['avg_latency_ms'] for r in network_results['http_latency']])
        print(f"  HTTP Latency: {http_avg:.1f}ms average across {len(network_results['http_latency'])} endpoints")
    else:
        print(f"  HTTP Latency: No successful tests")
    
    if "download_speed" in network_results and network_results["download_speed"]:
        speed_avg = np.mean([r['speed_mbps'] for r in network_results['download_speed']])
        max_speed = max([r['speed_mbps'] for r in network_results['download_speed']])
        print(f"  Download Speed: {speed_avg:.1f} Mbps average, {max_speed:.1f} Mbps peak")
    else:
        print(f"  Download Speed: No successful tests")
    
    # Matrix operations summary
    print("\n--- Matrix Operations Summary ---")
    for operation, data in all_results.items():
        successful = len(data['results'])
        failed = len(data.get('failed_sizes', []))
        print(f"  {operation}: {successful} successful, {failed} failed")
    
    print(f"\n✅ Comprehensive benchmark suite completed successfully!")
    print(f"   Mode: {mode_str}")
    print(f"   Results file: {json_filename}")
    print(f"   Matrix operations tested: {len(all_results)}")
    print(f"   Network tests completed: {len(network_results)}")

# ===============================================================================
# SECTION 8: SCRIPT EXECUTION
# ===============================================================================

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        mode_str = "kernel" if IS_GPU_MODE else "Python process"
        print(f"\n--- Benchmark terminated. Please restart {mode_str} if instructed. ---")
    except Exception as e:
        import traceback
        print(f"\n💥 Critical error occurred: {e}")
        traceback.print_exc()
    finally:
        clear_memory()
        print("\n--- Comprehensive benchmark script finished ---")