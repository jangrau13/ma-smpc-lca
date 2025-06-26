"""
Complete Enhanced Critical Path Time Calculator for Distributed LCA Protocol
Reads from timing_files folder (GPU data) + local-results folder (CPU/GPU + network data)
Performs exact same analysis as paste1 + paste2 with local integration

Updated formula based on the critical path analysis:
T_total = 6 × T_transfer(N×N) + 3 × T_transfer(N×1) + 2 × T_transfer(R×N) +
          1 × T_DecentralizedParticipationProtocol + 6 × T_matmul(N×N) +
          3 × T_matmul((N×N)×(N×1)) + 3 × T_matmul((R×N)×(N×N)) +
          1 × T_inverse(N×N) + 6 × T_add(N×N) + 2 × T_add(N×1) +
          2 × T_add(R×N) + 3 × T_randgen(N×N) + 1 × T_randgen(N×1)
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import itertools
from typing import Dict, List, Tuple
import ast
import math
import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

class CompleteLCACriticalPathCalculator:
    def __init__(self, timing_files_dir: str = "timing_files", local_results_dir: str = "local-results"):
        """Initialize the calculator with benchmark data from timing_files and local JSON results."""
        self.timing_files_dir = timing_files_dir
        self.local_results_dir = local_results_dir
        self.gpu_timings = {}
        
        # Base network scenarios
        self.network_scenarios = {
            'Residential': {'latency_ms': 30, 'bandwidth_mbps': 100},
            'Enterprise': {'latency_ms': 5, 'bandwidth_mbps': 1000},
            'DataCenter': {'latency_ms': 0.5, 'bandwidth_mbps': 10000}
        }
        
        # Decentralized participation protocol parameters
        self.protocol_params = {
            'P_light': 1500,
            'SizeDataType': 8,
            'T_overhead': 0.005,
        }
        
        self.load_all_data()

    def load_all_data(self):
        """Load data from timing_files and local-results folders."""
        print("Loading benchmark data from multiple sources...")
        
        # Load timing_files data (GPU benchmarks)
        self.load_timing_files()
        
        # Load local JSON results (CPU/GPU + network data)
        self.load_local_results()

    def load_timing_files(self):
        """Load GPU benchmark data from timing_files folder (like original paste1)."""
        if not os.path.exists(self.timing_files_dir):
            print(f"⚠️  timing_files directory {self.timing_files_dir} not found")
            return
        
        json_files = glob.glob(os.path.join(self.timing_files_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files in {self.timing_files_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                system_name = self.extract_system_name_from_timing_files(data)
                print(f"  Processing {json_file} -> {system_name}")
                
                # Process benchmark results from timing_files
                self.process_timing_files_benchmark_data(data, system_name)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {json_file}: {e}")

    def extract_system_name_from_timing_files(self, data: dict) -> str:
        """Extract system name from timing_files JSON structure."""
        system_info = data.get('system_info', {})
        
        if 'gpu_name' in system_info:
            gpu_name = system_info['gpu_name']
            # Normalize H100 variants
            if 'H100' in gpu_name:
                return 'NVIDIA H100'
            elif 'NVIDIA' in gpu_name:
                return gpu_name.replace('NVIDIA ', '').strip()
            return gpu_name
        else:
            return "Unknown GPU"

    def process_timing_files_benchmark_data(self, data: dict, system_name: str):
        """Process benchmark results from timing_files JSON data."""
        benchmark_results = data.get('benchmark_results', {})
        
        if system_name not in self.gpu_timings:
            self.gpu_timings[system_name] = {}
        
        print(f"      Found {len(benchmark_results)} benchmark operations: {list(benchmark_results.keys())}")
        
        # Map timing_files benchmark names to operation types - CORRECTED to match actual timing_files format
        operation_mapping = {
            'float64_matmul': ('square_matmul', 'floating-point'),
            'int64_matmul': ('square_matmul', 'fixed-point'),
            'float64_inverse': ('matrix_inverse', 'floating-point'),
            'float64_matmul_rectangular': ('rectangular_matmul', 'floating-point'),
            'int64_matmul_rectangular': ('rectangular_matmul', 'fixed-point'),
            'float64_matmul_vector': ('vector_matmul', 'floating-point'),
            'int64_matmul_vector': ('vector_matmul', 'fixed-point'),
            'float64_addition_square': ('matrix_addition', 'floating-point'),
            'int64_addition_square': ('matrix_addition', 'fixed-point'),
            'float64_addition_rectangular': ('rectangular_addition', 'floating-point'),
            'int64_addition_rectangular': ('rectangular_addition', 'fixed-point'),
            'rand_float64': ('random_generation', 'floating-point'),
            'randint_int64': ('random_generation', 'fixed-point')
        }
        
        for benchmark_name, result_data in benchmark_results.items():
            if benchmark_name in operation_mapping:
                operation_type, approach = operation_mapping[benchmark_name]
                
                if operation_type not in self.gpu_timings[system_name]:
                    self.gpu_timings[system_name][operation_type] = {}
                
                if approach not in self.gpu_timings[system_name][operation_type]:
                    self.gpu_timings[system_name][operation_type][approach] = {}
                
                # Extract results data
                results = result_data.get('results', [])
                if results and len(results) >= 2:
                    
                    # Handle different result formats based on operation type
                    if operation_type in ['square_matmul', 'matrix_inverse', 'matrix_addition', 'random_generation', 'vector_matmul']:
                        # Simple format: [[size, time], [size, time], ...]
                        sizes = [result[0] for result in results]
                        times = [result[1] for result in results]
                        
                        # Create interpolation function
                        sizes, times = zip(*sorted(zip(sizes, times)))
                        self.gpu_timings[system_name][operation_type][approach]['interpolator'] = interp1d(
                            sizes, times, kind='linear', bounds_error=False, fill_value='extrapolate'
                        )
                        self.gpu_timings[system_name][operation_type][approach]['data_points'] = list(zip(sizes, times))
                        print(f"      ✓ Added {operation_type} ({approach}): {len(results)} data points")
                        
                    elif operation_type in ['rectangular_matmul', 'rectangular_addition']:
                        # Rectangular format: [[[rows, cols], time], [[rows, cols], time], ...]
                        rect_data = {}
                        for result in results:
                            try:
                                dims = result[0]  # [rows, cols]
                                time = result[1]
                                if len(dims) == 2:
                                    key = f"{dims[0]}x{dims[1]}"
                                    rect_data[key] = time
                            except (ValueError, TypeError, IndexError):
                                continue
                        
                        self.gpu_timings[system_naame][operation_type][approach]['rectangular_data'] = rect_data
                        print(f"      ✓ Added {operation_type} ({approach}): {len(rect_data)} rectangular combinations")
                        
                else:
                    print(f"      ⚠️  Insufficient data for {benchmark_name}: {len(results)} points")
            else:
                print(f"      ⚠️  Unknown benchmark: {benchmark_name}")
                print(f"      ✓ Added {operation_type} ({approach}): {len(results)} data points")

    def extract_system_name_from_local_results(self, data: dict) -> str:
        """Extract system name from local-results JSON structure."""
        system_info = data.get('system_info', {})
        
        if 'gpu_name' in system_info:
            # GPU system
            gpu_name = system_info['gpu_name']
            if 'NVIDIA' in gpu_name:
                gpu_name = gpu_name.replace('NVIDIA ', '').strip()
            return gpu_name
        elif 'cpu_name' in system_info:
            # CPU system
            cpu_name = system_info['cpu_name']
            if 'Intel' in cpu_name:
                parts = cpu_name.split()
                model_parts = []
                for part in parts:
                    if any(x in part for x in ['Xeon', 'Core', 'i3', 'i5', 'i7', 'i9']):
                        model_parts.append(part)
                    elif '@' in part:
                        model_parts.append(part.replace('@', '').strip())
                        break
                if model_parts:
                    return f"Intel {' '.join(model_parts)} CPU"
            return "CPU System"
        else:
            return "Unknown System"

    def load_local_results(self):
        """Load and process local JSON benchmark results."""
        if not os.path.exists(self.local_results_dir):
            print(f"⚠️  Local results directory {self.local_results_dir} not found")
            return
        
        json_files = glob.glob(os.path.join(self.local_results_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files in {self.local_results_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                system_name = self.extract_system_name_from_local_results(data)
                print(f"  Processing {json_file} -> {system_name}")
                
                # Process benchmark results
                self.process_local_benchmark_data(data, system_name)
                
                # Process network data and add new scenarios
                self.process_network_data(data, system_name)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {json_file}: {e}")

    def process_network_data(self, data: dict, system_name: str):
        """Process network performance data and add new network scenarios."""
        network_results = data.get('network_results', {})
        
        if 'http_latency' in network_results and 'download_speed' in network_results:
            # Calculate average latency from http_latency results
            latency_data = network_results['http_latency']
            if latency_data:
                avg_latencies = [entry['avg_latency_ms'] for entry in latency_data if 'avg_latency_ms' in entry]
                if avg_latencies:
                    measured_latency = np.mean(avg_latencies)
                else:
                    measured_latency = 50
            else:
                measured_latency = 50
            
            # Calculate average bandwidth from download_speed results
            speed_data = network_results['download_speed']
            if speed_data:
                speeds = [entry['speed_mbps'] for entry in speed_data if 'speed_mbps' in entry]
                if speeds:
                    measured_bandwidth = np.mean(speeds)
                else:
                    measured_bandwidth = 100
            else:
                measured_bandwidth = 100
            
            # Create new network scenario
            scenario_name = f"{system_name}_Measured"
            self.network_scenarios[scenario_name] = {
                'latency_ms': measured_latency,
                'bandwidth_mbps': measured_bandwidth
            }
            
            print(f"    ✓ Added network scenario: {scenario_name} (latency: {measured_latency:.1f}ms, bandwidth: {measured_bandwidth:.1f}Mbps)")

    def process_local_benchmark_data(self, data: dict, system_name: str):
        """Process benchmark results from local JSON data."""
        benchmark_results = data.get('benchmark_results', {})
        
        if system_name not in self.gpu_timings:
            self.gpu_timings[system_name] = {}
        
        print(f"      Found {len(benchmark_results)} benchmark operations: {list(benchmark_results.keys())}")
        
        # Map local benchmark names to operation types
        operation_mapping = {
            'float64_matrix_multiplication_cpu': ('square_matmul', 'floating-point'),
            'float64_matmul': ('square_matmul', 'floating-point'),
            'float64_matrix_inverse_cpu': ('matrix_inverse', 'floating-point'),
            'float64_inverse': ('matrix_inverse', 'floating-point'),
            'float64_matrix_vector_multiplication_cpu': ('vector_matmul', 'floating-point'),
            'float64_matrix_addition_cpu': ('matrix_addition', 'floating-point'),
            'float64_random_rand_cpu': ('random_generation', 'floating-point'),
            'int64_matrix_multiplication_cpu': ('square_matmul', 'fixed-point'),
            'int64_matmul': ('square_matmul', 'fixed-point'),
            'int64_matrix_vector_multiplication_cpu': ('vector_matmul', 'fixed-point'),
            'int64_matrix_addition_cpu': ('matrix_addition', 'fixed-point'),
            'int64_random_randint_cpu': ('random_generation', 'fixed-point'),
        }
        
        for benchmark_name, result_data in benchmark_results.items():
            if benchmark_name in operation_mapping:
                operation_type, approach = operation_mapping[benchmark_name]
                
                if operation_type not in self.gpu_timings[system_name]:
                    self.gpu_timings[system_name][operation_type] = {}
                
                if approach not in self.gpu_timings[system_name][operation_type]:
                    self.gpu_timings[system_name][operation_type][approach] = {}
                
                # Extract results data
                results = result_data.get('results', [])
                if results and len(results) >= 2:
                    sizes = [result[0] for result in results]
                    times = [result[1] for result in results]
                    
                    # Create interpolation function
                    sizes, times = zip(*sorted(zip(sizes, times)))
                    self.gpu_timings[system_name][operation_type][approach]['interpolator'] = interp1d(
                        sizes, times, kind='linear', bounds_error=False, fill_value='extrapolate'
                    )
                    self.gpu_timings[system_name][operation_type][approach]['data_points'] = list(zip(sizes, times))

    def estimate_vector_addition_from_rectangular(self, gpu: str, N: int, approach: str) -> float:
        """Estimate vector addition time from rectangular addition data or other available data."""
        vector_add_time = self.get_computation_time(gpu, 'vector_addition', N, approach)
        if not np.isnan(vector_add_time):
            return vector_add_time

        rect_add_time = self.get_computation_time(gpu, 'rectangular_addition', (N, 1), approach)
        if not np.isnan(rect_add_time):
            return rect_add_time

        square_add_time = self.get_computation_time(gpu, 'matrix_addition', N, approach)
        if not np.isnan(square_add_time):
            return square_add_time * (1 / N)

        # If matrix_addition is missing, estimate from vector multiplication
        vector_matmul_time = self.get_computation_time(gpu, 'vector_matmul', N, approach)
        if not np.isnan(vector_matmul_time):
            # Vector addition is typically much faster than vector multiplication
            return vector_matmul_time * 0.01  # ~1% of multiplication time
            
        # Last resort: estimate from square matrix multiplication
        square_matmul_time = self.get_computation_time(gpu, 'square_matmul', N, approach)
        if not np.isnan(square_matmul_time):
            # Vector addition for N elements vs N×N matrix multiplication
            return square_matmul_time * 0.001 / N  # Very rough estimate

        return np.nan

    def calculate_transfer_time(self, matrix_dims: Tuple[int, int], network_scenario: str) -> float:
        """Calculate transfer time for a matrix given network scenario."""
        rows, cols = matrix_dims
        data_size_bits = rows * cols * 64  # Always 64-bit for transfers

        network = self.network_scenarios[network_scenario]
        latency_s = network['latency_ms'] / 1000.0
        bandwidth_bps = network['bandwidth_mbps'] * 1_000_000

        transfer_time_s = data_size_bits / bandwidth_bps
        total_time = latency_s + transfer_time_s

        return total_time

    def calculate_decentralized_participation_protocol_time(self, N: int, R: int, D: int, network_scenario: str) -> float:
        """Calculate the time for the decentralized participation protocol."""
        P_light = self.protocol_params['P_light']
        SizeDataType = self.protocol_params['SizeDataType']
        T_overhead = self.protocol_params['T_overhead']

        network = self.network_scenarios[network_scenario]
        latency_s = network['latency_ms'] / 1000.0
        bandwidth_bps = network['bandwidth_mbps'] * 1_000_000

        # Calculate tree height h ≈ √(N/D) + log_D(N)
        h = math.sqrt(N / D) + math.log(N) / math.log(D)

        # Calculate T_https,light
        T_https_light = latency_s + (P_light * 8) / bandwidth_bps + T_overhead

        # Calculate T_https,heavy
        P_heavy_bytes = P_light + (N + R) * SizeDataType
        T_https_heavy = latency_s + (P_heavy_bytes * 8) / bandwidth_bps + T_overhead

        # Calculate total protocol time
        T_registration = h * T_https_light
        T_notification = h * T_https_light
        T_creation = h * T_https_heavy

        T_total = T_registration + T_notification + T_creation

        return T_total

    def get_computation_time(self, gpu: str, operation: str, size_or_dims, approach: str = 'floating-point') -> float:
        """Get computation time for a specific operation with specified computational approach."""
        if gpu not in self.gpu_timings:
            return np.nan

        gpu_ops = self.gpu_timings[gpu]

        if operation in ['square_matmul', 'matrix_inverse', 'matrix_addition', 'random_generation']:
            # Matrix inverse always uses floating-point
            if operation == 'matrix_inverse':
                approach = 'floating-point'

            if (operation in gpu_ops and
                approach in gpu_ops[operation] and
                'interpolator' in gpu_ops[operation][approach]):
                return float(gpu_ops[operation][approach]['interpolator'](size_or_dims))

        elif operation in ['rectangular_matmul', 'rectangular_addition']:
            if (operation in gpu_ops and
                approach in gpu_ops[operation]):
                key = f"{size_or_dims[0]}x{size_or_dims[1]}"
                rect_data = gpu_ops[operation][approach].get('rectangular_data', {})
                if key in rect_data:
                    return rect_data[key]

        elif operation in ['vector_matmul', 'vector_addition']:
            if (operation in gpu_ops and
                approach in gpu_ops[operation] and
                'interpolator' in gpu_ops[operation][approach]):
                return float(gpu_ops[operation][approach]['interpolator'](size_or_dims))

        return np.nan

    def calculate_critical_path_time(self, N: int, R: int, D: int, gpu: str, network_scenario: str, approach: str) -> Dict:
        """Calculate the total critical path time for given parameters."""

        results = {
            'N': N, 'R': R, 'D': D, 'GPU': gpu, 'Network': network_scenario, 'Approach': approach,
            'transfer_NxN': np.nan, 'matmul_NxN': np.nan, 'matmul_NxN_x_Nx1': np.nan,
            'matmul_RxN_x_NxN': np.nan, 'inverse_NxN': np.nan, 'transfer_Nx1': np.nan,
            'transfer_RxN': np.nan, 'addition_NxN': np.nan, 'addition_Nx1': np.nan,
            'addition_RxN': np.nan, 'randgen_NxN': np.nan, 'randgen_Nx1': np.nan,
            'decentralized_protocol': np.nan, 'total_time': np.nan
        }

        # Calculate transfer times
        results['transfer_NxN'] = self.calculate_transfer_time((N, N), network_scenario)
        results['transfer_Nx1'] = self.calculate_transfer_time((N, 1), network_scenario)
        results['transfer_RxN'] = self.calculate_transfer_time((R, N), network_scenario)

        # Calculate computation times
        results['matmul_NxN'] = self.get_computation_time(gpu, 'square_matmul', N, approach)
        results['inverse_NxN'] = self.get_computation_time(gpu, 'matrix_inverse', N, 'floating-point')
        results['matmul_NxN_x_Nx1'] = self.get_computation_time(gpu, 'vector_matmul', N, approach)

        # Calculate addition times - with robust estimation for missing data
        results['addition_NxN'] = self.get_computation_time(gpu, 'matrix_addition', N, approach)
        
        # If matrix_addition is missing, estimate from matrix multiplication (typically ~10x faster)
        if np.isnan(results['addition_NxN']):
            matmul_time = results['matmul_NxN']
            if not np.isnan(matmul_time):
                # Matrix addition is typically much faster than multiplication
                # Rough estimate: addition is ~1/10th the time of multiplication
                results['addition_NxN'] = matmul_time * 0.1
                print(f"    Estimated addition_NxN for {gpu} from matmul: {results['addition_NxN']:.6f}s")
        
        vector_add_time = self.estimate_vector_addition_from_rectangular(gpu, N, approach)
        results['addition_Nx1'] = vector_add_time

        # Calculate random generation times
        results['randgen_NxN'] = self.get_computation_time(gpu, 'random_generation', N, approach)
        results['randgen_Nx1'] = self.get_computation_time(gpu, 'random_generation', N, approach)

        # Calculate decentralized participation protocol time
        results['decentralized_protocol'] = self.calculate_decentralized_participation_protocol_time(N, R, D, network_scenario)

        # For R×N × N×N multiplication, try to get rectangular data or estimate
        rect_time = self.get_computation_time(gpu, 'rectangular_matmul', (R, N), approach)
        if np.isnan(rect_time):
            nn_time = results['matmul_NxN']
            if not np.isnan(nn_time):
                rect_time = nn_time * (R / N)
        results['matmul_RxN_x_NxN'] = rect_time

        # For R×N addition, try to get rectangular data or estimate
        rect_add_time = self.get_computation_time(gpu, 'rectangular_addition', (R, N), approach)
        if np.isnan(rect_add_time):
            # If rectangular addition not available, estimate from square matrix addition
            nn_add_time = results['addition_NxN']
            if not np.isnan(nn_add_time):
                rect_add_time = nn_add_time * (R / N)
            else:
                # If matrix addition completely missing, estimate from matrix multiplication
                nn_matmul_time = results['matmul_NxN']
                if not np.isnan(nn_matmul_time):
                    # Addition is ~10x faster than multiplication, scale by element ratio
                    rect_add_time = nn_matmul_time * 0.1 * (R / N)
                    print(f"    Estimated addition_RxN for {gpu} from matmul: {rect_add_time:.6f}s")
        results['addition_RxN'] = rect_add_time

        # Improved estimation logic for missing addition times
        if np.isnan(results['addition_NxN']) and not np.isnan(results['addition_RxN']):
            results['addition_NxN'] = results['addition_RxN'] * (N / R)

        if np.isnan(results['addition_Nx1']) and not np.isnan(results['addition_RxN']):
            results['addition_Nx1'] = results['addition_RxN'] * (1 / R)

        if np.isnan(results['addition_Nx1']) and not np.isnan(results['addition_NxN']):
            results['addition_Nx1'] = results['addition_NxN'] * (1 / N)

        if np.isnan(results['addition_RxN']) and not np.isnan(results['addition_NxN']):
            results['addition_RxN'] = results['addition_NxN'] * (R / N)
            
        # Final fallback: if addition times are still missing, estimate from multiplication
        if np.isnan(results['addition_Nx1']):
            vector_matmul_time = results['matmul_NxN_x_Nx1']
            if not np.isnan(vector_matmul_time):
                results['addition_Nx1'] = vector_matmul_time * 0.01
                print(f"    Estimated addition_Nx1 for {gpu} from vector matmul: {results['addition_Nx1']:.6f}s")
        
        if np.isnan(results['addition_RxN']):
            rect_matmul_time = results['matmul_RxN_x_NxN']
            if not np.isnan(rect_matmul_time):
                results['addition_RxN'] = rect_matmul_time * 0.01
                print(f"    Estimated addition_RxN for {gpu} from rect matmul: {results['addition_RxN']:.6f}s")

        # Calculate total time according to updated critical path formula
        if all(not np.isnan(results[key]) for key in ['transfer_NxN', 'matmul_NxN', 'matmul_NxN_x_Nx1',
                                                     'matmul_RxN_x_NxN', 'inverse_NxN', 'transfer_Nx1',
                                                     'transfer_RxN', 'addition_NxN', 'addition_Nx1', 'addition_RxN',
                                                     'randgen_NxN', 'randgen_Nx1', 'decentralized_protocol']):

            total = (6 * results['transfer_NxN'] +
                    3 * results['transfer_Nx1'] +
                    2 * results['transfer_RxN'] +
                    1 * results['decentralized_protocol'] +
                    6 * results['matmul_NxN'] +
                    3 * results['matmul_NxN_x_Nx1'] +
                    3 * results['matmul_RxN_x_NxN'] +
                    1 * results['inverse_NxN'] +
                    6 * results['addition_NxN'] +
                    2 * results['addition_Nx1'] +
                    2 * results['addition_RxN'] +
                    3 * results['randgen_NxN'] +
                    1 * results['randgen_Nx1'])

            results['total_time'] = total

        return results

    def generate_all_combinations(self) -> pd.DataFrame:
        """Generate results for all combinations of N × R × D × GPU × Network × Approach."""

        N_values = [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 50000]
        R_values = [100, 500, 1000]
        D_values = [2, 3, 4, 5, 8, 10]
        gpu_list = list(self.gpu_timings.keys())
        network_scenarios = list(self.network_scenarios.keys())
        approaches = ['floating-point', 'fixed-point']

        total_combinations = len(N_values) * len(R_values) * len(D_values) * len(gpu_list) * len(network_scenarios) * len(approaches)
        print(f"Generating results for {total_combinations} combinations...")

        all_results = []

        for N, R, D, gpu, network, approach in itertools.product(N_values, R_values, D_values, gpu_list, network_scenarios, approaches):
            result = self.calculate_critical_path_time(N, R, D, gpu, network, approach)
            all_results.append(result)

        return pd.DataFrame(all_results)

    def print_available_data_ranges(self):
        """Print the available data ranges for each system."""
        print("\n" + "="*80)
        print("AVAILABLE BENCHMARK DATA RANGES")
        print("="*80)

        for system in self.gpu_timings:
            print(f"\n{system}:")
            for operation in self.gpu_timings[system]:
                for approach in self.gpu_timings[system][operation]:
                    if 'data_points' in self.gpu_timings[system][operation][approach]:
                        data_points = self.gpu_timings[system][operation][approach]['data_points']
                        if data_points:
                            sizes = [point[0] for point in data_points]
                            min_size, max_size = min(sizes), max(sizes)
                            print(f"  {operation} ({approach}): {min_size} to {max_size} (data points: {len(data_points)})")

    def print_summary_statistics(self, results_df: pd.DataFrame):
        """Print summary statistics of the results."""
        print("\n" + "="*80)
        print("ENHANCED CRITICAL PATH TIME ANALYSIS SUMMARY")
        print("="*80)

        valid_results = results_df.dropna(subset=['total_time'])
        incomplete_results = results_df[results_df['total_time'].isna()]

        print(f"\nData Completeness Analysis:")
        print(f"  Valid combinations: {len(valid_results)} out of {len(results_df)}")
        print(f"  Incomplete combinations: {len(incomplete_results)} out of {len(results_df)}")

        if len(valid_results) == 0:
            print("\n⚠️  No valid results found!")
            return

        print(f"\nOverall execution time statistics:")
        print(f"  Minimum: {valid_results['total_time'].min():.3f} seconds")
        print(f"  Maximum: {valid_results['total_time'].max():.3f} seconds")
        print(f"  Mean: {valid_results['total_time'].mean():.3f} seconds")

        # Best and worst cases
        best_case = valid_results.loc[valid_results['total_time'].idxmin()]
        worst_case = valid_results.loc[valid_results['total_time'].idxmax()]

        print(f"\nBest case: N={best_case['N']}, R={best_case['R']}, D={best_case['D']}, System={best_case['GPU']}, Network={best_case['Network']}, Approach={best_case['Approach']}")
        print(f"  Total time: {best_case['total_time']:.3f} seconds")

        print(f"\nWorst case: N={worst_case['N']}, R={worst_case['R']}, D={worst_case['D']}, System={worst_case['GPU']}, Network={worst_case['Network']}, Approach={worst_case['Approach']}")
        print(f"  Total time: {worst_case['total_time']:.3f} seconds")

        # Analysis by System
        print(f"\nPerformance by System:")
        system_stats = valid_results.groupby('GPU')['total_time'].agg(['mean', 'min', 'max', 'count'])
        for system in system_stats.index:
            stats = system_stats.loc[system]
            print(f"  {system}: {stats['mean']:.3f}s (min: {stats['min']:.3f}s, max: {stats['max']:.3f}s, samples: {stats['count']})")

        # Analysis by Network
        print(f"\nPerformance by Network:")
        network_stats = valid_results.groupby('Network')['total_time'].agg(['mean', 'min', 'max'])
        for network in network_stats.index:
            stats = network_stats.loc[network]
            measured_indicator = "(Measured)" if "Measured" in network else "(Base)"
            print(f"  {network} {measured_indicator}: {stats['mean']:.3f}s")

    # ============== VISUALIZATION FUNCTIONS ==============

    def calculate_time_components(self, df):
        """Calculate time components using the updated LCA critical path formula."""
        # Transfer times
        df['total_transfer'] = (6 * df['transfer_NxN'] +
                               3 * df['transfer_Nx1'] +
                               2 * df['transfer_RxN'] +
                               1 * df['decentralized_protocol'])

        # Computation times
        df['total_computation'] = (6 * df['matmul_NxN'] +
                                  3 * df['matmul_NxN_x_Nx1'] +
                                  3 * df['matmul_RxN_x_NxN'] +
                                  1 * df['inverse_NxN'] +
                                  6 * df['addition_NxN'] +
                                  2 * df['addition_Nx1'] +
                                  2 * df['addition_RxN'] +
                                  3 * df['randgen_NxN'] +
                                  1 * df['randgen_Nx1'])

        # Calculate percentages
        df['transfer_percentage'] = (df['total_transfer'] / df['total_time']) * 100
        df['computation_percentage'] = (df['total_computation'] / df['total_time']) * 100

        return df

    def plot_network_impact_all_configs(self, df, r_value=100, figsize=(20, 16)):
        """Create a grid of plots showing network impact for all system/approach combinations."""
        filtered_df = df[df['R'] == r_value].copy()

        if filtered_df.empty:
            print(f"⚠️  No data available for R={r_value}")
            return

        # Define GPU order and add CPU systems
        desired_gpu_order = [
            'NVIDIA A40', 'A40',
            'NVIDIA A100', 'A100', 'A100 80GB PCIe', 'NVIDIA A100 80GB PCIe',
            'NVIDIA H100', 'H100', 'H100 80GB PCIe', 'NVIDIA H100 80GB PCIe', 'H100 NVL',
            'NVIDIA H200', 'H200', 'H200 141GB', 'NVIDIA H200 141GB'
        ]
        available_systems = filtered_df['GPU'].unique()

        system_types = []
        for system in desired_gpu_order:
            if system in available_systems and system not in system_types:
                system_types.append(system)

        # Add CPU systems
        cpu_systems = [system for system in available_systems if 'CPU' in system and system not in system_types]
        system_types.extend(cpu_systems)

        # Add any remaining systems
        for system in available_systems:
            if system not in system_types:
                system_types.append(system)

        approaches = filtered_df['Approach'].unique()

        n_rows, n_cols = len(system_types), len(approaches)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        # Separate base and measured network scenarios
        base_scenarios = ['Residential', 'Enterprise', 'DataCenter']
        measured_scenarios = [net for net in filtered_df['Network'].unique() if net not in base_scenarios]
        
        # Create colors
        base_colors = ['#E74C3C', '#3498DB', '#2ECC71']
        measured_colors = ['#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
        
        network_colors = {}
        for i, scenario in enumerate(base_scenarios):
            if i < len(base_colors):
                network_colors[scenario] = base_colors[i]
        
        for i, scenario in enumerate(measured_scenarios):
            if i < len(measured_colors):
                network_colors[scenario] = measured_colors[i]

        print(f"Found {len(base_scenarios)} base networks: {base_scenarios}")
        print(f"Found {len(measured_scenarios)} measured networks: {measured_scenarios}")

        for i, system in enumerate(system_types):
            for j, approach in enumerate(approaches):
                ax = axes[i, j]
                combo_df = filtered_df[(filtered_df['GPU'] == system) & (filtered_df['Approach'] == approach)]

                if combo_df.empty:
                    ax.set_title(f'{system.replace("NVIDIA ", "")}\n{approach}\n(No Data)')
                    ax.set_xlabel('Matrix Size (N)')
                    ax.set_ylabel('Transfer %')
                    continue

                # Plot base scenarios
                for scenario in base_scenarios:
                    scenario_data = combo_df[combo_df['Network'] == scenario]
                    if not scenario_data.empty:
                        ax.plot(scenario_data['N'], scenario_data['transfer_percentage'], 
                               color=network_colors[scenario], marker='o', linewidth=2.5, 
                               linestyle='-', label=scenario)
                
                # Plot measured scenarios
                for scenario in measured_scenarios:
                    scenario_data = combo_df[combo_df['Network'] == scenario]
                    if not scenario_data.empty:
                        ax.plot(scenario_data['N'], scenario_data['transfer_percentage'], 
                               color=network_colors.get(scenario, '#95A5A6'), marker='s', linewidth=2.5, 
                               linestyle='--', label=scenario, alpha=0.8)

                system_type = "CPU" if "CPU" in system else "GPU"
                ax.set_title(f'{system.replace("NVIDIA ", "")} ({system_type})\n{approach}', fontweight='bold')
                ax.set_xlabel('Matrix Size (N)')
                ax.set_ylabel('Transfer %')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                ax.set_ylim(0, 105)
                
                # Remove legend from individual plots
                legend = ax.get_legend()
                if legend:
                    legend.remove()

        # Create master legend
        legend_elements = []
        for scenario in base_scenarios:
            if scenario in network_colors:
                legend_elements.append(plt.Line2D([0], [0], color=network_colors[scenario], 
                                                linestyle='-', marker='o', label=f'{scenario} (Base)'))
        
        for scenario in measured_scenarios:
            if scenario in network_colors:
                legend_elements.append(plt.Line2D([0], [0], color=network_colors[scenario], 
                                                linestyle='--', marker='s', label=f'{scenario} (Measured)', alpha=0.8))
        
        if legend_elements:
            fig.legend(legend_elements, [elem.get_label() for elem in legend_elements], 
                      bbox_to_anchor=(1.01, 0.9), loc='upper left', title='Network Type')
        
        plt.suptitle(f'Enhanced Network Impact Analysis: All Systems & Measured Networks (R={r_value})',
                     fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        plt.savefig('enhanced-network-analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_d_parameter_effect(self, df, r_value=100, figsize=(10, 6)):
        """Analyze the effect of parameter D on total execution time."""
        filtered_df = df[df['R'] == r_value].copy()

        if 'D' not in filtered_df.columns or filtered_df.empty:
            print("Warning: 'D' parameter not found or no data available.")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        d_effect_total = filtered_df.groupby(['N', 'D'])['total_time'].mean().reset_index()

        unique_d_values = sorted(d_effect_total['D'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_d_values)))

        for i, d_val in enumerate(unique_d_values):
            d_data = d_effect_total[d_effect_total['D'] == d_val]
            ax.plot(d_data['N'], d_data['total_time'],
                    marker='o', linewidth=2.5, color=colors[i],
                    label=f'D = {d_val}')

        ax.set_xlabel('Matrix Size (N)')
        ax.set_ylabel('Average Total Time (seconds)')
        ax.set_title('Effect of Parameter D on Total Execution Time')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(title='D Parameter')

        plt.tight_layout()
        plt.savefig('d-parameter-analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_protocol_overhead_analysis(self, df, r_value=100, figsize=(10, 6)):
        """Analyze how parameter D affects protocol time."""
        filtered_df = df[df['R'] == r_value].copy()

        if 'decentralized_protocol' not in filtered_df.columns or 'D' not in filtered_df.columns or filtered_df.empty:
            print("Warning: Cannot perform protocol analysis - missing columns or no data.")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        protocol_df = filtered_df.groupby(['N', 'D'])['decentralized_protocol'].mean().reset_index()

        unique_n_values = sorted(protocol_df['N'].unique())
        colors = plt.cm.plasma(np.linspace(0, 1, len(unique_n_values)))

        for i, n_val in enumerate(unique_n_values):
            n_data = protocol_df[protocol_df['N'] == n_val]
            ax.plot(n_data['D'], n_data['decentralized_protocol'],
                    marker='o', linewidth=2.5, color=colors[i],
                    label=f'N = {n_val}')

        ax.set_xlabel('Supply Chain Density (D)')
        ax.set_ylabel('Protocol Time (seconds)')
        ax.set_title('Effect of D on Protocol Time')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Supply Chain Size (N)', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig('d-parameter-protocol-effect.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_complete_analysis(self):
        """Run the complete analysis including calculation and visualization."""
        print("="*80)
        print("ENHANCED LCA CRITICAL PATH ANALYSIS - COMPLETE WORKFLOW")
        print("="*80)
        
        # Show available data ranges
        self.print_available_data_ranges()
        
        # Generate all combinations
        results_df = self.generate_all_combinations()

        # Print summary statistics
        self.print_summary_statistics(results_df)

        # Save detailed results to CSV
        output_file = 'enhanced_lca_critical_path_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

        # Only proceed with visualization if we have valid results
        valid_results = results_df.dropna(subset=['total_time'])
        if len(valid_results) == 0:
            print("\n⚠️  No valid results for visualization. Check benchmark data completeness.")
            return

        # Prepare data for visualization
        results_df = self.calculate_time_components(results_df)

        print(f"\nData Overview for Visualization:")
        print(f"  Total combinations: {len(results_df)}")
        print(f"  Valid results: {len(valid_results)}")
        print(f"  Systems: {results_df['GPU'].nunique()} ({', '.join(results_df['GPU'].unique())})")
        print(f"  Network scenarios: {results_df['Network'].nunique()} ({', '.join(results_df['Network'].unique())})")

        # Run visualizations
        print("\n--- Running Enhanced Network Impact Analysis ---")
        self.plot_network_impact_all_configs(results_df, r_value=1000)

        print("\n--- Running D Parameter Analysis ---")
        self.plot_d_parameter_effect(results_df, r_value=1000)
        self.plot_protocol_overhead_analysis(results_df, r_value=1000)

        print("\n" + "="*80)
        print("COMPLETE ANALYSIS FINISHED!")
        print("Generated files:")
        print("  • enhanced_lca_critical_path_results.csv - Complete results")
        print("  • enhanced-network-analysis.png - Network impact analysis")
        print("  • d-parameter-analysis.png - D parameter effects")
        print("  • d-parameter-protocol-effect.png - Protocol overhead analysis")
        print("="*80)

def main():
    """Main function to run the complete enhanced critical path analysis."""
    
    # Initialize calculator - reads from timing_files AND local-results
    calculator = CompleteLCACriticalPathCalculator('timing_files', 'local-results')
    
    # Run the complete analysis workflow
    calculator.run_complete_analysis()

if __name__ == "__main__":
    main()