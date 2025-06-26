"""
SECURE MULTI-PARTY COMPUTATION ORCHESTRATOR

SECURITY-CRITICAL RESPONSIBILITIES:
- Generate matrices and create additive secret shares (x = x0 + x1 + x2)
- Distribute ONLY the appropriate share to each party (party i gets x_i only)
- Coordinate protocol phases without seeing intermediate results
- Perform public operations (like matrix inversion) on reconstructed matrices
- Collect shares ONLY for final reconstruction or public operations

ORCHESTRATOR NEVER:
- Collects shares during secure computation phases
- Reconstructs intermediate results unless required for public operations
- Violates the security properties of the protocol
"""

import os
import time
import hashlib
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.sparse import random as sparse_random
import grpc
from grpc import aio as grpc_aio

# Import generated protobuf files
import smpc_pb2
import smpc_pb2_grpc
from common.utils import (
    matrix_to_proto, proto_to_matrix, is_invertible, 
    create_shares_float, reconstruct_shares, calculate_error_metrics
)

class SMPCOrchestrator:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.http_port = int(os.getenv('HTTP_PORT', 8080))
        self.grpc_port = int(os.getenv('GRPC_PORT', 50051))
        self.party_addresses = os.getenv('PARTY_GRPC_ADDRESSES', '').split(',')
        
        # State management
        self.computations = {}
        self.results_df = pd.DataFrame()
        self.csv_file = '/app/results/floating_point_results.csv'
        
        # Load existing results
        if os.path.exists(self.csv_file):
            self.results_df = pd.read_csv(self.csv_file)
        
        # Setup routes
        self.setup_routes()
        
        # gRPC clients for parties
        self.party_clients = {}
        self.init_party_clients()
    
    def init_party_clients(self):
        """Initialize gRPC clients for each computation party."""
        for i, address in enumerate(self.party_addresses):
            if address.strip():
                channel = grpc.insecure_channel(address.strip())
                self.party_clients[i+1] = smpc_pb2_grpc.PartyComputationServiceStub(channel)
    
    def setup_routes(self):
        """Setup HTTP routes for the web interface and API."""
        
        @self.app.route('/')
        def index():
            return render_template_string(WEB_INTERFACE_HTML)
        
        @self.app.route('/api/start_computation', methods=['POST'])
        def start_computation():
            try:
                config = request.json
                result = self.run_computation(config)
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/get_results', methods=['GET'])
        def get_results():
            try:
                return jsonify({
                    'success': True,
                    'results': self.results_df.to_dict('records') if not self.results_df.empty else []
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/download_results', methods=['GET'])
        def download_results():
            try:
                if os.path.exists(self.csv_file):
                    return send_file(self.csv_file, as_attachment=True, download_name='results.csv')
                else:
                    return jsonify({'success': False, 'error': 'No results file found'}), 404
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def run_computation(self, config):
        """Execute the complete SMPC computation."""
        print(f"Starting computation with config: {config}")
        
        # Generate unique run ID
        config_str = str(sorted(config.items()))
        run_id = hashlib.md5(config_str.encode()).hexdigest()
        
        # Check if already computed
        if not self.results_df.empty and run_id in self.results_df.get('run_id', pd.Series()).values:
            return {'success': True, 'message': 'Computation already exists', 'run_id': run_id}
        
        # Set random seed for reproducibility
        seed = int(run_id, 16) % (2**32)
        np.random.seed(seed)
        
        try:
            # Generate matrices
            matrices = self.generate_matrices(config)
            if not matrices:
                return {'success': False, 'error': 'Matrix generation failed'}
            
            # Run secure computation
            result_metrics = self.execute_secure_computation(matrices, config, run_id)
            
            # Save results
            result_metrics.update(config)
            result_metrics['run_id'] = run_id
            result_metrics['seed'] = seed
            
            new_row = pd.DataFrame([result_metrics])
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            self.results_df.to_csv(self.csv_file, index=False)
            
            return {'success': True, 'metrics': result_metrics, 'run_id': run_id}
            
        except Exception as e:
            print(f"Computation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_matrices(self, config):
        """Generate all required matrices for the computation."""
        try:
            N = config['N']
            R = config['R']
            
            # Generate sparse matrix A
            sparse_matrix = sparse_random(
                N, N, 
                density=config['A_DENSITY'], 
                format='dok',
                data_rvs=lambda size: -np.random.rand(size)
            )
            A_float = sparse_matrix.toarray()
            
            # Add outliers if configured
            num_outliers = 0
            if config['INTRODUCE_OUTLIERS']:
                mask = (np.random.rand(N, N) < config['OUTLIER_PROBABILITY']) & (np.eye(N) == 0)
                num_outliers = np.sum(mask)
                outlier_values = np.random.uniform(
                    config['OUTLIER_RANGE_MIN'], 
                    config['OUTLIER_RANGE_MAX'], 
                    size=num_outliers
                )
                A_float[mask] = outlier_values
            
            # Ensure diagonal is 1 for invertibility
            np.fill_diagonal(A_float, 1)
            
            # Check invertibility
            is_inv, cond_A = is_invertible(A_float)
            if not is_inv:
                print("Generated matrix A is not invertible")
                return None
            
            # Generate orthogonal transformation matrix T
            T_SCALING_FACTOR = 10000
            MAX_COND_T = 10
            
            while True:
                z = np.random.randn(N, N)
                q, _ = np.linalg.qr(z)
                T_int = np.round(q * T_SCALING_FACTOR).astype(np.int64)
                cond_t = np.linalg.cond(T_int.astype(np.float64))
                
                if cond_t < 1 / np.finfo(np.float64).eps and cond_t < MAX_COND_T:
                    break
            
            T_float = T_int.astype(np.float64)
            
            # Generate secret matrix B and vector f
            B_int = np.random.randint(
                config['B_INT_RANGE_MIN'], 
                config['B_INT_RANGE_MAX'], 
                size=(R, N), 
                dtype=np.int64
            )
            f_int = np.zeros((N, 1), dtype=np.int64)
            f_int[0, 0] = 1
            
            B_float = B_int.astype(np.float64)
            f_float = f_int.astype(np.float64)
            
            # Compute plaintext results for verification
            A_inv_plaintext = np.linalg.inv(A_float)
            s_plaintext_vec = A_inv_plaintext @ f_float
            diag_s_plaintext = np.diag(s_plaintext_vec.flatten())
            final_plaintext_result = B_float @ diag_s_plaintext
            
            return {
                'A': A_float,
                'T': T_float, 
                'B': B_float,
                'f': f_float,
                'A_inv_plaintext': A_inv_plaintext,
                'final_plaintext_result': final_plaintext_result,
                'matrix_cond_A': float(cond_A),
                'matrix_cond_T': float(cond_t),
                'num_outliers_injected': num_outliers
            }
            
        except Exception as e:
            print(f"Matrix generation failed: {e}")
            return None
    
    def execute_secure_computation(self, matrices, config, run_id):
        """Execute the secure multi-party computation protocol."""
        start_time = time.time()
        
        try:
            # Create shares for input matrices
            A_shares = create_shares_float(matrices['A'], config['USE_ADAPTIVE_SHARING'], config)
            T_shares = create_shares_float(matrices['T'], config['USE_ADAPTIVE_SHARING'], config)
            B_shares = create_shares_float(matrices['B'], config['USE_ADAPTIVE_SHARING'], config)
            f_shares = create_shares_float(matrices['f'], config['USE_ADAPTIVE_SHARING'], config)
            
            # Distribute shares to parties
            self.distribute_shares(run_id, 'A', A_shares)
            self.distribute_shares(run_id, 'T', T_shares)
            self.distribute_shares(run_id, 'B', B_shares)
            self.distribute_shares(run_id, 'f', f_shares)
            
            # Step 1: Compute A*T securely
            AT_shares = self.secure_matrix_multiply(run_id, 'A', 'T', 'AT')
            
            # Step 2: Reconstruct A*T and compute inverse (public operation)
            AT_reconstructed = self.reconstruct_matrix(run_id, 'AT')
            AT_inv_computed = np.linalg.inv(AT_reconstructed)
            
            # Step 3: Create shares for (A*T)^(-1) and distribute
            AT_inv_shares = create_shares_float(AT_inv_computed, config['USE_ADAPTIVE_SHARING'], config)
            self.distribute_shares(run_id, 'AT_inv', AT_inv_shares)
            
            # Step 4: Compute A^(-1) = T * (A*T)^(-1) securely with randomization
            A_inv_shares = self.secure_matrix_multiply_with_randomization(run_id, 'T', 'AT_inv', 'A_inv_rand', config)
            
            # Step 5: Compute s = A^(-1) * f securely with randomization
            s_shares = self.secure_matrix_multiply_with_randomization(run_id, 'A_inv_rand', 'f', 's_rand', config)
            
            # Step 6: Create diagonal matrix from s vector
            self.create_diagonal_matrix(run_id, 's_rand', 'diag_s')
            
            # Step 7: Compute final result B * diag(s) securely (no randomization needed for final step)
            final_shares = self.secure_matrix_multiply(run_id, 'B', 'diag_s', 'final')
            
            # Step 8: Reconstruct final result
            final_computed = self.reconstruct_matrix(run_id, 'final')
            
            # Calculate metrics
            duration = time.time() - start_time
            error_metrics = calculate_error_metrics(final_computed, matrices['final_plaintext_result'])
            
            result_metrics = {
                'float_point_time_s': duration,
                'matrix_cond_A': matrices['matrix_cond_A'],
                'matrix_cond_T': matrices['matrix_cond_T'],
                'num_outliers_injected': matrices['num_outliers_injected']
            }
            
            # Add error metrics with float_ prefix
            for key, value in error_metrics.items():
                result_metrics[f'float_{key}'] = value
            
            return result_metrics
            
        except Exception as e:
            print(f"Secure computation failed: {e}")
            raise e
    
    def distribute_shares(self, computation_id, matrix_name, shares):
        """Distribute matrix shares to computation parties via gRPC - each party gets ONLY their share."""
        for party_id, share in enumerate(shares, 1):
            try:
                share_msg = smpc_pb2.ShareDistribution(
                    computation_id=computation_id,
                    matrix_name=matrix_name,
                    share=matrix_to_proto(share),
                    share_index=party_id-1
                )
                
                response = self.party_clients[party_id].ReceiveShares(share_msg)
                if not response.success:
                    raise Exception(f"Failed to send share to party {party_id}: {response.message}")
                
                print(f"Sent share {party_id-1} of {matrix_name} to party {party_id}")
                    
            except Exception as e:
                print(f"Error distributing share to party {party_id}: {e}")
                raise e
    
    def secure_matrix_multiply(self, computation_id, matrix_a, matrix_b, result_name):
        """Coordinate secure matrix multiplication between parties."""
        try:
            # Send multiplication request to all parties - they handle conversion themselves
            for party_id in self.party_clients:
                request = smpc_pb2.MatMulRequest(
                    computation_id=computation_id,
                    matrix_a_name=matrix_a,
                    matrix_b_name=matrix_b,
                    result_name=result_name
                )
                
                response = self.party_clients[party_id].SecureMatMul(request)
                if not response.success:
                    raise Exception(f"Party {party_id} matrix multiplication failed: {response.message}")
            
            print(f"Secure matrix multiplication {matrix_a} * {matrix_b} = {result_name} completed")
            return result_name
            
        except Exception as e:
            print(f"Secure matrix multiplication failed: {e}")
            raise e
    
    def secure_matrix_multiply_with_randomization(self, computation_id, matrix_a, matrix_b, result_name, config):
        """Coordinate secure matrix multiplication with automatic randomization."""
        try:
            # Send multiplication with randomization request to all parties
            for party_id in self.party_clients:
                request = smpc_pb2.MatMulRequest(
                    computation_id=computation_id,
                    matrix_a_name=matrix_a,
                    matrix_b_name=matrix_b,
                    result_name=result_name
                )
                
                # Add randomization parameters
                request.parameters.update({
                    'randomize': 'true',
                    'adaptive': str(config['USE_ADAPTIVE_SHARING']),
                    'minimum_noise_range_val': str(config['MINIMUM_NOISE_RANGE_VAL']),
                    'obfuscation_factor_min': str(config['OBFUSCATION_FACTOR_MIN']),
                    'obfuscation_factor_max': str(config['OBFUSCATION_FACTOR_MAX'])
                })
                
                response = self.party_clients[party_id].SecureMatMulWithRandomization(request)
                if not response.success:
                    raise Exception(f"Party {party_id} matrix multiplication with randomization failed: {response.message}")
            
            print(f"Secure matrix multiplication with randomization {matrix_a} * {matrix_b} = {result_name} completed")
            return result_name
            
        except Exception as e:
            print(f"Secure matrix multiplication with randomization failed: {e}")
            raise e
    
    def coordinate_replicated_conversion(self, computation_id, matrix_name):
        """Coordinate the conversion from additive to replicated shares."""
        # Parties handle this themselves - no coordination needed
        pass
    
    def coordinate_replicated_restoration(self, computation_id, matrix_name):
        """Coordinate the conversion back from additive to replicated shares after randomization."""
        # Parties handle this themselves - no coordination needed  
        pass
    
    def reconstruct_matrix(self, computation_id, matrix_name):
        """
        Reconstruct a matrix by collecting shares from all parties.
        SECURITY CRITICAL: Only used for final results and public operations (like matrix inversion).
        """
        try:
            shares = []
            
            print(f"🔒 SECURITY CHECK: Reconstructing {matrix_name} - this should only happen for final results or public operations!")
            
            # Collect shares from all parties - ONLY for final reconstruction
            for party_id in self.party_clients:
                request = smpc_pb2.RevealRequest(
                    computation_id=computation_id,
                    matrix_name=matrix_name
                )
                
                response = self.party_clients[party_id].ReturnShares(request)
                if not response.success:
                    raise Exception(f"Failed to get share from party {party_id}: {response.message}")
                
                share_matrix = proto_to_matrix(response.result)
                shares.append(share_matrix)
                print(f"Received share from party {party_id} for final reconstruction")
            
            # Reconstruct the matrix
            reconstructed = reconstruct_shares(shares)
            print(f"Matrix {matrix_name} reconstructed from {len(shares)} shares")
            return reconstructed
            
        except Exception as e:
            print(f"Matrix reconstruction failed: {e}")
            raise e
    
            print(f"Diagonal matrix created: {vector_matrix} -> {diag_matrix}")
            
        except Exception as e:
            print(f"Diagonal matrix creation failed: {e}")
            raise e
    
    def create_diagonal_matrix(self, computation_id, vector_matrix, diag_matrix):
        """Request parties to create diagonal matrices from vector shares."""
        try:
            for party_id in self.party_clients:
                request = smpc_pb2.ComputationStep(
                    computation_id=computation_id,
                    operation="diagonal",
                    input_matrices=[vector_matrix],
                    output_matrix=diag_matrix
                )
                
                response = self.party_clients[party_id].CreateDiagonal(request)
                if not response.success:
                    raise Exception(f"Party {party_id} diagonal creation failed: {response.message}")
            
            print(f"Diagonal matrix created: {vector_matrix} -> {diag_matrix}")
            
        except Exception as e:
            print(f"Diagonal matrix creation failed: {e}")
            raise e
    
    def run(self):
        """Start the orchestrator HTTP server."""
        print(f"Starting SMPC Orchestrator on port {self.http_port}")
        print(f"Party addresses: {self.party_addresses}")
        self.app.run(host='0.0.0.0', port=self.http_port, debug=False)

# HTML template for the web interface
WEB_INTERFACE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>SMPC Orchestrator</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
        button:hover { background-color: #0056b3; }
        .results { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 4px; }
        .error { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin-top: 10px; }
        .success { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin-top: 10px; }
        .loading { color: #007bff; margin-top: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Secure Multi-Party Computation Orchestrator</h1>
        
        <form id="computationForm">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <div class="form-group">
                        <label for="N">Matrix Dimension (N):</label>
                        <input type="number" id="N" name="N" value="100" min="10" max="2000">
                    </div>
                    
                    <div class="form-group">
                        <label for="R">Row Dimension (R):</label>
                        <input type="number" id="R" name="R" value="100" min="10" max="2000">
                    </div>
                    
                    <div class="form-group">
                        <label for="A_DENSITY">Matrix Density:</label>
                        <input type="number" id="A_DENSITY" name="A_DENSITY" value="0.1" min="0.01" max="1.0" step="0.01">
                    </div>
                    
                    <div class="form-group">
                        <label for="INTRODUCE_OUTLIERS">Introduce Outliers:</label>
                        <select id="INTRODUCE_OUTLIERS" name="INTRODUCE_OUTLIERS">
                            <option value="true">Yes</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="OUTLIER_PROBABILITY">Outlier Probability:</label>
                        <input type="number" id="OUTLIER_PROBABILITY" name="OUTLIER_PROBABILITY" value="0.11" min="0" max="1" step="0.01">
                    </div>
                </div>
                
                <div>
                    <div class="form-group">
                        <label for="OUTLIER_RANGE_MIN">Outlier Range Min:</label>
                        <input type="number" id="OUTLIER_RANGE_MIN" name="OUTLIER_RANGE_MIN" value="-5000">
                    </div>
                    
                    <div class="form-group">
                        <label for="OUTLIER_RANGE_MAX">Outlier Range Max:</label>
                        <input type="number" id="OUTLIER_RANGE_MAX" name="OUTLIER_RANGE_MAX" value="-2000">
                    </div>
                    
                    <div class="form-group">
                        <label for="B_INT_RANGE_MIN">B Matrix Range Min:</label>
                        <input type="number" id="B_INT_RANGE_MIN" name="B_INT_RANGE_MIN" value="100">
                    </div>
                    
                    <div class="form-group">
                        <label for="B_INT_RANGE_MAX">B Matrix Range Max:</label>
                        <input type="number" id="B_INT_RANGE_MAX" name="B_INT_RANGE_MAX" value="5000">
                    </div>
                    
                    <div class="form-group">
                        <label for="USE_ADAPTIVE_SHARING">Use Adaptive Sharing:</label>
                        <select id="USE_ADAPTIVE_SHARING" name="USE_ADAPTIVE_SHARING">
                            <option value="true">Yes</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 20px;">
                <div class="form-group">
                    <label for="MINIMUM_NOISE_RANGE_VAL">Minimum Noise Range:</label>
                    <input type="number" id="MINIMUM_NOISE_RANGE_VAL" name="MINIMUM_NOISE_RANGE_VAL" value="2" min="0.1" step="0.1">
                </div>
                
                <div class="form-group">
                    <label for="OBFUSCATION_FACTOR_MIN">Obfuscation Factor Min:</label>
                    <input type="number" id="OBFUSCATION_FACTOR_MIN" name="OBFUSCATION_FACTOR_MIN" value="0.1" min="0.01" step="0.01">
                </div>
                
                <div class="form-group">
                    <label for="OBFUSCATION_FACTOR_MAX">Obfuscation Factor Max:</label>
                    <input type="number" id="OBFUSCATION_FACTOR_MAX" name="OBFUSCATION_FACTOR_MAX" value="0.5" min="0.01" step="0.01">
                </div>
            </div>
            
            <button type="submit">Start Computation</button>
            <button type="button" onclick="loadResults()">Load Results</button>
            <button type="button" onclick="downloadResults()">Download Results</button>
        </form>
        
        <div id="status"></div>
        <div id="results" class="results" style="display: none;"></div>
    </div>

    <script>
        document.getElementById('computationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const config = {};
            
            for (let [key, value] of formData.entries()) {
                if (key.includes('OUTLIERS') || key.includes('ADAPTIVE')) {
                    config[key] = value === 'true';
                } else if (key === 'PRECISION') {
                    config[key] = parseInt(value);
                } else {
                    config[key] = parseFloat(value);
                }
            }
            
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = '<div class="loading">Starting computation...</div>';
            
            try {
                const response = await fetch('/api/start_computation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusDiv.innerHTML = '<div class="success">Computation completed successfully!</div>';
                    displayResults(result.metrics);
                } else {
                    statusDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            }
        });
        
        function displayResults(metrics) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            let html = '<h3>Computation Results</h3><table>';
            html += '<tr><th>Metric</th><th>Value</th></tr>';
            
            for (const [key, value] of Object.entries(metrics)) {
                html += `<tr><td>${key}</td><td>${typeof value === 'number' ? value.toFixed(6) : value}</td></tr>`;
            }
            
            html += '</table>';
            resultsDiv.innerHTML = html;
        }
        
        async function loadResults() {
            try {
                const response = await fetch('/api/get_results');
                const result = await response.json();
                
                if (result.success && result.results.length > 0) {
                    displayAllResults(result.results);
                } else {
                    document.getElementById('status').innerHTML = '<div class="error">No results found</div>';
                }
            } catch (error) {
                document.getElementById('status').innerHTML = `<div class="error">Error loading results: ${error.message}</div>`;
            }
        }
        
        function displayAllResults(results) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            let html = '<h3>All Computation Results</h3>';
            html += `<p>Total experiments: ${results.length}</p>`;
            html += '<table><tr>';
            
            // Headers
            const keys = Object.keys(results[0]);
            keys.forEach(key => html += `<th>${key}</th>`);
            html += '</tr>';
            
            // Data rows
            results.forEach(result => {
                html += '<tr>';
                keys.forEach(key => {
                    const value = result[key];
                    html += `<td>${typeof value === 'number' ? value.toFixed(6) : value}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</table>';
            resultsDiv.innerHTML = html;
        }
        
        async function downloadResults() {
            try {
                const response = await fetch('/api/download_results');
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'smpc_results.csv';
                    a.click();
                    window.URL.revokeObjectURL(url);
                } else {
                    document.getElementById('status').innerHTML = '<div class="error">No results file available for download</div>';
                }
            } catch (error) {
                document.getElementById('status').innerHTML = `<div class="error">Download error: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    orchestrator = SMPCOrchestrator()
    orchestrator.run()