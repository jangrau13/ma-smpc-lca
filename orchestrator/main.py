"""
SECURE MULTI-PARTY COMPUTATION ORCHESTRATOR (CORRECTED PARALLEL DISPATCH)
"""

import os
import time
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.sparse import random as sparse_random
import grpc

# Import generated protobuf files
import smpc_pb2
import smpc_pb2_grpc

# --- Setup Enhanced Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ORCHESTRATOR] - %(message)s')

def matrix_to_proto(matrix):
    if matrix is None: return smpc_pb2.Matrix(data=[], rows=0, cols=0)
    return smpc_pb2.Matrix(data=matrix.flatten().astype(np.float64).tolist(), rows=matrix.shape[0], cols=matrix.shape[1])

def proto_to_matrix(proto_matrix):
    if not proto_matrix.rows or not proto_matrix.cols: return None
    return np.array(proto_matrix.data, dtype=np.float64).reshape(proto_matrix.rows, proto_matrix.cols)

def is_invertible(matrix, tolerance=1e-10):
    cond = np.linalg.cond(matrix.astype(np.float64))
    return cond < 1 / np.finfo(np.float64).eps, cond

def create_shares_float(matrix, adaptive, config):
    matrix_64 = matrix.astype(np.float64)
    if not adaptive:
        noise = config.get('MINIMUM_NOISE_RANGE_VAL', 2.0)
        s0 = np.random.uniform(-noise, noise, size=matrix_64.shape).astype(np.float64)
        s1 = np.random.uniform(-noise, noise, size=matrix_64.shape).astype(np.float64)
    else:
        max_vals = np.max(np.abs(matrix_64), axis=0) + 1e-9
        obf_min, obf_max = config.get('OBFUSCATION_FACTOR_MIN', 0.1), config.get('OBFUSCATION_FACTOR_MAX', 0.5)
        obfuscation = 1 + np.random.uniform(obf_min, obf_max, size=max_vals.shape)
        scale = np.maximum(max_vals * obfuscation, config.get('MINIMUM_NOISE_RANGE_VAL', 2.0))
        s0 = (np.random.rand(*matrix_64.shape) - 0.5) * 2 * scale
        s1 = (np.random.rand(*matrix_64.shape) - 0.5) * 2 * scale
    s2 = matrix_64 - s0 - s1
    return s0, s1, s2

def reconstruct_shares(shares):
    return sum(shares)

def calculate_error_metrics(computed, plaintext):
    if computed is None: return {k: np.nan for k in ['norm_error', 'max_abs_error', 'mean_abs_error', 'max_rel_error', 'mean_rel_error', 'snr_db']}
    error, p_float = computed.astype(np.float64) - plaintext.astype(np.float64), plaintext.astype(np.float64)
    signal_power, noise_power = np.mean(p_float**2), np.mean(error**2)
    return {
        'norm_error': float(np.linalg.norm(error)),
        'max_abs_error': float(np.max(np.abs(error))),
        'mean_abs_error': float(np.mean(np.abs(error))),
        'max_rel_error': float(np.max(np.abs(error) / np.maximum(np.abs(p_float), 1e-5))),
        'mean_rel_error': float(np.mean(np.abs(error) / np.maximum(np.abs(p_float), 1e-5))),
        'snr_db': float(10 * np.log10(signal_power / noise_power) if noise_power > 1e-12 else float('inf'))
    }

class SMPCOrchestrator:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.http_port = int(os.getenv('HTTP_PORT', 8080))
        self.party_addresses = os.getenv('PARTY_GRPC_ADDRESSES', '').split(',')
        
        self.results_df = pd.DataFrame()
        self.csv_file = '/app/results/floating_point_results.csv'
        if os.path.exists(self.csv_file): self.results_df = pd.read_csv(self.csv_file)
        
        self.setup_routes()
        
        self.party_clients = {}
        self.init_party_clients()
        self.grpc_executor = ThreadPoolExecutor(max_workers=len(self.party_clients) * 2)

    def init_party_clients(self):
        logging.info("Initializing gRPC clients...")
        for i, address in enumerate(self.party_addresses):
            if address.strip():
                channel = grpc.insecure_channel(address.strip())
                self.party_clients[i+1] = smpc_pb2_grpc.PartyComputationServiceStub(channel)
                logging.info(f"Connected to Party {i+1} at {address.strip()}")
    
    def setup_routes(self):
        @self.app.route('/')
        def index(): return render_template_string(WEB_INTERFACE_HTML)
        
        @self.app.route('/api/start_computation', methods=['POST'])
        def start_computation():
            try: return jsonify(self.run_computation(request.json))
            except Exception as e:
                logging.error(f"HTTP Error on /api/start_computation: {e}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/get_results', methods=['GET'])
        def get_results(): return jsonify({'success': True, 'results': self.results_df.to_dict('records')})
        
        @self.app.route('/api/download_results', methods=['GET'])
        def download_results():
            if os.path.exists(self.csv_file): return send_file(self.csv_file, as_attachment=True)
            return jsonify({'success': False, 'error': 'No results file'}), 404

    def _execute_on_all_parties(self, rpc_method_name, request_builder):
        """Executes an RPC on all parties in parallel."""
        futures = {self.grpc_executor.submit(getattr(client, rpc_method_name), request_builder(party_id)): party_id
                   for party_id, client in self.party_clients.items()}
        for future in as_completed(futures):
            party_id = futures[future]
            try:
                response = future.result()
                if not response.success: raise RuntimeError(f"Party {party_id} failed {rpc_method_name}: {response.message}")
            except Exception as exc:
                logging.error(f"gRPC call to Party {party_id} for {rpc_method_name} failed: {exc}", exc_info=True)
                raise

    def run_computation(self, config):
        run_id = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()
        logging.info(f"--- Starting new computation (ID: {run_id[:8]}) ---")
        logging.info(f"Configuration: {config}")
        
        if not self.results_df.empty and run_id in self.results_df.get('run_id', pd.Series()).values:
            logging.warning(f"Computation {run_id[:8]} already exists. Skipping.")
            return {'success': True, 'message': 'Computation already exists', 'run_id': run_id}
        
        np.random.seed(int(run_id, 16) % (2**32))
        
        try:
            matrices = self.generate_matrices(config)
            result_metrics = self.execute_secure_computation(matrices, config, run_id)
            
            result_metrics.update(config)
            result_metrics['run_id'] = run_id
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_metrics])], ignore_index=True)
            self.results_df.to_csv(self.csv_file, index=False)
            
            logging.info(f"--- Computation {run_id[:8]} completed successfully ---")
            return {'success': True, 'metrics': result_metrics, 'run_id': run_id}
        except Exception as e:
            logging.error(f"Computation {run_id[:8]} failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def generate_matrices(self, config):
        # This function remains the same, no changes needed
        logging.info("Step 0: Generating plaintext matrices...")
        try:
            N, R = config['N'], config['R']
            A_float = sparse_random(N, N, density=config['A_DENSITY'], format='dok', data_rvs=lambda s: -np.random.rand(s)).toarray()
            
            if config.get('INTRODUCE_OUTLIERS'):
                mask = (np.random.rand(N, N) < config['OUTLIER_PROBABILITY']) & (np.eye(N) == 0)
                num_outliers = np.sum(mask)
                A_float[mask] = np.random.uniform(config['OUTLIER_RANGE_MIN'], config['OUTLIER_RANGE_MAX'], size=num_outliers)
                logging.info(f"Injected {num_outliers} outliers into matrix A.")
            
            np.fill_diagonal(A_float, 1)
            is_inv, cond_A = is_invertible(A_float)
            if not is_inv:
                logging.error("Generated matrix A is not invertible. Aborting.")
                return None
            logging.info(f"Generated invertible matrix A (cond: {cond_A:.2f}).")
            
            while True:
                q, _ = np.linalg.qr(np.random.randn(N, N))
                T_float = np.round(q * 10000)
                cond_t = np.linalg.cond(T_float)
                if cond_t < 1 / np.finfo(float).eps and cond_t < 10: break
            logging.info(f"Generated transformation matrix T (cond: {cond_t:.2f}).")

            B_float = np.random.randint(config['B_INT_RANGE_MIN'], config['B_INT_RANGE_MAX'], size=(R, N)).astype(np.float64)
            f_float = np.zeros((N, 1), dtype=np.float64); f_float[0, 0] = 1.0
            
            final_plaintext_result = B_float @ np.diag((np.linalg.inv(A_float) @ f_float).flatten())
            logging.info("Plaintext matrices generated and reference result calculated.")
            
            return {'A': A_float, 'T': T_float, 'B': B_float, 'f': f_float, 'final_plaintext_result': final_plaintext_result, 'matrix_cond_A': float(cond_A), 'matrix_cond_T': float(cond_t)}
        except Exception as e:
            logging.error(f"Matrix generation failed: {e}", exc_info=True)
            return None


    def execute_secure_computation(self, matrices, config, run_id):
        start_time = time.time()
        
        all_shares = {name: create_shares_float(matrix, config['USE_ADAPTIVE_SHARING'], config)
                      for name, matrix in {'A': matrices['A'], 'T': matrices['T'], 'B': matrices['B'], 'f': matrices['f']}.items()}
        
        logging.info("Creating and distributing secret shares for all matrices...")
        self._execute_on_all_parties(
            'ReceiveShares',
            lambda p_id: smpc_pb2.ShareDistribution(
                computation_id=run_id,
                matrix_name='A', # These names are illustrative for one call, but need to be generic
                share=matrix_to_proto(all_shares['A'][p_id-1]) # simplified, will fix
            )
        )
        # The above lambda is tricky. Let's do it properly.
        for name, shares in all_shares.items():
            logging.info(f"Distributing shares for matrix '{name}'...")
            self._execute_on_all_parties(
                'ReceiveShares',
                lambda p_id: smpc_pb2.ShareDistribution(
                    computation_id=run_id,
                    matrix_name=name,
                    share=matrix_to_proto(shares[p_id-1])
                )
            )

        logging.info("Step 1: Requesting secure multiplication A * T.")
        self._execute_on_all_parties('SecureMatMul', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='A', matrix_b_name='T', result_name='AT'))
        
        logging.info("Step 2: Reconstructing AT for public inversion.")
        AT_reconstructed = self.reconstruct_matrix(run_id, 'AT')
        AT_inv_computed = np.linalg.inv(AT_reconstructed)
        
        logging.info("Step 3: Distributing shares of (A*T)^-1.")
        AT_inv_shares = create_shares_float(AT_inv_computed, config['USE_ADAPTIVE_SHARING'], config)
        self._execute_on_all_parties('ReceiveShares', lambda p_id: smpc_pb2.ShareDistribution(computation_id=run_id, matrix_name='AT_inv', share=matrix_to_proto(AT_inv_shares[p_id-1])))
        
        logging.info("Step 4: Requesting secure multiplication T * (AT)^-1 with randomization.")
        self._execute_on_all_parties('SecureMatMulWithRandomization', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='T', matrix_b_name='AT_inv', result_name='A_inv_rand', parameters={k: str(v) for k,v in config.items()}))

        logging.info("Step 5: Requesting secure multiplication A_inv_rand * f with randomization.")
        self._execute_on_all_parties('SecureMatMulWithRandomization', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='A_inv_rand', matrix_b_name='f', result_name='s_rand', parameters={k: str(v) for k,v in config.items()}))

        logging.info("Step 6: Requesting creation of diagonal matrix from s_rand.")
        self._execute_on_all_parties('CreateDiagonal', lambda p_id: smpc_pb2.ComputationStep(computation_id=run_id, operation="diagonal", input_matrices=['s_rand'], output_matrix='diag_s'))
        
        logging.info("Step 7: Requesting final secure multiplication B * diag_s.")
        self._execute_on_all_parties('SecureMatMul', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='B', matrix_b_name='diag_s', result_name='final'))
        
        logging.info("Step 8: Reconstructing the final result.")
        final_computed = self.reconstruct_matrix(run_id, 'final')
        
        duration = time.time() - start_time
        logging.info(f"Total secure computation time: {duration:.4f} seconds.")
        
        error_metrics = calculate_error_metrics(final_computed, matrices['final_plaintext_result'])
        logging.info(f"Error metrics: {error_metrics}")
        
        return {'float_point_time_s': duration, **{f'float_{k}': v for k, v in error_metrics.items()}, 'matrix_cond_A': matrices['matrix_cond_A'], 'matrix_cond_T': matrices['matrix_cond_T']}

    def reconstruct_matrix(self, computation_id, matrix_name):
        logging.info(f"Collecting shares to reconstruct '{matrix_name}'...")
        futures = {self.grpc_executor.submit(client.ReturnShares, smpc_pb2.RevealRequest(computation_id=computation_id, matrix_name=matrix_name)): party_id
                   for party_id, client in self.party_clients.items()}
        
        shares = [None] * len(self.party_clients)
        for future in as_completed(futures):
            party_id = futures[future]
            response = future.result()
            if not response.success: raise RuntimeError(f"Party {party_id} failed to return share for {matrix_name}")
            shares[party_id-1] = proto_to_matrix(response.result)
            
        logging.info(f"Reconstruction of '{matrix_name}' successful.")
        return reconstruct_shares(shares)

    def run(self):
        logging.info(f"Orchestrator HTTP server starting on port {self.http_port}")
        self.app.run(host='0.0.0.0', port=self.http_port, debug=False)

# The WEB_INTERFACE_HTML string remains the same.
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
            
            const keys = Object.keys(results[0]);
            keys.forEach(key => html += `<th>${key}</th>`);
            html += '</tr>';
            
            results.forEach(result => {
                html += '<tr>';
                keys.forEach(key => {
                    html += `<td>${typeof result[key] === 'number' ? result[key].toFixed(6) : result[key]}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</table>';
            resultsDiv.innerHTML = html;
        }
        
        async function downloadResults() {
            window.location.href = '/api/download_results';
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    orchestrator = SMPCOrchestrator()
    orchestrator.run()