"""
SECURE MULTI-PARTY COMPUTATION ORCHESTRATOR (WITH ASYNC FINISH PROTOCOL)
"""

import os
import time
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template, send_file, Response
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

# --- Utility Functions ---
def matrix_to_proto(matrix):
    if matrix is None: return smpc_pb2.Matrix(data=[], rows=0, cols=0)
    return smpc_pb2.Matrix(data=matrix.flatten().astype(np.int64).tolist(), rows=matrix.shape[0], cols=matrix.shape[1])

def proto_to_matrix(proto_matrix):
    if not proto_matrix.rows or not proto_matrix.cols: return None
    return np.array(proto_matrix.data, dtype=np.int64).reshape(proto_matrix.rows, proto_matrix.cols)

def is_invertible(matrix, tolerance=1e-10):
    cond = np.linalg.cond(matrix.astype(np.float64))
    return cond < 1 / np.finfo(np.float64).eps, cond

def create_shares(matrix):
    """Creates three uint64 additive shares using Z_2^64 modular arithmetic. Returns three uint64 matrices."""
    ring_size = 2**64
    share0 = np.random.randint(0, ring_size, size=matrix.shape, dtype=np.uint64)
    share1 = np.random.randint(0, ring_size, size=matrix.shape, dtype=np.uint64)
    share2 = (matrix.astype(np.uint64) - share0 - share1)
    return share0.astype(np.int64), share1.astype(np.int64), share2.astype(np.int64)

def reconstruct_shares(shares):
    """Reconstructs using proper modular arithmetic handling in Z_2^64 with uint64."""
    result = (shares[0].astype(np.uint64) + shares[1].astype(np.uint64) + shares[2].astype(np.uint64))
    return result

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

def get_matrix_info(matrix):
    """
    Calculates various informational metrics for a given matrix with flexible bucketing.
    """
    if matrix is None:
        return {
            'condition_number': 'N/A',
            'max_element': 'N/A',
            'percentage_of_ones': 0,
            'percentage_of_zeros': 0,
            'range_in_buckets': {}
        }

    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    # a) Condition Number
    condition_number = 'N/A'
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        if np.linalg.det(matrix) != 0:
            condition_number = np.linalg.cond(matrix)
        else:
            condition_number = 'N/A'
    else:
        condition_number = 'N/A'

    # b) Max Absolute Element
    max_element = np.max(np.abs(matrix)) if matrix.size > 0 else 'N/A'

    # c) Percentage of 1s
    percentage_of_ones = (np.count_nonzero(matrix == 1) / matrix.size) * 100 if matrix.size > 0 else 0

    # d) Percentage of 0s
    percentage_of_zeros = (np.count_nonzero(matrix == 0) / matrix.size) * 100 if matrix.size > 0 else 0

    # e) Flexible Range in Buckets
    buckets = {}
    if matrix.size > 0:
        min_val, max_val = np.min(matrix), np.max(matrix)
        if min_val == max_val:
            buckets[f"[{min_val:.2e}]"] = matrix.size
        else:
            num_buckets = 10
            # np.histogram handles the bins as [e0, e1), [e1, e2), ..., [eN-1, eN] (inclusive on last)
            hist, bin_edges = np.histogram(matrix, bins=num_buckets)
            for i in range(num_buckets):
                if i < num_buckets - 1:
                    label = f"[{bin_edges[i]:.2e}, {bin_edges[i+1]:.2e})"
                else:
                    # Last bucket is inclusive of both ends
                    label = f"[{bin_edges[i]:.2e}, {bin_edges[i+1]:.2e}]"
                buckets[label] = int(hist[i])

    return {
        'condition_number': condition_number,
        'max_element': max_element,
        'percentage_of_ones': percentage_of_ones,
        'percentage_of_zeros': percentage_of_zeros,
        'range_in_buckets': buckets
    }

# --- Orchestrator gRPC Service for Party Callbacks ---
class OrchestratorService(smpc_pb2_grpc.OrchestratorServiceServicer):
    def __init__(self, orchestrator_instance):
        self.orchestrator = orchestrator_instance

    def ComputationFinished(self, request, context):
        self.orchestrator.handle_computation_finished(request)
        return smpc_pb2.Response(success=True)

class SMPCOrchestrator:
    def __init__(self):
        self.app = Flask(__name__, template_folder='templates')
        CORS(self.app)
        
        self.http_port = int(os.getenv('HTTP_PORT', 8080))
        self.grpc_port = int(os.getenv('GRPC_PORT', 50051))
        self.party_addresses = os.getenv('PARTY_GRPC_ADDRESSES', '').split(',')
        
        self.latest_matrices = {}
        
        self.results_df = pd.DataFrame()
        self.csv_file = '/app/results/fixed_point_results.csv'
        if os.path.exists(self.csv_file): self.results_df = pd.read_csv(self.csv_file)
        
        self.setup_routes()
        
        self.party_clients = {}
        self.init_party_clients()
        self.grpc_executor = ThreadPoolExecutor(max_workers=len(self.party_clients) * 5)
        
        self.computation_events = {}
        self.running_computations = set() 
        self.computation_lock = threading.Lock()

    def init_party_clients(self):
        logging.info("Initializing gRPC clients to parties...")
        options = [('grpc.max_send_message_length', 100 * 1024 * 1024), ('grpc.max_receive_message_length', 100 * 1024 * 1024)] # 100 MB limit for large matrices
        for i, address in enumerate(self.party_addresses):
            if address.strip():
                channel = grpc.insecure_channel(address.strip(), options=options)
                self.party_clients[i+1] = smpc_pb2_grpc.PartyComputationServiceStub(channel)
                logging.info(f"Connected to Party {i+1} at {address.strip()}")
    
    def setup_routes(self):
        @self.app.route('/')
        def index(): 
            return render_template('index.html')
        
        @self.app.route('/api/start_computation', methods=['POST'])
        def start_computation():
            if request.is_json:
                config = request.get_json()
            else:
                form_data = request.form
                config = {
                    'N': int(form_data.get('N', 100)),
                    'R': int(form_data.get('R', 100)),
                    'A_DENSITY': float(form_data.get('A_DENSITY', 0.1)),
                    'INTRODUCE_OUTLIERS': form_data.get('INTRODUCE_OUTLIERS', 'false').lower() == 'true',
                    'OUTLIER_PROBABILITY': float(form_data.get('OUTLIER_PROBABILITY', 0.11)),
                    'OUTLIER_RANGE_MIN': float(form_data.get('OUTLIER_RANGE_MIN', -5000)),
                    'OUTLIER_RANGE_MAX': float(form_data.get('OUTLIER_RANGE_MAX', -2000)),
                    'B_INT_RANGE_MIN': int(form_data.get('B_INT_RANGE_MIN', 100)),
                    'B_INT_RANGE_MAX': int(form_data.get('B_INT_RANGE_MAX', 5000)),
                    'USE_ADAPTIVE_SHARING': form_data.get('USE_ADAPTIVE_SHARING', 'true').lower() == 'true',
                    'MINIMUM_NOISE_RANGE_VAL': float(form_data.get('MINIMUM_NOISE_RANGE_VAL', 2.0)),
                    'OBFUSCATION_FACTOR_MIN': float(form_data.get('OBFUSCATION_FACTOR_MIN', 0.1)),
                    'OBFUSCATION_FACTOR_MAX': float(form_data.get('OBFUSCATION_FACTOR_MAX', 0.5)),
                    'PRECISION': int(form_data.get('PRECISION', 20)),  # Fixed-point precision bits
                    'T_SCALING_FACTOR': int(form_data.get('T_SCALING_FACTOR', 10000)),
                    'MAX_COND_T': float(form_data.get('MAX_COND_T', 10.0)),
                }

            run_id = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()
            
            with self.computation_lock:
                if run_id in self.running_computations:
                    msg = f"Computation with same configuration (ID: {run_id[:8]}) is already running."
                    logging.warning(msg)
                    return jsonify({'success': False, 'message': msg}), 409

                if not self.results_df.empty and run_id in self.results_df.get('run_id', pd.Series()).values:
                    msg = f"Computation with same configuration (ID: {run_id[:8]}) already exists in results."
                    logging.warning(msg)
                    return jsonify({'success': False, 'message': msg}), 409

                self.running_computations.add(run_id)

            try:
                computation_thread = threading.Thread(target=self.run_computation, args=(config, run_id))
                computation_thread.start()
                return jsonify({'success': True, 'message': 'Computation started in background.'})
            except Exception as e:
                with self.computation_lock: self.running_computations.remove(run_id)
                logging.error(f"HTTP Error on /api/start_computation: {e}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/get_results', methods=['GET'])
        def get_results():
            results = self.results_df.to_dict('records')
            table_html = render_template('results_table.html', results=results)
            return Response(table_html)

        @self.app.route('/api/latest_comparison')
        def latest_comparison():
            return render_template('latest_comparison.html', matrices=self.latest_matrices)

        @self.app.route('/api/download_results', methods=['GET'])
        def download_results():
            if os.path.exists(self.csv_file): return send_file(self.csv_file, as_attachment=True)
            return jsonify({'success': False, 'error': 'No results file'}), 404

    def handle_computation_finished(self, request):
        with self.computation_lock:
            event_key = (request.computation_id, request.step_name)
            if event_key in self.computation_events:
                logging.info(f"Received finish signal for step '{request.step_name}' from Party {request.party_id}. Success: {request.success}")
                if not request.success:
                    self.computation_events[event_key]['failed'] = True
                    self.computation_events[event_key]['error_message'] = f"Party {request.party_id}: {request.message}"
                self.computation_events[event_key]['events'][request.party_id - 1].set()
            else:
                logging.warning(f"Received unexpected finish signal for {event_key}")

    def _trigger_and_wait(self, run_id, step_name, rpc_method_name, request_builder):
        event_key = (run_id, step_name)
        with self.computation_lock:
            self.computation_events[event_key] = {
                'events': [threading.Event() for _ in self.party_clients],
                'failed': False,
                'error_message': ''
            }
        
        logging.info(f"Triggering step '{step_name}' on all parties...")
        for party_id, client in self.party_clients.items():
            try:
                getattr(client, rpc_method_name)(request_builder(party_id))
            except Exception as exc:
                logging.error(f"Failed to trigger {rpc_method_name} on Party {party_id}: {exc}", exc_info=True)
                raise
        
        logging.info(f"Waiting for all parties to complete step '{step_name}'...")
        for event in self.computation_events[event_key]['events']:
            event.wait(timeout=60.0)
        
        with self.computation_lock:
            step_result = self.computation_events.pop(event_key)

        if step_result['failed']:
            raise RuntimeError(f"A party failed during step '{step_name}'. Error: {step_result['error_message']}")

        if not all(e.is_set() for e in step_result['events']):
            raise TimeoutError(f"Timeout waiting for parties to complete step '{step_name}'")

        logging.info(f"All parties have completed step '{step_name}' successfully.")

    def run_computation(self, config, run_id):
        logging.info(f"--- Starting new computation (ID: {run_id[:8]}) ---")
        logging.info(f"Configuration: {config}")
        
        np.random.seed(int(run_id, 16) % (2**32))
        
        try:
            matrices = self.generate_matrices(config)
            if matrices is None: return

            result_metrics = self.execute_secure_computation(matrices, config, run_id)

            final_computed = result_metrics['smpc_result']
            
            result_metrics.update(config)
            result_metrics['run_id'] = run_id
            
            with self.computation_lock:
                self.results_df = pd.concat([self.results_df, pd.DataFrame([result_metrics])], ignore_index=True)
                self.results_df.to_csv(self.csv_file, index=False)
                self.results_df.tail(1).to_csv('/app/results/latest_result.csv', index=False)

            self.latest_matrices = {
                'A': matrices['A'],
                'T': matrices['T'],
                'B': matrices['B'],
                'f': matrices['f'],
                'plaintext_result': matrices['final_plaintext_result'],
                'smpc_result': final_computed
            }
            
            matrix_info = {}
            for name, matrix_data in self.latest_matrices.items():
                matrix_info[name] = get_matrix_info(matrix_data)
            self.latest_matrices['matrix_info'] = matrix_info

            logging.info(f"--- Computation {run_id[:8]} completed successfully ---")
        except Exception as e:
            logging.error(f"Computation {run_id[:8]} failed: {e}", exc_info=True)
        finally:
             with self.computation_lock:
                 if run_id in self.running_computations:
                       self.running_computations.remove(run_id)

    def generate_matrices(self, config):
        logging.info("Step 0: Generating plaintext matrices...")
        try:
            N, R = int(config['N']), int(config['R'])
            A_float = sparse_random(N, N, density=float(config['A_DENSITY']), format='dok', data_rvs=lambda s: -np.random.rand(s)).toarray()
            
            if config.get('INTRODUCE_OUTLIERS'):
                mask = (np.random.rand(N, N) < float(config['OUTLIER_PROBABILITY'])) & (np.eye(N) == 0)
                num_outliers = np.sum(mask)
                A_float[mask] = np.random.uniform(float(config['OUTLIER_RANGE_MIN']), float(config['OUTLIER_RANGE_MAX']), size=num_outliers)
                logging.info(f"Injected {num_outliers} outliers into matrix A.")
            
            np.fill_diagonal(A_float, 1)
            is_inv, cond_A = is_invertible(A_float)
            if not is_inv:
                logging.error("Generated matrix A is not invertible. Aborting.")
                return None
            logging.info(f"Generated invertible matrix A (cond: {cond_A:.2f}).")
            
            while True:
                z = np.random.randn(N, N)
                q, _ = np.linalg.qr(z)
                T_int = np.round(q * config.get('T_SCALING_FACTOR', 100)).astype(np.int64)
                cond_t = np.linalg.cond(T_int.astype(np.float64))
                
                if cond_t < 1 / np.finfo(np.float64).eps and cond_t < config.get('MAX_COND_T', 10.0):
                    break


            B_int = np.random.randint(int(config['B_INT_RANGE_MIN']), int(config['B_INT_RANGE_MAX']), size=(R, N), dtype=np.int64)
            f_int = np.zeros((N, 1), dtype=np.int64); f_int[0, 0] = 1
            
            B_float = B_int.astype(np.float64)
            f_float = f_int.astype(np.float64)
            # Compute plaintext results on CPU
            A_inv_plaintext = np.linalg.inv(A_float)
            s_plaintext_vec = A_inv_plaintext @ f_float
            diag_s_plaintext = np.diag(s_plaintext_vec.flatten())
            final_plaintext_result = B_float @ diag_s_plaintext

            logging.info("Plaintext matrices generated and reference result calculated.")
            
            return {'A': A_float, 'T': T_int, 'B': B_int, 'f': f_int, 'final_plaintext_result': final_plaintext_result, 'matrix_cond_A': float(cond_A), 'matrix_cond_T': float(cond_t)}
        except Exception as e:
            logging.error(f"Matrix generation failed: {e}", exc_info=True)
            return None

    def execute_secure_computation(self, matrices, config, run_id):
        start_time = time.time()
        scale_factor = 2**config['PRECISION']
        
        # Convert A to fixed-point representation
        A_fix = (matrices['A'] * scale_factor).astype(np.int64)
        T_int = matrices['T']  
        B_int = matrices['B']  
        f_int = matrices['f']  
        

        all_shares = {
            'A': create_shares(A_fix),
            'T': create_shares(T_int),
            'B': create_shares(B_int),
            'f': create_shares(f_int)
        }
        logging.info("Distributing all initial secret shares...")
        for name, shares in all_shares.items():
            for party_id, client in self.party_clients.items():
                client.ReceiveShares(smpc_pb2.ShareDistribution(
                    computation_id=run_id, matrix_name=name, share=matrix_to_proto(shares[party_id-1])))

        logging.info("Step 1: Triggering SecureMatMul A * T.")
        self._trigger_and_wait(run_id, 'SecureMatMul:AT', 'SecureMatMul', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='A', matrix_b_name='T', result_name='AT'))
        
        logging.info("Step 2: Reconstructing AT for public inversion.")
        AT_reconstructed_uint = self.reconstruct_matrix(run_id, 'AT')
        # Convert reconstructed AT to float for inversion
        AT_reconstructed = AT_reconstructed_uint.astype(np.int64)
        # Remove the scale factor from A before inversion
        AT_recon_float = AT_reconstructed.astype(np.float64) / scale_factor 
        AT_inv_computed = np.linalg.inv(AT_recon_float)
        # add scaling factor back to the inverted matrix
        AT_inv_computed = (AT_inv_computed * scale_factor).astype(np.int64)
        
        logging.info("Step 3: Distributing shares of (A*T)^-1.")
        AT_inv_shares = create_shares(AT_inv_computed)
        for party_id, client in self.party_clients.items():
            client.ReceiveShares(smpc_pb2.ShareDistribution(computation_id=run_id, matrix_name='AT_inv', share=matrix_to_proto(AT_inv_shares[party_id-1])))
        
        logging.info("Step 4: Triggering SecureMatMulWithRandomization T * (AT)^-1.")
        self._trigger_and_wait(run_id, 'SecureMatMulWithRandomization:A_inv_rand', 'SecureMatMulWithRandomization', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='T', matrix_b_name='AT_inv', result_name='A_inv_rand', parameters={k: str(v) for k,v in config.items()}))

        logging.info("Step 5: Triggering SecureMatMulWithRandomization A_inv_rand * f.")
        self._trigger_and_wait(run_id, 'SecureMatMulWithRandomization:s_rand', 'SecureMatMulWithRandomization', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='A_inv_rand', matrix_b_name='f', result_name='s_rand', parameters={k: str(v) for k,v in config.items()}))
        
        logging.info("Step 6: Triggering creation of diagonal matrix from s_rand.")
        self._trigger_and_wait(run_id, 'CreateDiagonal:diag_s', 'CreateDiagonal', lambda p_id: smpc_pb2.ComputationStep(computation_id=run_id, operation="diagonal", input_matrices=['s_rand'], output_matrix='diag_s'))
        
        logging.info("Step 7: Triggering final SecureMatMul B * diag_s.")
        self._trigger_and_wait(run_id, 'SecureMatMul:final', 'SecureMatMul', lambda p_id: smpc_pb2.MatMulRequest(computation_id=run_id, matrix_a_name='B', matrix_b_name='diag_s', result_name='final'))
        
        logging.info("Step 8: Reconstructing the final result.")
        final_reconstructed_uint = self.reconstruct_matrix(run_id, 'final')
        # .astype(np.int64).astype(np.float64)
        final_computed = final_reconstructed_uint.astype(np.int64).astype(np.float64) / scale_factor
        
        duration = time.time() - start_time
        
        logging.info(f"Total secure computation time: {duration:.4f} seconds.")
        
        error_metrics = calculate_error_metrics(final_computed, matrices['final_plaintext_result'])
        logging.info(f"Error metrics: {error_metrics}")
        
        return {
            'float_point_time_s': duration,
            'smpc_result': final_computed, 
            **{f'float_{k}': v for k, v in error_metrics.items()},
            'matrix_cond_A': matrices['matrix_cond_A'],
            'matrix_cond_T': matrices['matrix_cond_T']
        }

    def reconstruct_matrix(self, computation_id, matrix_name):
        logging.info(f"Collecting shares to reconstruct '{matrix_name}'...")
        futures = {self.grpc_executor.submit(client.ReturnShares, smpc_pb2.RevealRequest(computation_id=computation_id, matrix_name=matrix_name)): party_id
                   for party_id, client in self.party_clients.items()}
        
        shares = [None] * len(self.party_clients)
        for future in as_completed(futures):
            party_id = futures[future]
            response = future.result()
            if not response.success: raise RuntimeError(f"Party {party_id} failed to return share for {matrix_name}")
            share_matrix = proto_to_matrix(response.result) 
            shares[party_id-1] = share_matrix
            
        logging.info(f"Reconstruction of '{matrix_name}' successful.")
        return reconstruct_shares(shares)

    def run(self):
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB limit for large matrices
            ('grpc.max_receive_message_length', 100 * 1024 * 1024) # 100 MB limit for large matrices
        ]
        grpc_server = grpc.server(ThreadPoolExecutor(max_workers=10), options=options)
        smpc_pb2_grpc.add_OrchestratorServiceServicer_to_server(OrchestratorService(self), grpc_server)
        grpc_server.add_insecure_port(f'0.0.0.0:{self.grpc_port}')
        grpc_server.start()
        logging.info(f"Orchestrator gRPC server started on port {self.grpc_port}")
        
        logging.info(f"Orchestrator HTTP server starting on port {self.http_port}")
        self.app.run(host='0.0.0.0', port=self.http_port, debug=False)

if __name__ == '__main__':
    orchestrator = SMPCOrchestrator()
    orchestrator.run()