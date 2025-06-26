"""
SECURE MULTI-PARTY COMPUTATION PARTY IMPLEMENTATION (CORRECTED LOGIC)
"""

import os
import threading
import logging
from concurrent import futures
import grpc
import numpy as np

# Import generated protobuf files
import smpc_pb2
import smpc_pb2_grpc

def matrix_to_proto(matrix):
    if matrix is None: return smpc_pb2.Matrix(data=[], rows=0, cols=0)
    return smpc_pb2.Matrix(data=matrix.flatten().astype(np.float64).tolist(), rows=matrix.shape[0], cols=matrix.shape[1])

def proto_to_matrix(proto_matrix):
    if not proto_matrix.rows or not proto_matrix.cols: return None
    return np.array(proto_matrix.data, dtype=np.float64).reshape(proto_matrix.rows, proto_matrix.cols)

class SMPCParty(smpc_pb2_grpc.PartyComputationServiceServicer):
    def __init__(self, party_id):
        self.party_id = party_id
        self.log_prefix = f"[PARTY {self.party_id}]"
        logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - {self.log_prefix} - %(message)s')

        self.grpc_port = int(os.getenv('GRPC_PORT', 50052 + (party_id - 1)))
        self.other_parties_addrs = os.getenv('OTHER_PARTIES_GRPC', '').split(',')
        
        self.additive_shares = {}
        self.received_shares = {}
        
        self.party_clients = {}
        self.init_party_clients()
        
        self.state_lock = threading.Lock()
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=20)
        
        logging.info(f"Initialized on port {self.grpc_port}")

    def init_party_clients(self):
        for addr in self.other_parties_addrs:
            if not addr.strip(): continue
            try:
                channel = grpc.insecure_channel(addr.strip())
                client = smpc_pb2_grpc.PartyComputationServiceStub(channel)
                if 'party1' in addr: self.party_clients[1] = client
                elif 'party2' in addr: self.party_clients[2] = client
                elif 'party3' in addr: self.party_clients[3] = client
                logging.info(f"Connected to party at {addr.strip()}")
            except Exception as e:
                logging.error(f"Failed to connect to party at {addr}: {e}")

    def ReceiveShares(self, request, context):
        comp_id, mat_name = request.computation_id[:8], request.matrix_name
        logging.info(f"Received additive share for '{mat_name}' from orchestrator.")
        with self.state_lock:
            if request.computation_id not in self.additive_shares:
                self.additive_shares[request.computation_id] = {}
            self.additive_shares[request.computation_id][request.matrix_name] = proto_to_matrix(request.share)
        return smpc_pb2.Response(success=True)

    def wait_for_share(self, computation_id, matrix_name):
        """Blocks until a specific share is received from another party."""
        key = (computation_id, matrix_name)
        logging.info(f"Waiting to receive share for '{matrix_name}'.")
        while True:
            with self.state_lock:
                if key in self.received_shares:
                    logging.info(f"Share for '{matrix_name}' has been received.")
                    return self.received_shares.pop(key)
            threading.Event().wait(0.01)

    def SecureMatMul(self, request, context):
        comp_id, mat_a, mat_b, res_name = request.computation_id, request.matrix_a_name, request.matrix_b_name, request.result_name
        logging.info(f"-> Starting SecureMatMul: {mat_a} * {mat_b} = {res_name}")
        try:
            # --- NON-BLOCKING SEND PHASE ---
            with self.state_lock:
                my_share_A = self.additive_shares[comp_id][mat_a]
                my_share_B = self.additive_shares[comp_id][mat_b]
            
            send_to_party_id = (self.party_id % 3) + 1
            logging.info(f"Sending shares for '{mat_a}' and '{mat_b}' to Party {send_to_party_id}.")
            self.thread_pool.submit(self.party_clients[send_to_party_id].ReceiveFromParty, smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=mat_a, share=matrix_to_proto(my_share_A)))
            self.thread_pool.submit(self.party_clients[send_to_party_id].ReceiveFromParty, smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=mat_b, share=matrix_to_proto(my_share_B)))

            # --- BLOCKING WAIT PHASE ---
            received_share_A = self.wait_for_share(comp_id, mat_a)
            received_share_B = self.wait_for_share(comp_id, mat_b)
            
            # --- COMPUTE PHASE ---
            logging.info(f"All shares received for '{res_name}'. Computing result.")
            a_replicated = [my_share_A, received_share_A]
            b_replicated = [my_share_B, received_share_B]
            result = a_replicated[0] @ b_replicated[0] + a_replicated[0] @ b_replicated[1] + a_replicated[1] @ b_replicated[0]
            
            with self.state_lock:
                self.additive_shares[comp_id][res_name] = result
            logging.info(f"<- Finished SecureMatMul for '{res_name}'.")
            return smpc_pb2.Response(success=True)
        except Exception as e:
            logging.error(f"Error in SecureMatMul for '{res_name}': {e}", exc_info=True)
            return smpc_pb2.Response(success=False, message=str(e))
            
    def SecureMatMulWithRandomization(self, request, context):
        comp_id, mat_a, mat_b, res_name = request.computation_id, request.matrix_a_name, request.matrix_b_name, request.result_name
        logging.info(f"-> Starting SecureMatMulWithRandomization: {mat_a} * {mat_b} = {res_name}")
        try:
            # This re-uses the now-correct SecureMatMul logic
            temp_result_name = f"{res_name}_pre_rand"
            self.SecureMatMul(smpc_pb2.MatMulRequest(computation_id=comp_id, matrix_a_name=mat_a, matrix_b_name=mat_b, result_name=temp_result_name), context)

            with self.state_lock:
                result = self.additive_shares[comp_id][temp_result_name]

            logging.info(f"Randomizing share for '{res_name}'.")
            config = {k: v for k, v in request.parameters.items()}
            config['adaptive'] = config.get('USE_ADAPTIVE_SHARING', 'True').lower() == 'true'
            for k in ['MINIMUM_NOISE_RANGE_VAL', 'OBFUSCATION_FACTOR_MIN', 'OBFUSCATION_FACTOR_MAX']:
                 if k in config: config[k] = float(config[k])

            rand_key = f"{res_name}_rand"
            if self.party_id == 1:
                r0 = self.generate_randomness(result.shape, config)
                result += r0
                logging.info(f"I am P1, sending r0 for '{res_name}' to P3.")
                self.party_clients[3].ReceiveFromParty(smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=rand_key, share=matrix_to_proto(r0)))
            elif self.party_id == 2:
                r1 = self.generate_randomness(result.shape, config)
                result += r1
                logging.info(f"I am P2, sending r1 for '{res_name}' to P3.")
                self.party_clients[3].ReceiveFromParty(smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=rand_key, share=matrix_to_proto(r1)))
            else: # Party 3
                logging.info(f"I am P3, waiting for r0 and r1 for '{res_name}'.")
                r0 = self.wait_for_share(comp_id, f"{rand_key}_p1")
                r1 = self.wait_for_share(comp_id, f"{rand_key}_p2")
                logging.info(f"I am P3, received randomness for '{res_name}', calculating result.")
                result -= (r0 + r1)

            with self.state_lock:
                self.additive_shares[comp_id][res_name] = result

            logging.info(f"<- Finished SecureMatMulWithRandomization for '{res_name}'.")
            return smpc_pb2.Response(success=True)
        except Exception as e:
            logging.error(f"Error in SecureMatMulWithRandomization for '{res_name}': {e}", exc_info=True)
            return smpc_pb2.Response(success=False, message=str(e))

    def generate_randomness(self, shape, config):
        noise_key = 'MINIMUM_NOISE_RANGE_VAL'
        if config['adaptive']:
            scale = config[noise_key] * (1 + np.random.uniform(config['OBFUSCATION_FACTOR_MIN'], config['OBFUSCATION_FACTOR_MAX']))
            return (np.random.rand(*shape) - 0.5) * 2 * scale
        return np.random.uniform(-config[noise_key], config[noise_key], size=shape)

    def CreateDiagonal(self, request, context):
        comp_id, vec_name, diag_name = request.computation_id, request.input_matrices[0], request.output_matrix
        logging.info(f"Creating diagonal matrix '{diag_name}' from '{vec_name}'.")
        with self.state_lock:
            self.additive_shares[comp_id][diag_name] = np.diag(self.additive_shares[comp_id][vec_name].flatten())
        return smpc_pb2.Response(success=True)

    def ReturnShares(self, request, context):
        logging.info(f"Returning share for '{request.matrix_name}' to orchestrator.")
        with self.state_lock:
            my_share = self.additive_shares[request.computation_id][request.matrix_name]
        return smpc_pb2.Response(success=True, result=matrix_to_proto(my_share))

    def ReceiveFromParty(self, request, context):
        comp_id, mat_name = request.computation_id, request.matrix_name
        logging.info(f"Received message for '{mat_name}' from another party.")
        key = (comp_id, mat_name)
        
        # Special handling for randomness shares to make them unique for Party 3
        if "_rand" in mat_name:
            # context.peer() is not reliable for getting party ID, let's use share_index if available
            # But since we're not setting it, let's just make the key unique for P3
            if self.party_id == 3:
                # This is fragile, assumes P1 calls first, then P2. A better way would be to include sender_id.
                # For now, let's assume order.
                p_key = (comp_id, f"{mat_name}_p1")
                with self.state_lock:
                    if p_key not in self.received_shares:
                        key = p_key
                    else:
                        key = (comp_id, f"{mat_name}_p2")
        
        with self.state_lock:
            self.received_shares[key] = proto_to_matrix(request.share)
        return smpc_pb2.Response(success=True)

def serve():
    party_id = int(os.getenv('PARTY_ID', 1))
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - [PARTY {party_id}] - %(message)s')
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    party_service = SMPCParty(party_id)
    smpc_pb2_grpc.add_PartyComputationServiceServicer_to_server(party_service, server)
    
    listen_addr = f'0.0.0.0:{party_service.grpc_port}'
    server.add_insecure_port(listen_addr)
    server.start()
    logging.info(f"gRPC server started, listening on {listen_addr}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        server.stop(0)

if __name__ == '__main__':
    serve()