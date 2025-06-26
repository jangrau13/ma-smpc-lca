"""
SECURE MULTI-PARTY COMPUTATION PARTY IMPLEMENTATION (WITH ASYNC FINISH PROTOCOL)
"""

import os
import threading
import logging
import time # <-- ADDED MISSING IMPORT
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
        self.orchestrator_addr = os.getenv('ORCHESTRATOR_GRPC_ADDRESS')
        self.other_parties_addrs = os.getenv('OTHER_PARTIES_GRPC', '').split(',')
        
        self.additive_shares = {}
        self.received_shares = {}
        
        self.orchestrator_client = None
        self.party_clients = {}
        self.init_clients()
        
        self.state_lock = threading.Lock()
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=20)
        
        logging.info(f"Initialized on port {self.grpc_port}")

    def init_clients(self):
        # Connect to Orchestrator
        try:
            channel = grpc.insecure_channel(self.orchestrator_addr)
            self.orchestrator_client = smpc_pb2_grpc.OrchestratorServiceStub(channel)
            logging.info(f"Connected to orchestrator at {self.orchestrator_addr}")
        except Exception as e:
            logging.error(f"Failed to connect to orchestrator: {e}")

        # Connect to other Parties
        # This assumes a specific naming convention in docker-compose for party discovery
        for i in range(1, 4):
            if i == self.party_id: continue
            addr = f"party{i}:{50052 + (i-1)}"
            try:
                channel = grpc.insecure_channel(addr)
                self.party_clients[i] = smpc_pb2_grpc.PartyComputationServiceStub(channel)
                logging.info(f"Connected to Party {i} at {addr}")
            except Exception as e:
                logging.error(f"Failed to connect to party at {addr}: {e}")

    def _execute_and_report_back(self, target_func, step_name, *args):
        """Wrapper to run a function in a background thread and report completion."""
        def wrapper():
            computation_id = args[0].computation_id
            success = True
            message = ""
            try:
                target_func(*args)
            except Exception as e:
                logging.error(f"Error during step '{step_name}': {e}", exc_info=True)
                success = False
                message = str(e)
            
            logging.info(f"Finished step '{step_name}'. Notifying orchestrator...")
            self.orchestrator_client.ComputationFinished(smpc_pb2.ComputationFinishedRequest(
                computation_id=computation_id,
                party_id=self.party_id,
                step_name=step_name,
                success=success,
                message=message
            ))
        
        self.thread_pool.submit(wrapper)

    def ReceiveShares(self, request, context):
        with self.state_lock:
            if request.computation_id not in self.additive_shares:
                self.additive_shares[request.computation_id] = {}
            self.additive_shares[request.computation_id][request.matrix_name] = proto_to_matrix(request.share)
        logging.info(f"Received and stored share for '{request.matrix_name}'.")
        return smpc_pb2.Response(success=True)

    def wait_for_share(self, computation_id, matrix_name):
        key = (computation_id, matrix_name)
        #logging.info(f"Waiting to receive share for '{matrix_name}'.")
        while True:
            with self.state_lock:
                if key in self.received_shares:
                    #logging.info(f"Share for '{matrix_name}' has been received.")
                    return self.received_shares.pop(key)
            time.sleep(0.01) # Avoid busy-waiting

    def SecureMatMul(self, request, context):
        step_name = f"SecureMatMul:{request.result_name}"
        logging.info(f"Trigger received for {step_name}")
        self._execute_and_report_back(self._secure_matmul_logic, step_name, request)
        return smpc_pb2.Response(success=True, message="Computation triggered.")

    def _secure_matmul_logic(self, request):
        comp_id, mat_a, mat_b, res_name = request.computation_id, request.matrix_a_name, request.matrix_b_name, request.result_name
        
        with self.state_lock:
            my_share_A = self.additive_shares[comp_id][mat_a]
            my_share_B = self.additive_shares[comp_id][mat_b]
        
        send_to_party_id = (self.party_id % 3) + 1
        receive_from_party_id = ((self.party_id - 2 + 3) % 3) + 1
        
        logging.info(f"Sending my shares of '{mat_a}' & '{mat_b}' to Party {send_to_party_id}.")
        self.party_clients[send_to_party_id].ReceiveFromParty(smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=f"{mat_a}_{self.party_id}", share=matrix_to_proto(my_share_A)))
        self.party_clients[send_to_party_id].ReceiveFromParty(smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=f"{mat_b}_{self.party_id}", share=matrix_to_proto(my_share_B)))

        logging.info(f"Waiting for shares from Party {receive_from_party_id}.")
        received_share_A = self.wait_for_share(comp_id, f"{mat_a}_{receive_from_party_id}")
        received_share_B = self.wait_for_share(comp_id, f"{mat_b}_{receive_from_party_id}")
        
        logging.info(f"All shares received for '{res_name}'. Computing result.")
        result = my_share_A @ my_share_B + my_share_A @ received_share_B + received_share_A @ my_share_B
        
        with self.state_lock:
            self.additive_shares[comp_id][res_name] = result
        logging.info(f"<- Finished SecureMatMul for '{res_name}'.")
            
    def SecureMatMulWithRandomization(self, request, context):
        step_name = f"SecureMatMulWithRandomization:{request.result_name}"
        logging.info(f"Trigger received for {step_name}")
        self._execute_and_report_back(self._secure_matmul_rand_logic, step_name, request)
        return smpc_pb2.Response(success=True, message="Computation triggered.")

    def _secure_matmul_rand_logic(self, request):
        comp_id, mat_a, mat_b, res_name = request.computation_id, request.matrix_a_name, request.matrix_b_name, request.result_name
        temp_result_name = f"{res_name}_pre_rand"
        
        self._secure_matmul_logic(smpc_pb2.MatMulRequest(computation_id=comp_id, matrix_a_name=mat_a, matrix_b_name=mat_b, result_name=temp_result_name))

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
            self.party_clients[3].ReceiveFromParty(smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=f"{rand_key}_1", share=matrix_to_proto(r0)))
        elif self.party_id == 2:
            r1 = self.generate_randomness(result.shape, config)
            result += r1
            self.party_clients[3].ReceiveFromParty(smpc_pb2.ShareDistribution(computation_id=comp_id, matrix_name=f"{rand_key}_2", share=matrix_to_proto(r1)))
        else: # Party 3
            r0 = self.wait_for_share(comp_id, f"{rand_key}_1")
            r1 = self.wait_for_share(comp_id, f"{rand_key}_2")
            result -= (r0 + r1)

        with self.state_lock:
            self.additive_shares[comp_id][res_name] = result
        logging.info(f"<- Finished SecureMatMulWithRandomization for '{res_name}'.")

    def generate_randomness(self, shape, config):
        noise_key = 'MINIMUM_NOISE_RANGE_VAL'
        if config.get('adaptive'):
            scale = config.get(noise_key, 2.0) * (1 + np.random.uniform(config.get('OBFUSCATION_FACTOR_MIN', 0.1), config.get('OBFUSCATION_FACTOR_MAX', 0.5)))
            return (np.random.rand(*shape) - 0.5) * 2 * scale
        return np.random.uniform(-config.get(noise_key, 2.0), config.get(noise_key, 2.0), size=shape).astype(np.float64)

    def CreateDiagonal(self, request, context):
        step_name = f"CreateDiagonal:{request.output_matrix}"
        logging.info(f"Trigger received for {step_name}")
        self._execute_and_report_back(self._create_diagonal_logic, step_name, request)
        return smpc_pb2.Response(success=True, message="Computation triggered.")

    def _create_diagonal_logic(self, request):
        comp_id, vec_name, diag_name = request.computation_id, request.input_matrices[0], request.output_matrix
        logging.info(f"Creating diagonal matrix '{diag_name}' from '{vec_name}'.")
        with self.state_lock:
            self.additive_shares[comp_id][diag_name] = np.diag(self.additive_shares[comp_id][vec_name].flatten())
        logging.info(f"<- Finished CreateDiagonal for '{diag_name}'.")

    def ReturnShares(self, request, context):
        with self.state_lock:
            try:
                my_share = self.additive_shares[request.computation_id][request.matrix_name]
                return smpc_pb2.Response(success=True, result=matrix_to_proto(my_share))
            except KeyError:
                logging.error(f"Could not find share for '{request.matrix_name}' to return.")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Key '{request.matrix_name}' not found.")
                return smpc_pb2.Response(success=False, message="Key not found")


    def ReceiveFromParty(self, request, context):
        with self.state_lock:
            self.received_shares[(request.computation_id, request.matrix_name)] = proto_to_matrix(request.share)
        #logging.info(f"Received inter-party share for '{request.matrix_name}'.")
        return smpc_pb2.Response(success=True)

def serve():
    party_id = int(os.getenv('PARTY_ID', 1))
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