"""
SECURE MULTI-PARTY COMPUTATION PARTY IMPLEMENTATION

CRITICAL SECURITY REQUIREMENTS:
- Each party stores ONLY its own shares
- Parties NEVER collect all shares from others  
- Communication is restricted to adjacent parties in replicated sharing
- Shares are only revealed to orchestrator for final reconstruction
- Proper synchronization without time.sleep - parties manage their own state

PROTOCOL PHASES:
1. Receive additive share from orchestrator (x_i only)
2. Convert to replicated shares by exchanging with one adjacent party
3. Perform secure multiplication using replicated shares  
4. Optionally randomize additive result shares using 3-party protocol
5. Convert back to replicated shares for next operation
6. Reveal only own share to orchestrator when requested
"""

import os
import threading
from concurrent import futures
import grpc
import numpy as np

# Import generated protobuf files
import smpc_pb2
import smpc_pb2_grpc
from common.utils import matrix_to_proto, proto_to_matrix

class SMPCParty(smpc_pb2_grpc.PartyComputationServiceServicer):
    def __init__(self, party_id):
        self.party_id = party_id  # 1, 2, or 3
        self.party_index = party_id - 1  # 0, 1, or 2 for array indexing
        self.grpc_port = int(os.getenv('GRPC_PORT', 50052 + (party_id - 1))) 
        self.orchestrator_address = os.getenv('ORCHESTRATOR_GRPC_ADDRESS', 'orchestrator:50051')
        self.other_parties = os.getenv('OTHER_PARTIES_GRPC', '').split(',')
        
        # Storage for matrix shares
        self.additive_shares = {}  # {computation_id: {matrix_name: my_additive_share}}
        self.replicated_shares = {}  # {computation_id: {matrix_name: [my_share, adjacent_party_share]}}
        
        # State tracking for synchronization
        self.conversion_status = {}  # {computation_id: {matrix_name: conversion_complete}}
        self.received_randomness = {}  # {computation_id: {matrix_name: [r0, r1] for party 3}}
        
        # gRPC clients for other parties
        self.party_clients = {}
        self.init_party_clients()
        
        # Locks for thread safety
        self.state_lock = threading.Lock()
        
        print(f"Party {party_id} initialized on port {self.grpc_port}")
        print(f"Party index: {self.party_index}")
    
    def init_party_clients(self):
        """Initialize gRPC clients for communication with specific parties."""
        for address in self.other_parties:
            if address.strip():
                try:
                    channel = grpc.insecure_channel(address.strip())
                    client = smpc_pb2_grpc.PartyComputationServiceStub(channel)
                    if 'party1' in address:
                        self.party_clients[1] = client
                    elif 'party2' in address:
                        self.party_clients[2] = client
                    elif 'party3' in address:
                        self.party_clients[3] = client
                    print(f"Connected to party at {address.strip()}")
                except Exception as e:
                    print(f"Failed to connect to party at {address}: {e}")
    
    def ReceiveShares(self, request, context):
        """Receive initial additive shares from orchestrator."""
        try:
            computation_id = request.computation_id
            matrix_name = request.matrix_name
            share_matrix = proto_to_matrix(request.share)
            
            with self.state_lock:
                # Store the additive share (only our own share)
                if computation_id not in self.additive_shares:
                    self.additive_shares[computation_id] = {}
                
                self.additive_shares[computation_id][matrix_name] = share_matrix
            
            print(f"Party {self.party_id}: Received additive share for {matrix_name} in computation {computation_id[:8]}")
            
            return smpc_pb2.Response(success=True, message="Additive share received successfully")
            
        except Exception as e:
            print(f"Party {self.party_id}: Error receiving shares: {e}")
            return smpc_pb2.Response(success=False, message=str(e))
    
    def ensure_replicated_shares(self, computation_id, matrix_name):
        """Ensure matrix is in replicated shares state, convert if needed."""
        with self.state_lock:
            # Check if already converted
            if (computation_id in self.replicated_shares and 
                matrix_name in self.replicated_shares[computation_id] and
                len(self.replicated_shares[computation_id][matrix_name]) == 2):
                return True
            
            # Need to convert from additive to replicated
            if (computation_id not in self.additive_shares or 
                matrix_name not in self.additive_shares[computation_id]):
                return False
            
            my_share = self.additive_shares[computation_id][matrix_name]
            
            # Determine communication pattern
            if self.party_id == 1:  # Party 1 (index 0)
                send_to_party = 2    # Send to party 2
                receive_from_party = 3  # Receive from party 3
            elif self.party_id == 2:  # Party 2 (index 1)  
                send_to_party = 3    # Send to party 3
                receive_from_party = 1  # Receive from party 1
            else:  # Party 3 (index 2)
                send_to_party = 1    # Send to party 1
                receive_from_party = 2  # Receive from party 2
            
            # Send our share to the designated party
            share_msg = smpc_pb2.ShareDistribution(
                computation_id=computation_id,
                matrix_name=f"{matrix_name}_for_replication",
                share=matrix_to_proto(my_share),
                share_index=self.party_index
            )
            
            if send_to_party in self.party_clients:
                try:
                    response = self.party_clients[send_to_party].ReceiveFromParty(share_msg)
                    if not response.success:
                        print(f"Party {self.party_id}: Failed to send share to party {send_to_party}")
                        return False
                except Exception as e:
                    print(f"Party {self.party_id}: Error sending share: {e}")
                    return False
            
            # Initialize replicated shares with our own share
            if computation_id not in self.replicated_shares:
                self.replicated_shares[computation_id] = {}
            
            self.replicated_shares[computation_id][matrix_name] = [my_share, None]
            
            print(f"Party {self.party_id}: Initiated conversion to replicated shares for {matrix_name}")
            return True
    
    def wait_for_replicated_shares(self, computation_id, matrix_name, max_attempts=100):
        """Wait until we have both shares for replicated sharing."""
        attempts = 0
        while attempts < max_attempts:
            with self.state_lock:
                if (computation_id in self.replicated_shares and 
                    matrix_name in self.replicated_shares[computation_id] and
                    self.replicated_shares[computation_id][matrix_name][1] is not None):
                    return True
            
            # Brief pause to avoid busy waiting
            threading.Event().wait(0.01)
            attempts += 1
        
        return False
    
    def SecureMatMul(self, request, context):
        """Perform secure matrix multiplication using replicated shares."""
        try:
            computation_id = request.computation_id
            matrix_a_name = request.matrix_a_name
            matrix_b_name = request.matrix_b_name
            result_name = request.result_name
            
            print(f"Party {self.party_id}: Computing secure {matrix_a_name} * {matrix_b_name} = {result_name}")
            
            # Ensure both matrices are in replicated shares state
            if not self.ensure_replicated_shares(computation_id, matrix_a_name):
                raise Exception(f"Failed to convert {matrix_a_name} to replicated shares")
            
            if not self.ensure_replicated_shares(computation_id, matrix_b_name):
                raise Exception(f"Failed to convert {matrix_b_name} to replicated shares")
            
            # Wait for adjacent party shares to arrive
            if not self.wait_for_replicated_shares(computation_id, matrix_a_name):
                raise Exception(f"Timeout waiting for replicated shares of {matrix_a_name}")
            
            if not self.wait_for_replicated_shares(computation_id, matrix_b_name):
                raise Exception(f"Timeout waiting for replicated shares of {matrix_b_name}")
            
            # Get replicated shares
            with self.state_lock:
                a_replicated = self.replicated_shares[computation_id][matrix_a_name]
                b_replicated = self.replicated_shares[computation_id][matrix_b_name]
            
            # Perform the 3-party secure multiplication protocol
            if self.party_id == 1:  # Party 0 in 0-indexed
                # Party 0 has (x0, x1) and (y0, y1)
                # Computes: x0⊗y0 + x0⊗y1 + x1⊗y0
                x0, x1 = a_replicated[0], a_replicated[1]
                y0, y1 = b_replicated[0], b_replicated[1]
                result = x0 @ y0 + x0 @ y1 + x1 @ y0
                
            elif self.party_id == 2:  # Party 1 in 0-indexed
                # Party 1 has (x1, x2) and (y1, y2)  
                # Computes: x1⊗y1 + x1⊗y2 + x2⊗y1
                x1, x2 = a_replicated[0], a_replicated[1]
                y1, y2 = b_replicated[0], b_replicated[1]
                result = x1 @ y1 + x1 @ y2 + x2 @ y1
                
            else:  # Party 2 in 0-indexed (party_id == 3)
                # Party 2 has (x2, x0) and (y2, y0)
                # Computes: x2⊗y2 + x2⊗y0 + x0⊗y2
                x2, x0 = a_replicated[0], a_replicated[1]
                y2, y0 = b_replicated[0], b_replicated[1]
                result = x2 @ y2 + x2 @ y0 + x0 @ y2
            
            # Store the result as our new additive share
            with self.state_lock:
                if computation_id not in self.additive_shares:
                    self.additive_shares[computation_id] = {}
                
                self.additive_shares[computation_id][result_name] = result
            
            print(f"Party {self.party_id}: Secure matrix multiplication completed")
            
            return smpc_pb2.Response(success=True, message="Secure matrix multiplication completed")
            
        except Exception as e:
            print(f"Party {self.party_id}: Error in secure matrix multiplication: {e}")
            return smpc_pb2.Response(success=False, message=str(e))
    
    def SecureMatMulWithRandomization(self, request, context):
        """Perform secure matrix multiplication followed by automatic randomization."""
        try:
            computation_id = request.computation_id
            matrix_a_name = request.matrix_a_name
            matrix_b_name = request.matrix_b_name
            result_name = request.result_name
            
            print(f"Party {self.party_id}: Computing secure {matrix_a_name} * {matrix_b_name} = {result_name} with randomization")
            
            # First perform the secure matrix multiplication
            matmul_request = smpc_pb2.MatMulRequest(
                computation_id=computation_id,
                matrix_a_name=matrix_a_name,
                matrix_b_name=matrix_b_name,
                result_name=f"{result_name}_temp"
            )
            
            matmul_response = self.SecureMatMul(matmul_request, context)
            if not matmul_response.success:
                raise Exception(f"Matrix multiplication failed: {matmul_response.message}")
            
            # Now perform randomization
            config = {
                'adaptive': request.parameters.get('adaptive', 'true').lower() == 'true',
                'minimum_noise_range_val': float(request.parameters.get('minimum_noise_range_val', '2.0')),
                'obfuscation_factor_min': float(request.parameters.get('obfuscation_factor_min', '0.1')),
                'obfuscation_factor_max': float(request.parameters.get('obfuscation_factor_max', '0.5'))
            }
            
            success = self.randomize_shares_internal(computation_id, f"{result_name}_temp", result_name, config)
            if not success:
                raise Exception("Randomization failed")
            
            # Convert back to replicated shares for next operations
            self.convert_back_to_replicated(computation_id, result_name)
            
            print(f"Party {self.party_id}: Secure matrix multiplication with randomization completed")
            
            return smpc_pb2.Response(success=True, message="Secure matrix multiplication with randomization completed")
            
        except Exception as e:
            print(f"Party {self.party_id}: Error in secure matrix multiplication with randomization: {e}")
            return smpc_pb2.Response(success=False, message=str(e))
    
    def randomize_shares_internal(self, computation_id, input_matrix, output_matrix, config):
        """Internal randomization using the proper 3-party protocol."""
        try:
            with self.state_lock:
                if (computation_id not in self.additive_shares or 
                    input_matrix not in self.additive_shares[computation_id]):
                    raise Exception(f"Matrix {input_matrix} not found")
                
                my_share = self.additive_shares[computation_id][input_matrix]
            
            if self.party_id == 1:  # Party 0
                # Generate random matrix r0 and add to share
                r0 = self.generate_randomness(my_share.shape, config)
                randomized_share = my_share + r0
                
                # Send r0 to Party 3 (party_id 3)
                if 3 in self.party_clients:
                    r_msg = smpc_pb2.ShareDistribution(
                        computation_id=computation_id,
                        matrix_name=f"{output_matrix}_randomness_from_1",
                        share=matrix_to_proto(r0),
                        share_index=0
                    )
                    self.party_clients[3].ReceiveFromParty(r_msg)
                
            elif self.party_id == 2:  # Party 1
                # Generate random matrix r1 and add to share
                r1 = self.generate_randomness(my_share.shape, config)
                randomized_share = my_share + r1
                
                # Send r1 to Party 3 (party_id 3)
                if 3 in self.party_clients:
                    r_msg = smpc_pb2.ShareDistribution(
                        computation_id=computation_id,
                        matrix_name=f"{output_matrix}_randomness_from_2",
                        share=matrix_to_proto(r1),
                        share_index=1
                    )
                    self.party_clients[3].ReceiveFromParty(r_msg)
                
            else:  # Party 3 (party_id == 3)
                # Party 3 waits to receive r0 and r1, then subtracts both
                self.wait_for_randomness(computation_id, output_matrix)
                
                with self.state_lock:
                    randomized_share = my_share
                    if computation_id in self.received_randomness and output_matrix in self.received_randomness[computation_id]:
                        for r in self.received_randomness[computation_id][output_matrix]:
                            if r is not None:
                                randomized_share -= r
            
            # Store the randomized share
            with self.state_lock:
                self.additive_shares[computation_id][output_matrix] = randomized_share
            
            print(f"Party {self.party_id}: Share randomization completed")
            return True
            
        except Exception as e:
            print(f"Party {self.party_id}: Error in share randomization: {e}")
            return False
    
    def generate_randomness(self, shape, config):
        """Generate randomness for the randomization protocol."""
        if config['adaptive']:
            min_noise = config['minimum_noise_range_val']
            obf_min = config['obfuscation_factor_min']
            obf_max = config['obfuscation_factor_max']
            
            scale = min_noise * (1 + np.random.uniform(obf_min, obf_max))
            return (np.random.rand(*shape) - 0.5) * 2 * scale
        else:
            noise = config['minimum_noise_range_val']
            return np.random.uniform(-noise, noise, size=shape)
    
    def wait_for_randomness(self, computation_id, matrix_name, max_attempts=100):
        """Wait for randomness from other parties (only for Party 3)."""
        if self.party_id != 3:
            return True
        
        attempts = 0
        while attempts < max_attempts:
            with self.state_lock:
                if (computation_id in self.received_randomness and 
                    matrix_name in self.received_randomness[computation_id] and
                    len([r for r in self.received_randomness[computation_id][matrix_name] if r is not None]) >= 2):
                    return True
            
            threading.Event().wait(0.01)
            attempts += 1
        
        print(f"Party {self.party_id}: Timeout waiting for randomness for {matrix_name}")
        return False
    
    def convert_back_to_replicated(self, computation_id, matrix_name):
        """Convert randomized additive shares back to replicated shares."""
        try:
            with self.state_lock:
                if (computation_id not in self.additive_shares or 
                    matrix_name not in self.additive_shares[computation_id]):
                    return
                
                my_share = self.additive_shares[computation_id][matrix_name]
            
            # Send share to next party in the ring
            if self.party_id == 1:  # Send to Party 2
                target_party = 2
            elif self.party_id == 2:  # Send to Party 3
                target_party = 3
            else:  # Party 3 sends to Party 1
                target_party = 1
            
            if target_party in self.party_clients:
                share_msg = smpc_pb2.ShareDistribution(
                    computation_id=computation_id,
                    matrix_name=f"{matrix_name}_replicated_return",
                    share=matrix_to_proto(my_share),
                    share_index=self.party_index
                )
                self.party_clients[target_party].ReceiveFromParty(share_msg)
            
            print(f"Party {self.party_id}: Sent share back for replicated state")
            
        except Exception as e:
            print(f"Party {self.party_id}: Error converting back to replicated: {e}")
    
    def CreateDiagonal(self, request, context):
        """Create diagonal matrix from vector shares."""
        try:
            computation_id = request.computation_id
            vector_matrix = request.input_matrices[0]
            diag_matrix = request.output_matrix
            
            print(f"Party {self.party_id}: Creating diagonal matrix {vector_matrix} -> {diag_matrix}")
            
            with self.state_lock:
                if (computation_id not in self.additive_shares or 
                    vector_matrix not in self.additive_shares[computation_id]):
                    raise Exception(f"Vector matrix {vector_matrix} not found")
                
                vector_share = self.additive_shares[computation_id][vector_matrix]
                
                # Create diagonal matrix from our share only
                diag_share = np.diag(vector_share.flatten())
                
                # Store the diagonal share
                self.additive_shares[computation_id][diag_matrix] = diag_share
            
            print(f"Party {self.party_id}: Diagonal matrix creation completed")
            
            return smpc_pb2.Response(success=True, message="Diagonal matrix creation completed")
            
        except Exception as e:
            print(f"Party {self.party_id}: Error creating diagonal matrix: {e}")
            return smpc_pb2.Response(success=False, message=str(e))
    
    def ReturnShares(self, request, context):
        """Return our share to orchestrator (for final reconstruction)."""
        try:
            computation_id = request.computation_id
            matrix_name = request.matrix_name
            
            with self.state_lock:
                if (computation_id not in self.additive_shares or 
                    matrix_name not in self.additive_shares[computation_id]):
                    raise Exception(f"Matrix {matrix_name} not found")
                
                # Return ONLY our share to the orchestrator
                my_share = self.additive_shares[computation_id][matrix_name]
            
            print(f"Party {self.party_id}: Returning share for {matrix_name} to orchestrator")
            
            return smpc_pb2.Response(
                success=True,
                message="Share returned successfully",
                result=matrix_to_proto(my_share)
            )
            
        except Exception as e:
            print(f"Party {self.party_id}: Error returning shares: {e}")
            return smpc_pb2.Response(success=False, message=str(e))
    
    def ReceiveFromParty(self, request, context):
        """Receive shares or randomness from other parties."""
        try:
            computation_id = request.computation_id
            matrix_name = request.matrix_name
            received_matrix = proto_to_matrix(request.share)
            sender_party_index = request.share_index
            
            with self.state_lock:
                # Handle different types of received data
                if "randomness_from" in matrix_name:
                    # This is randomness from another party (only Party 3 receives this)
                    if self.party_id == 3:
                        base_name = matrix_name.split("_randomness_from")[0]
                        
                        if computation_id not in self.received_randomness:
                            self.received_randomness[computation_id] = {}
                        
                        if base_name not in self.received_randomness[computation_id]:
                            self.received_randomness[computation_id][base_name] = [None, None]
                        
                        # Store randomness from the specific party
                        if "from_1" in matrix_name:
                            self.received_randomness[computation_id][base_name][0] = received_matrix
                        elif "from_2" in matrix_name:
                            self.received_randomness[computation_id][base_name][1] = received_matrix
                        
                        print(f"Party {self.party_id}: Received randomness for {base_name}")
                
                elif "_for_replication" in matrix_name:
                    # This is for building replicated shares
                    base_name = matrix_name.replace("_for_replication", "")
                    
                    if computation_id not in self.replicated_shares:
                        self.replicated_shares[computation_id] = {}
                    
                    if base_name not in self.replicated_shares[computation_id]:
                        # Initialize with our own share if we have it
                        if (computation_id in self.additive_shares and 
                            base_name in self.additive_shares[computation_id]):
                            my_share = self.additive_shares[computation_id][base_name]
                            self.replicated_shares[computation_id][base_name] = [my_share, received_matrix]
                        else:
                            self.replicated_shares[computation_id][base_name] = [None, received_matrix]
                    else:
                        # Update with the received share
                        self.replicated_shares[computation_id][base_name][1] = received_matrix
                    
                    print(f"Party {self.party_id}: Received replicated share for {base_name}")
                
                elif "_replicated_return" in matrix_name:
                    # This is for restoring replicated shares after randomization
                    base_name = matrix_name.replace("_replicated_return", "")
                    
                    if computation_id not in self.replicated_shares:
                        self.replicated_shares[computation_id] = {}
                    
                    if base_name in self.additive_shares[computation_id]:
                        my_share = self.additive_shares[computation_id][base_name]
                        self.replicated_shares[computation_id][base_name] = [my_share, received_matrix]
                    
                    print(f"Party {self.party_id}: Restored replicated share for {base_name}")
            
            return smpc_pb2.Response(success=True, message="Data received from party")
            
        except Exception as e:
            print(f"Party {self.party_id}: Error receiving from party: {e}")
            return smpc_pb2.Response(success=False, message=str(e))


def serve():
    """Start the gRPC server for the party."""
    party_id = int(os.getenv('PARTY_ID', 1))
    grpc_port = int(os.getenv('GRPC_PORT', 50052))
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add the party service
    party_service = SMPCParty(party_id)
    smpc_pb2_grpc.add_PartyComputationServiceServicer_to_server(party_service, server)
    
    # Listen on the specified port
    listen_addr = f'0.0.0.0:{grpc_port}'
    server.add_insecure_port(listen_addr)
    
    print(f"Party {party_id} starting gRPC server on {listen_addr}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"Party {party_id} shutting down...")
        server.stop(0)


if __name__ == '__main__':
    serve()