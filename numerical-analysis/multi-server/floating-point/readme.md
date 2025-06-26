# Secure Multi-Party Computation (SMPC) Docker System

This system implements a distributed secure multi-party computation platform for matrix operations using floating-point arithmetic. The system consists of one orchestrator and three computation parties that communicate via gRPC for matrix streaming and HTTP for control operations.

## Architecture

- **Orchestrator**: Manages the computation workflow, provides web interface, handles matrix generation and result analysis
- **Computation Parties (3)**: Execute secure matrix operations using secret sharing protocols
- **Communication**: 
  - HTTP for orchestrator web interface and control
  - gRPC for efficient matrix streaming between components

## Features

- Interactive web interface for parameter configuration

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 8GB RAM (for larger matrix computations)

### Setup and Run

1. **Setup the system:**
```bash
chmod +x setup.sh
./setup.sh
```

2. **Build the Container:**
```bash
./manage.sh rebuild
```


3. **Start the system:**
```bash
./manage.sh start
```

4. **Stop the system:**
```bash
./manage.sh stop
```

5. **Access the web interface:**
   - Open your browser to `http://localhost:8080`
   - Configure computation parameters
   - Start computations and view results

### Default Ports

- Orchestrator Web Interface: `8080` (HTTP)
- Orchestrator gRPC: `50051`
- Party 1 gRPC: `50052`
- Party 2 gRPC: `50053`
- Party 3 gRPC: `50054`

## Usage

### Web Interface

The orchestrator provides an intuitive web interface for:

1. **Parameter Configuration:**
   - Matrix dimensions (N, R)
   - Density and outlier settings
   - Security parameters (noise levels, obfuscation factors)
   - Adaptive sharing options

2. **Computation Management:**
   - Start new computations
   - View results

3. **Results Analysis:**
   - Error metrics (absolute, relative, SNR)
   - Performance measurements
   - Export to CSV

### API Endpoints

- `POST /api/start_computation` - Start a new computation
- `GET /api/get_results` - Retrieve all results
- `GET /api/download_results` - Download results as CSV


## Security Protocol

The system implements a proper 3-party secure computation protocol with replicated secret sharing:

### Protocol Overview

1. **Initial Share Distribution (Additive Shares)**:
   - Orchestrator generates additive shares: `x = x0 + x1 + x2`
   - Party 1 receives only `x0`
   - Party 2 receives only `x1` 
   - Party 3 receives only `x2`

2. **Conversion to Replicated Shares**:
   - Parties exchange shares to establish replicated state:
     - Party 1 has `(x0, x1)` 
     - Party 2 has `(x1, x2)`
     - Party 3 has `(x2, x0)`

3. **Secure Matrix Multiplication**:
   - Each party computes specific matrix products based on their replicated shares:
     - Party 1: `x0⊗y0 + x0⊗y1 + x1⊗y0`
     - Party 2: `x1⊗y1 + x1⊗y2 + x2⊗y1`  
     - Party 3: `x2⊗y2 + x2⊗y0 + x0⊗y2`
   - Result is new additive shares (no longer replicated)

4. **Share Randomization**:
   - Party 1: generates `r0`, adds to share, sends `r0` to Party 3
   - Party 2: generates `r1`, adds to share, sends `r1` to Party 3
   - Party 3: receives `r0` and `r1`, subtracts both from its share
   - Preserves the secret while refreshing share distribution

5. **Return to Replicated State**:
   - Parties exchange their randomized shares in a ring:
     - Party 1 → Party 2
     - Party 2 → Party 3  
     - Party 3 → Party 1
   - Now ready for next secure multiplication

6. **Final Reconstruction**:
   - Each party sends **only their share** to orchestrator
   - Orchestrator reconstructs: `result = share1 + share2 + share3`


## Development

### Adding New Operations

1. Define protobuf messages in `proto/smpc.proto`
2. Implement orchestrator logic in `orchestrator/main.py`
3. Add party computation in `party/main.py`

### Any Error? See here
```bash
./manage.sh logs
```



