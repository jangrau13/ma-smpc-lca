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
- Secure 3-party computation using additive secret sharing
- Matrix operations: multiplication, inversion, diagonal creation
- Adaptive noise scaling and share randomization
- Comprehensive error and leakage analysis
- Real-time results visualization and CSV export

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM (for larger matrix computations)

### Setup and Run

1. **Clone or create the project structure:**
```bash
mkdir smpc-system && cd smpc-system
```

2. **Create the directory structure:**
```bash
mkdir -p orchestrator party common proto results
```

3. **Copy all the provided files to their respective directories:**
   - `docker-compose.yml` (root)
   - `Dockerfile.orchestrator` (root)
   - `Dockerfile.party` (root)
   - `requirements.txt` (root)
   - `proto/smpc.proto`
   - `orchestrator/main.py`
   - `party/main.py`
   - `common/utils.py`
   - `common/__init__.py` (empty file)

4. **Build and start the system:**
```bash
docker-compose up --build
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
   - Monitor progress
   - View real-time results

3. **Results Analysis:**
   - Error metrics (absolute, relative, SNR)
   - Performance measurements
   - Privacy leakage analysis
   - Export to CSV

### API Endpoints

- `POST /api/start_computation` - Start a new computation
- `GET /api/get_results` - Retrieve all results
- `GET /api/download_results` - Download results as CSV

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `N` | Matrix dimension | 100 |
| `R` | Row dimension for secret matrix B | 100 |
| `A_DENSITY` | Sparsity of matrix A | 0.1 |
| `INTRODUCE_OUTLIERS` | Add outliers to matrix A | true |
| `OUTLIER_PROBABILITY` | Probability of outlier elements | 0.11 |
| `USE_ADAPTIVE_SHARING` | Use adaptive noise scaling | true |
| `MINIMUM_NOISE_RANGE_VAL` | Base noise level | 2.0 |

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

### Security Guarantees

- **No party ever sees more than 2 out of 3 shares**
- **Parties cannot reconstruct secrets during computation**
- **Communication is minimal and only between adjacent parties**
- **Randomization prevents correlation attacks across operations**

### Supported Operations

- **Secure Matrix Multiplication**: Using Beaver triplet strategy
- **Share Randomization**: Preserve secrets while refreshing shares
- **Diagonal Matrix Creation**: From vector shares
- **Matrix Inversion**: Public operation on reconstructed matrices

## Development

### Adding New Operations

1. Define protobuf messages in `proto/smpc.proto`
2. Implement orchestrator logic in `orchestrator/main.py`
3. Add party computation in `party/main.py`
4. Update utility functions in `common/utils.py`

### Debugging

- View orchestrator logs: `docker-compose logs orchestrator`
- View party logs: `docker-compose logs party1` (or party2, party3)
- Access containers: `docker-compose exec orchestrator bash`

### Performance Tuning

- Adjust matrix dimensions based on available memory
- Modify noise parameters for security/accuracy trade-offs
- Scale party count (requires protocol modifications)

## Research Applications

This system is designed for academic research in:

- Secure multi-party computation protocols
- Privacy-preserving machine learning
- Cryptographic protocol analysis
- Distributed matrix computations

### Metrics and Analysis

The system provides comprehensive metrics for research evaluation:

- **Accuracy**: Multiple error measures (L2 norm, max/mean absolute/relative errors)
- **Performance**: Computation times, memory usage
- **Security**: Privacy leakage analysis via confusion matrices
- **Scalability**: Support for various matrix sizes and configurations

### Citation

If you use this system in your research, please cite:

```bibtex
@software{smpc_docker_system,
  title={Distributed Secure Multi-Party Computation System},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8080, 50051-50054 are available
2. **Memory issues**: Reduce matrix dimensions for large computations
3. **gRPC connection errors**: Check Docker network connectivity
4. **Build failures**: Verify all files are in correct directories

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs orchestrator
docker-compose logs party1

# Access container shell
docker-compose exec orchestrator bash
```

## License

This project is released under the MIT License for academic and research use.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

For major changes, please open an issue first to discuss proposed modifications.