# PEP-LCA (Privacy-Enhancing-Protocol for LCA)

This README provides instructions on how to run the different analyses and generate diagrams used for the PEP-LCA master thesis.

## Speed Analysis

To run the speed analysis, follow these steps:

1.  Navigate to the `speed-analysis` directory:
    ```bash
    cd speed-analysis
    ```
2.  Source the `setup.sh` script to set up the environment:
    ```bash
    source setup.sh
    ```
3.  Run the speed experiments locally:
    ```bash
    python run-speed-experiments-locally.py
    ```
4.  Analyze the results (including the already provided GPUs):
    ```bash
    python analysis.py
    ```

## Numerical Analysis

The numerical analysis can be performed in two ways: single-server or multi-server.

### Single-Server Setup

This setup is ideal for running experiments on a single machine, especially if a GPU is available.

1.  Navigate to the `single-server` directory within the `numerical-analysis` folder.
2.  Input the desired testing variables (details should be provided in the documentation within this folder).
3.  Run the experiments.
4.  After the experiments are complete, navigate to the `numerical-analysis` directory:
    ```bash
    cd numerical-analysis
    ```
5.  Source the `setup.sh` script:
    ```bash
    source setup.sh
    ```
6.  Run the analysis to generate the graphics:
    ```bash
    python analysis.py
    ```

### Multi-Server Setup

This setup is for a multi-party computation (MPC) environment, is CPU-only, and requires Docker.

1.  Ensure Docker is installed and running on your system.
2.  Navigate to the `multi-server` directory within the `numerical-analysis` folder.
3.  Follow the instructions within that directory to build and run the Docker containers for the multi-party setup.
4.  After the experiments are complete, navigate to the `numerical-analysis` directory:
    ```bash
    cd numerical-analysis
    ```
5.  Source the `setup.sh` script:
    ```bash
    source setup.sh
    ```
6.  Run the analysis to generate the graphics:
    ```bash
    python analysis.py
    ```

## Diagrams

The diagrams in this project are created using PlantUML.

### PlantUML Installation

To generate the diagrams, you need to have PlantUML installed. PlantUML requires Java. For some diagrams, Graphviz is also required.

**1. Install Java:**

If you don't have Java installed, you can download it from the official [Java website](https://www.java.com/en/download/).

**2. Install Graphviz:**

* **macOS (using Homebrew):**
    ```bash
    brew install graphviz
    ```
* **Windows (using Chocolatey):**
    ```bash
    choco install graphviz
    ```
* **Linux (using apt):**
    ```bash
    sudo apt-get update
    sudo apt-get install graphviz
    ```

**3. Install PlantUML:**

* **macOS (using Homebrew):**
    ```bash
    brew install plantuml
    ```
* **Windows (using Chocolatey):**
    ```bash
    choco install plantuml
    ```
* **Manual Installation (using the .jar file):**
    1.  Download the PlantUML `.jar` file from the [PlantUML website](https://plantuml.com/download).
    2.  You can then run PlantUML from the command line using `java -jar plantuml.jar`.

### Generating PNGs

To create `.png` files from the PlantUML source files (`.puml`), you can use the following command in your terminal:

```bash
java -jar path/to/plantuml.jar /path/to/diagrams/*.puml