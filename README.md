# ML Inference Cluster Simulator

A Discrete Event Simulator (DES) for modeling Large Language Model (LLM) inference clusters.

This tool helps you estimate the required cluster size (e.g., number of H100s) to meet specific latency SLAs (P90, P99) under high request loads.

## 🚀 Features

-   **Discrete Event Simulation**: Models request arrival (Poisson process), queuing, and processing time.
-   **Hardware Profiles**: Includes presets for **NVIDIA H100, A100, GB200**, and Google **TPU v5**.
-   **Interactive Web UI**: Built with Streamlit for real-time visualization of latency histograms and throughput.
-   **Advanced Workloads**: Support for Gaussian distributions of input/output token lengths.
-   **SLA Enforcement**: Optional logic to drop requests that miss strict latency deadlines.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/madhan80/ml-inference-sim.git
    cd ml-inference-sim
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

### 1. Web Dashboard (Recommended)
The easiest way to use the simulator is via the web UI.

```bash
streamlit run ml_inference_sim/app.py
```

This will open a dashboard in your browser (`http://localhost:8501`) where you can:
-   Select hardware (e.g., `NVIDIA_H100`).
-   Set traffic load (Requests Per Minute).
-   Visualize P99 latency and cluster saturation.

### 2. Command Line Interface

**Basic Simulation:**
Run a single headless simulation with default parameters:
```bash
python3 -m ml_inference_sim.simulator
```

**Capacity Finder:**
Find the maximum sustainable RPM for a given configuration using binary search:
```bash
python3 -m ml_inference_sim.capacity
```

## 🧠 Hardware Modeling
The simulator estimates request duration using:
$$ \text{Duration} = \text{TTFT} + \frac{\text{Output Tokens}}{\text{Tokens Per Second}} $$

Included profiles (estimates):
-   **NVIDIA H100**: ~110 tokens/s decode
-   **NVIDIA GB200**: ~220 tokens/s decode
-   **TPU v5p**: ~95 tokens/s decode

*Note: You can define custom hardware in `hardware.py` or the UI.*
