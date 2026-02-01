from dataclasses import dataclass

@dataclass
class HardwareProfile:
    name: str
    description: str
    ttft_ms: float  # Time To First Token for a reference model (e.g. Llama 70B)
    output_tokens_per_sec: float # Decode speed per device (or tensor parallel group)
    max_batch_size: int
    memory_gb: int

# Note: These are rough estimates for a Llama-3-70B scale model executing with 
# standard optimizations (vLLM/TensorRT-LLM) on a single instance of the hardware 
# (or a standard node).
# Real performance varies wildly based on quantization (FP8/FP16), TP-degree, etc.

HARDWARE_DB = {
    "NVIDIA_A100_80GB": HardwareProfile(
        name="NVIDIA A100 (80GB)",
        description="Standard workhorse for LLM inference (Ampere).",
        ttft_ms=150.0,
        output_tokens_per_sec=45.0,
        max_batch_size=128,
        memory_gb=80
    ),
    "NVIDIA_H100": HardwareProfile(
        name="NVIDIA H100 (CNX)",
        description="High performance Hopper architecture with FP8 support.",
        ttft_ms=70.0,
        output_tokens_per_sec=110.0,
        max_batch_size=256,
        memory_gb=80
    ),
    "NVIDIA_GB200": HardwareProfile(
        name="NVIDIA GB200 (Blackwell)",
        description="Next-gen unified memory architecture. Extremely high throughput.",
        ttft_ms=40.0,
        output_tokens_per_sec=220.0, 
        max_batch_size=512,
        memory_gb=192 # Combined/Unified memory view often larger
    ),
    "Google_TPU_v5e": HardwareProfile(
        name="Google TPU v5e",
        description="Efficient inference-optimized TPU pod slice.",
        ttft_ms=180.0,
        output_tokens_per_sec=35.0,
        max_batch_size=64,
        memory_gb=32 # Per chip, usually aggregated
    ),
    "Google_TPU_v5p": HardwareProfile(
        name="Google TPU v5p",
        description="Performance-tier TPU for training and heavy inference.",
        ttft_ms=80.0,
        output_tokens_per_sec=95.0,
        max_batch_size=128,
        memory_gb=95
    ),
    "Custom": HardwareProfile(
        name="Custom Hardware",
        description="User defined specifications.",
        ttft_ms=100.0,
        output_tokens_per_sec=50.0,
        max_batch_size=128,
        memory_gb=80
    )
}

def get_profile(key: str) -> HardwareProfile:
    return HARDWARE_DB.get(key, HARDWARE_DB["Custom"])
