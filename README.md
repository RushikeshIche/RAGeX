# RAGeX

**RAGeX** is an extreme low latency, memory efficient Retrieval Augmented Generation (RAG) system built from scratch in pure C. 

It is specifically designed to run on resource constrained edge devices (like the Raspberry Pi or an Android phone via Termux) with **less than 512 MB of RAM, no GPU, and no network connection**. 

By completely bypassing massive frameworks like PyTorch or Python, RAGeX implements its own custom memory allocators, ARM NEON SIMD optimizations, and bare-metal mathematical operations to run state-of-the-art quantized AI models on the edge.

---

## Key Features

* **100% Offline & Self-Contained:** Zero cloud dependencies, zero external APIs, and no Python interpreter required at runtime.
* **Tiny Footprint:** Targets a total RAM usage of < 512 MB using custom bump/pool memory allocators to prevent heap fragmentation.
* **Hardware Accelerated:** Uses ARM NEON intrinsics to perform single-instruction, multiple-data (SIMD) matrix multiplications (GEMM).
* **Memory Mapped Tensors:** Neural network weights are stored in custom raw binary files and loaded instantly via `mmap` for zero-overhead startup.
* **Cross-Platform C:** Configured via CMake to compile natively on Windows (MSVC), Linux (GCC/Clang), and ARM edge devices.

---

## Technical Stack
* **Language:** Pure C (C11)
* **Build System:** CMake
* **SIMD Optimization:** ARM NEON (`<arm_neon.h>`)
* **Embedding Model:** `all-MiniLM-L6-v2` (Exported to custom binary)

---

## Getting Started

Follow these steps to set up, build, and run the project locally.

### 1. Export the Model Weights

RAGeX cannot read HuggingFace/PyTorch models directly. You must first use the provided Python script to download the model and convert it into a raw binary format (`.bin`).

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install the export dependencies
pip install torch transformers

# Run the exporter script
python scripts/export_model.py
```
This will generate a `models/minilm_weights.bin` file containing the raw, flattened tensor data.

### 2. Build the C Project

RAGeX uses CMake. Ensure you have CMake and a C compiler installed.

```bash
# 1. Configure the build directory
cmake -B build

# 2. Compile the project
cmake --build build
```

### 3. Run the Benchmarks

To verify that your build was successful and to measure the speed of the matrix multiplication and memory allocators:

```bash
# On Linux/macOS/Termux:
./build/benchmark

# On Windows (MSVC):
./build/Debug/benchmark.exe
```
*Note: If you are running on an ARM device, the benchmark will automatically test and compare the hardware accelerated NEON GEMM against the baseline C implementation.*

---

## Roadmap (Current Progress)

- [x] **Phase 1: Foundation** (CMake, Custom Memory Allocators, ARM NEON GEMM)
- [x] **Phase 2: Embedding Model Conversion** (Python INT8 Quantized Exporter)
- [x] **Phase 2: Model Loading** (Zero-copy `mmap` loader, C struct definitions)
- [x] **Phase 3: Transformer Inference (In Progress)**
  - [x] Token & Positional Embedding forward pass
  - [x] Multi-Head Self-Attention (with NEON optimization)
  - [x] Feed-Forward Network (FFN) with GELU activation
  - [x] Layer Normalization
  - [ ] Full Transformer block combination (Task 23)
  - [ ] Validation against PyTorch output (Task 24)
- [ ] **Phase 4: Tokenizer & Document Chunker** (BPE, chunking)
- [ ] **Phase 5: Vector Index** (IVF + PQ for fast retrieval)
- [ ] **Phase 6: Tiny LLM Generation** (4-bit quantization, Llama architecture)
- [ ] **Phase 7: Integration** (CLI/Web UI)
