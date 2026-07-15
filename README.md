# LaBelle
## Llama Transformer Runtime: Custom CUDA Kernels, GPU Optimization Techniques and Performance Analysis

### **Strictly limited to sub-2B parameter models**

```
PyTorch ONNX TensorRT CUDA FastAPI Linux Docker Triton Kubernetes Prometheus/Grafana Redis/Kafka Nsight Systems, Nsight Compute, onnx, trt, Prometheus, Grafana, Weights & Biases.
```
---

* Bottom-up development to make debugging manageable.
* Incremental optimization rather than optimizing everything at the end.
* Inference only to avoid the substantially greater complexity of training.
* Single-batch inference to keep the focus on runtime execution rather than request scheduling.
* Sub-2B model size to match the available hardware and allow extensive experimentation.

```
CUDA Events
Prometheus
Grafana
Occupancy Analysis
Memory Analysis
Kernel Fusion
```

```
Specifying "single-batch inference" and "sub-1B parameter model" 
```

```
Phase 1 – CUDA Primitive Construction and Optimization
Phase 2 – Transformer Block Construction and Optimization
Phase 3 – Full LLaMA Runtime Construction and Optimization
Phase 4 – Large-Scale Runtime Optimization
Phase 5 – Integration with LLaMA Model Weights
Phase 6 – Single-Batch Inference Implementation
Phase 7 – Comprehensive Benchmarking and Performance Evaluation
```

```
Execution time
Throughput
Occupancy
Register usage
Shared memory usage
Global memory throughput
Warp execution efficiency
SM utilization
L2 cache hit rate
Achieved FLOPS
Arithmetic intensity
```


sudo /usr/local/cuda/bin/ncu ./rmsnorm
sudo /usr/local/cuda/bin/ncu ./any compiled kernel binaries

if this happens 
Memory analysis

Your output says:

DRAM Throughput 84.96%
Memory Throughput 84.96%
Compute Throughput 35.89%

This screams:

memory-bound kernel

So collect memory:

sudo /usr/local/cuda/bin/ncu \
--section MemoryWorkloadAnalysis \
./rmsnorm

You will see:

global load efficiency
global store efficiency
cache hit rates
bandwidth utilization


```
model specs

Parameters:
1.23B

Layers:
16

Hidden dimension:
2048

Attention heads:
32

KV heads:
8  (GQA)

Head dimension:
64

Context:
128k

Datatype:
BF16
```


