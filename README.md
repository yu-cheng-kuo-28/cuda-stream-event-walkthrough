Medium article: 

[CUDA 01 — Streams & Events Walkthrough](https://yc-kuo.medium.com/cuda-01-streams-events-walkthrough-5ff0a32fc1ea?postPublishedType=initial)


---

# CUDA Streams & Events Walkthrough

Hands-on CUDA asynchrony, streams, and events by measurement.

This repository contains complete, working code examples demonstrating CUDA streams and events, from basic concepts to advanced overlap optimizations.

## Overview

This project accompanies the Medium article **"CUDA 01 - Streams & Events Walkthrough"** and provides:

- ✅ Fully working CUDA code examples
- ✅ Makefile for easy compilation
- ✅ Progressive examples from basic to advanced
- ✅ Measurable performance demonstrations

## System Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.0 or later recommended)
- GCC/G++ compiler
- Linux/WSL2 or native Linux environment

### Check Your Setup

```bash
nvidia-smi      # Verify GPU and driver
nvcc --version  # Verify CUDA toolkit
```

Example system used:
- **GPU**: NVIDIA GeForce MX450
- **CUDA Toolkit**: 11.5
- **Driver**: 561.19

## Quick Start

### Build All Examples

```bash
make
```

### Run All Examples

```bash
make run
```

### Clean Up

```bash
make clean
```

## Code Examples

### 1. `01_hello.cu` - Basic CUDA Kernel
Introduction to CUDA execution model with a simple kernel.

```bash
make 01_hello
./01_hello
```

**Expected output**: Hello messages from 6 threads (2 blocks × 3 threads)

---

### 2. `02_streams_basic.cu` - Stream Creation & Usage
Demonstrates how to create streams, launch kernels in different streams, and synchronize.

```bash
make 02_streams_basic
./02_streams_basic
```

**Concepts covered**:
- `cudaStreamCreate()`
- `cudaStreamSynchronize()`
- `cudaStreamDestroy()`

---

### 3. `03_async_memcpy.cu` - Async Memory Transfers
Shows async memory operations with pinned memory.

```bash
make 03_async_memcpy
./03_async_memcpy
```

**Concepts covered**:
- `cudaMallocHost()` (pinned memory)
- `cudaMemcpyAsync()`
- Async data transfer pipeline

---

### 4. `04_event_timing.cu` - Timing with Events
Precise GPU kernel timing using CUDA events.

```bash
make 04_event_timing
./04_event_timing
```

**Concepts covered**:
- `cudaEventCreate()`
- `cudaEventRecord()`
- `cudaEventElapsedTime()`

---

### 5. `05_cross_stream_sync_bad.cu` - Race Condition Demo
Demonstrates **what happens without proper synchronization**.

```bash
make 05_cross_stream_sync_bad
./05_cross_stream_sync_bad
```

**What to observe**: Consumer may read stale data (race condition).

---

### 6. `06_cross_stream_sync_good.cu` - Proper Cross-Stream Sync
Shows correct cross-stream synchronization using events.

```bash
make 06_cross_stream_sync_good
./06_cross_stream_sync_good
```

**Concepts covered**:
- `cudaEventRecord()` in one stream
- `cudaStreamWaitEvent()` in another stream
- Dependency management across streams

---

### 7. `07_overlap_demo.cu` - Performance Comparison ⚡
**Complete demonstration** comparing serial vs. multi-stream execution with real measurements.

```bash
make 07_overlap_demo
./07_overlap_demo
```

**Expected output** (actual times vary by GPU):
```
Serial:  45.23 ms
Streams: 28.17 ms   ← ~38% faster from overlap
```

**What it demonstrates**:
- Overlapping H→D transfer, compute, and D→H transfer
- Chunked data processing across multiple streams
- Measurable speedup from concurrency

---

## Project Structure

```
.
├── Makefile                         # Build automation
├── README.md                        # This file
├── cuda_check.h                     # Error checking utilities
├── 01_hello.cu                      # Basic kernel demo
├── 02_streams_basic.cu              # Stream creation/usage
├── 03_async_memcpy.cu               # Async memory transfers
├── 04_event_timing.cu               # Event-based timing
├── 05_cross_stream_sync_bad.cu      # Race condition demo
├── 06_cross_stream_sync_good.cu     # Proper synchronization
└── 07_overlap_demo.cu               # Performance comparison
```

## Makefile Targets

| Command | Description |
|---------|-------------|
| `make` | Build all examples |
| `make clean` | Remove all binaries |
| `make run` | Build and run all examples |
| `make help` | Show help message |
| `make <example>` | Build specific example (e.g., `make 07_overlap_demo`) |

## Key Concepts Summary

| Concept | Purpose | Key API |
|---------|---------|---------|
| **Stream** | Independent execution queue | `cudaStreamCreate`, `cudaStreamSynchronize` |
| **Event** | Timestamp / sync point | `cudaEventRecord`, `cudaEventElapsedTime` |
| **Async memcpy** | Non-blocking transfer | `cudaMemcpyAsync` + pinned memory |
| **Cross-stream sync** | Dependency between streams | `cudaStreamWaitEvent` |

## Common Pitfalls

1. **Using `cudaMemcpy` instead of `cudaMemcpyAsync`**
   - Problem: Blocks entire device
   - Solution: Use async version with streams

2. **Forgetting pinned memory**
   - Problem: Async copy silently falls back to sync
   - Solution: Use `cudaMallocHost()`

3. **Default stream synchronization**
   - Problem: Stream 0 syncs with all streams
   - Solution: Use explicit non-default streams

4. **Too many streams**
   - Problem: Overhead exceeds benefit
   - Solution: 2-8 streams typically optimal

## Profiling

To visualize actual overlap, use NVIDIA profilers:

```bash
# Using nvprof (legacy)
nvprof ./07_overlap_demo

# Using Nsight Systems (recommended)
nsys profile --trace=cuda ./07_overlap_demo
```

## References

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA C/C++ Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)
- [Coursera: CUDA at Scale for the Enterprise (JHU)](https://www.coursera.org/learn/cuda-at-scale-for-the-enterprise)

## License

MIT License - Feel free to use for learning and teaching.

## Contributing

Found an issue or want to add an example? Pull requests welcome!

---

**Found this useful?** ⭐ Star this repo and share it with others learning CUDA programming!