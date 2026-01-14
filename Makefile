# Makefile for CUDA Streams & Events Walkthrough
# Usage:
#   make          - Build all examples
#   make clean    - Remove all binaries
#   make run      - Build and run all examples

# Compiler and flags
NVCC = nvcc
# NVCCFLAGS = -O2 -arch=sm_75
NVCCFLAGS = -O2 -arch=sm_75

# Target binaries
TARGETS = 01_hello \
          02_streams_basic \
          03_async_memcpy \
          04_event_timing \
          05_cross_stream_sync_bad \
          06_cross_stream_sync_good \
          07_overlap_demo \
          08_overlap_demo_colab

# Default target: build all
all: $(TARGETS)

# Pattern rule for compiling .cu files
%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Individual targets (explicit dependencies)
01_hello: 01_hello.cu
02_streams_basic: 02_streams_basic.cu
03_async_memcpy: 03_async_memcpy.cu
04_event_timing: 04_event_timing.cu
05_cross_stream_sync_bad: 05_cross_stream_sync_bad.cu
06_cross_stream_sync_good: 06_cross_stream_sync_good.cu
07_overlap_demo: 07_overlap_demo.cu
08_overlap_demo_colab: 08_overlap_demo_colab.cu

# Clean up binaries
clean:
	rm -f $(TARGETS)

# Run all examples (optional)
run: all
	@echo "=== Running 01_hello ==="
	./01_hello
	@echo ""
	@echo "=== Running 02_streams_basic ==="
	./02_streams_basic
	@echo ""
	@echo "=== Running 03_async_memcpy ==="
	./03_async_memcpy
	@echo ""
	@echo "=== Running 04_event_timing ==="
	./04_event_timing
	@echo ""
	@echo "=== Running 05_cross_stream_sync_bad (race condition demo) ==="
	./05_cross_stream_sync_bad
	@echo ""
	@echo "=== Running 06_cross_stream_sync_good (proper synchronization) ==="
	./06_cross_stream_sync_good
	@echo ""
	@echo "=== Running 07_overlap_demo (performance comparison) ==="
	./07_overlap_demo
	@echo ""
	@echo "=== Running 08_overlap_demo_colab (GPU capability check) ==="
	./08_overlap_demo_colab

# Help target
help:
	@echo "CUDA Streams & Events Walkthrough - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build all examples"
	@echo "  make clean    - Remove all binaries"
	@echo "  make run      - Build and run all examples"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Individual examples:"
	@echo "  make 01_hello                   - Basic kernel demo"
	@echo "  make 02_streams_basic           - Stream creation and usage"
	@echo "  make 03_async_memcpy            - Async memory transfers"
	@echo "  make 04_event_timing            - Event-based timing"
	@echo "  make 05_cross_stream_sync_bad   - Race condition demo"
	@echo "  make 06_cross_stream_sync_good  - Proper cross-stream sync"
	@echo "  make 07_overlap_demo            - Performance comparison"
	@echo "  make 08_overlap_demo_colab      - GPU capability check version"

.PHONY: all clean run help
