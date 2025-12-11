# Three-Filters-to-Normal build recipes

# Default parameters (can be overridden)
input := "../../matlab_code/torusknot/depth/000001.bin"
width := "640"
height := "480"
fx := "1400"
fy := "1380"
uo := "350"
vo := "200"
offset := "600"
kernel := "basic"       # "basic" or "sobel"
aggregation := "median" # "mean" or "median"
threshold := "0.01"
fail_percent := "10.0"

# Default recipe
default:
    @just --list

# Build all
all: build-cpp build-cuda build-test

# === C++ ===

# Build C++ library
build-cpp:
    cd c_code && mkdir -p build && cd build && cmake .. && make -j4

# Clean C++ build
clean-cpp:
    rm -rf c_code/build

# Run C++ example
run-cpp: build-cpp
    cd c_code/example && ./demo

# === CUDA ===

# Build CUDA implementation
build-cuda:
    cd cuda_code && mkdir -p build && cd build && cmake .. && make -j4

# Clean CUDA build
clean-cuda:
    rm -rf cuda_code/build

# Run CUDA example
run-cuda: build-cuda
    cd cuda_code/build/src && ./surface_normal_estimation_cvpr

# === Test ===

# Build test suite
build-test: build-cpp
    cd test && mkdir -p build && cd build && cmake .. && make -j4

# Clean test build
clean-test:
    rm -rf test/build

# Run comparison test with default parameters
test: build-test
    cd test/build && \
    ./cpp_test \
        --input {{input}} \
        --output cpp_normal \
        --width {{width}} --height {{height}} \
        --fx {{fx}} --fy {{fy}} --uo {{uo}} --vo {{vo}} \
        --offset {{offset}} --kernel {{kernel}} --aggregation {{aggregation}} && \
    ./cuda_test \
        --input {{input}} \
        --output cuda_normal \
        --width {{width}} --height {{height}} \
        --fx {{fx}} --fy {{fy}} --uo {{uo}} --vo {{vo}} \
        --offset {{offset}} --kernel {{kernel}} --aggregation {{aggregation}} && \
    ./compare_results \
        --cpp cpp_normal.bin --cuda cuda_normal.bin \
        --width {{width}} --height {{height}} \
        --threshold {{threshold}} --fail-percent {{fail_percent}}

# Run C++ test only
test-cpp: build-test
    cd test/build && ./cpp_test \
        --input {{input}} \
        --output cpp_normal \
        --width {{width}} --height {{height}} \
        --fx {{fx}} --fy {{fy}} --uo {{uo}} --vo {{vo}} \
        --offset {{offset}} --kernel {{kernel}} --aggregation {{aggregation}}

# Run CUDA test only
test-cuda: build-test
    cd test/build && ./cuda_test \
        --input {{input}} \
        --output cuda_normal \
        --width {{width}} --height {{height}} \
        --fx {{fx}} --fy {{fy}} --uo {{uo}} --vo {{vo}} \
        --offset {{offset}} --kernel {{kernel}} --aggregation {{aggregation}}

# Compare results only
compare: build-test
    cd test/build && ./compare_results \
        --cpp cpp_normal.bin --cuda cuda_normal.bin \
        --width {{width}} --height {{height}} \
        --threshold {{threshold}} --fail-percent {{fail_percent}}

# Show help for test programs
test-help: build-test
    @echo "=== C++ Test ===" && cd test/build && ./cpp_test --help
    @echo ""
    @echo "=== CUDA Test ===" && cd test/build && ./cuda_test --help
    @echo ""
    @echo "=== Compare Results ===" && cd test/build && ./compare_results --help

# === Demo ===

# Run C++ demo with default parameters
demo-cpp: build-test
    @echo "=== C++ Demo ({{kernel}} + {{aggregation}}) ==="
    cd test/build && ./cpp_test \
        --input {{input}} \
        --output cpp_demo \
        --width {{width}} --height {{height}} \
        --fx {{fx}} --fy {{fy}} --uo {{uo}} --vo {{vo}} \
        --offset {{offset}} --kernel {{kernel}} --aggregation {{aggregation}}
    @echo "Output: test/build/cpp_demo.png"

# Run CUDA demo with default parameters
demo-cuda: build-test
    @echo "=== CUDA Demo ({{kernel}} + {{aggregation}}) ==="
    cd test/build && ./cuda_test \
        --input {{input}} \
        --output cuda_demo \
        --width {{width}} --height {{height}} \
        --fx {{fx}} --fy {{fy}} --uo {{uo}} --vo {{vo}} \
        --offset {{offset}} --kernel {{kernel}} --aggregation {{aggregation}}
    @echo "Output: test/build/cuda_demo.png"

# Run both demos
demo: demo-cpp demo-cuda
    @echo "=== Both demos completed ==="
    @echo "C++ output: test/build/cpp_demo.png"
    @echo "CUDA output: test/build/cuda_demo.png"

# === Clean ===

# Clean all builds
clean: clean-cpp clean-cuda clean-test
    rm -rf temp/*.bin temp/*.png

# === Utility ===

# Check common headers syntax
check-headers:
    @echo "Checking test/common headers..."
    @g++ -fsyntax-only test/common/tftn_config.h && echo "tftn_config.h: OK"

# Check dependencies
check-deps:
    @echo "Checking OpenCV..."
    @pkg-config --modversion opencv4 || echo "OpenCV4 not found"
    @echo "Checking CUDA..."
    @nvcc --version || echo "CUDA not found"
    @echo "Checking Eigen..."
    @pkg-config --modversion eigen3 || echo "Eigen3 not found"

# Install dependencies (Ubuntu/Debian)
install-deps:
    sudo apt-get install -y libeigen3-dev libopencv-dev libboost-all-dev

# Format check (placeholder)
fmt:
    @echo "No formatter configured"

# Lint check (placeholder)
lint:
    @echo "No linter configured"
