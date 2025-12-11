# Three-Filters-to-Normal build recipes

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

# Run comparison test
test: build-test
    cd test/build && \
    ./cpp_test ../../matlab_code/torusknot/depth/000001.bin && \
    ./cuda_test ../../matlab_code/torusknot/depth/000001.bin && \
    ./compare_results cpp_normal.bin cuda_normal.bin

# Run C++ test only
test-cpp: build-test
    cd test/build && ./cpp_test ../../matlab_code/torusknot/depth/000001.bin

# Run CUDA test only
test-cuda: build-test
    cd test/build && ./cuda_test ../../matlab_code/torusknot/depth/000001.bin

# Compare results
compare: build-test
    cd test/build && ./compare_results cpp_normal.bin cuda_normal.bin

# === Clean ===

# Clean all builds
clean: clean-cpp clean-cuda clean-test
    rm -rf temp/*.bin temp/*.png

# === Utility ===

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
