cmake_minimum_required(VERSION 3.10)
project(tftn_comparison_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mfma -msse2 -msse3 -msse4")

add_definitions(-DUSE_SSE_AUTO)
add_definitions(-DSSE_MATHFUN_WITH_CODE)
add_definitions(-mno-avx512f)
add_definitions(-mno-avx512vl)

find_package(OpenCV 4 REQUIRED)
find_package(CUDA)

# C++ test
add_executable(cpp_test
    compare_test.cpp
)

target_include_directories(cpp_test PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../c_code/include
    ${CMAKE_SOURCE_DIR}/../c_code/ThirdParty
)

target_link_libraries(cpp_test
    ${OpenCV_LIBS}
    ${CMAKE_SOURCE_DIR}/../c_code/build/libtftn.a
)

# Compare tool
add_executable(compare_results
    compare_results.cpp
)

# CUDA test (if CUDA found)
if(CUDA_FOUND)
    include(FindCUDA)
    cuda_add_executable(cuda_test cuda_test.cu)
    target_link_libraries(cuda_test ${OpenCV_LIBS})
endif()
