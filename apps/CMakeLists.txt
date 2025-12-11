cmake_minimum_required(VERSION 3.10)
project(tftn_apps LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV 4 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# bin2png - Convert .bin to uint16 PNG
add_executable(bin2png bin2png.cpp)
target_link_libraries(bin2png ${OpenCV_LIBS})

# tftn CLI tool (CUDA version)
add_executable(tftn tftn.cu)

target_include_directories(tftn PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../c_code/include
    ${CMAKE_SOURCE_DIR}/../c_code/ThirdParty
)

target_link_libraries(tftn
    ${OpenCV_LIBS}
    CUDA::cudart
)

set_target_properties(tftn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: CPU+CUDA version (requires TFTN library)
option(BUILD_CPU_SUPPORT "Build with CPU support using TFTN library" OFF)

if(BUILD_CPU_SUPPORT)
    add_executable(tftn_cpu tftn.cu)

    target_compile_definitions(tftn_cpu PRIVATE USE_CPU)

    target_include_directories(tftn_cpu PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../c_code/include
        ${CMAKE_SOURCE_DIR}/../c_code/ThirdParty
    )

    target_link_libraries(tftn_cpu
        ${OpenCV_LIBS}
        CUDA::cudart
        ${CMAKE_SOURCE_DIR}/../c_code/build/libtftn.a
    )

    set_target_properties(tftn_cpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()
