// CUDA test - outputs normal map for comparison
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"

// Default parameters
struct Config {
    int width = 640;
    int height = 480;
    float fx = 1400.0f;
    float fy = 1380.0f;  // Corrected from original 1400
    float uo = 350.0f;
    float vo = 200.0f;
    int offset = 600;
    std::string input_path = "../matlab_code/torusknot/depth/000001.bin";
    std::string output_prefix = "cuda_normal";
    std::string kernel = "basic";       // "basic" or "sobel"
    std::string aggregation = "median"; // "mean" or "median"
    int iterations = 1;                 // Number of iterations for benchmark
    bool help = false;
};

#define Block_x 32
#define Block_y 32

using namespace std;
using namespace cv;

texture<float, 2, cudaReadModeElementType> X_tex;
texture<float, 2, cudaReadModeElementType> Y_tex;
texture<float, 2, cudaReadModeElementType> Z_tex;
texture<float, 2, cudaReadModeElementType> D_tex;

enum normalization_type { POS, NEG };
enum visualization_type { OPEN, CLOSE };

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "\nOptions:\n"
              << "  -i, --input PATH       Input depth .bin file (default: ../matlab_code/torusknot/depth/000001.bin)\n"
              << "  -o, --output PREFIX    Output file prefix (default: cuda_normal)\n"
              << "  -W, --width N          Image width (default: 640)\n"
              << "  -H, --height N         Image height (default: 480)\n"
              << "  --fx VALUE             Focal length x (default: 1400.0)\n"
              << "  --fy VALUE             Focal length y (default: 1380.0)\n"
              << "  --uo VALUE             Principal point u (default: 350.0)\n"
              << "  --vo VALUE             Principal point v (default: 200.0)\n"
              << "  --offset VALUE         Depth offset multiplier (default: 600)\n"
              << "  -k, --kernel TYPE      Gradient kernel: basic, sobel (default: basic)\n"
              << "  -a, --aggregation TYPE nz aggregation: mean, median (default: median)\n"
              << "  -n, --iterations N     Number of iterations for benchmark (default: 1)\n"
              << "  -h, --help             Show this help message\n"
              << "\nKernel types:\n"
              << "  basic  - 2-point gradient: (D[u-1] - D[u+1]) * f\n"
              << "  sobel  - Sobel operator: weighted 3x3 neighborhood\n"
              << "\nAggregation types:\n"
              << "  mean   - Average of valid nz candidates\n"
              << "  median - Median of valid nz candidates\n"
              << std::endl;
}

Config parse_args(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            config.help = true;
            return config;
        }

        // Handle --key=value format
        std::string key = arg;
        std::string value;
        size_t eq_pos = arg.find('=');
        if (eq_pos != std::string::npos) {
            key = arg.substr(0, eq_pos);
            value = arg.substr(eq_pos + 1);
        } else if (i + 1 < argc && argv[i + 1][0] != '-') {
            value = argv[++i];
        }

        if (key == "-i" || key == "--input") {
            config.input_path = value;
        } else if (key == "-o" || key == "--output") {
            config.output_prefix = value;
        } else if (key == "-W" || key == "--width") {
            config.width = std::stoi(value);
        } else if (key == "-H" || key == "--height") {
            config.height = std::stoi(value);
        } else if (key == "--fx") {
            config.fx = std::stof(value);
        } else if (key == "--fy") {
            config.fy = std::stof(value);
        } else if (key == "--uo") {
            config.uo = std::stof(value);
        } else if (key == "--vo") {
            config.vo = std::stof(value);
        } else if (key == "--offset") {
            config.offset = std::stoi(value);
        } else if (key == "-k" || key == "--kernel") {
            if (value != "basic" && value != "sobel") {
                std::cerr << "Error: Invalid kernel type '" << value << "'. Use 'basic' or 'sobel'." << std::endl;
                exit(1);
            }
            config.kernel = value;
        } else if (key == "-a" || key == "--aggregation") {
            if (value != "mean" && value != "median") {
                std::cerr << "Error: Invalid aggregation type '" << value << "'. Use 'mean' or 'median'." << std::endl;
                exit(1);
            }
            config.aggregation = value;
        } else if (key == "-n" || key == "--iterations") {
            config.iterations = std::stoi(value);
        } else if (arg[0] != '-' && config.input_path == "../matlab_code/torusknot/depth/000001.bin") {
            // Positional argument: treat as input path (backward compatibility)
            config.input_path = arg;
            i--;  // Compensate for the ++i above
        }
    }

    return config;
}

inline int idivup(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void normal_estimation_bg_median(
    float* nx_dev,
    float* ny_dev,
    float* nz_dev,
    float* Volume_dev,
    int width,
    int height,
    float fx,
    float fy,
    normalization_type normalization,
    visualization_type visualization) {

    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int u = blockDim.x * blockIdx.x + threadIdx.x;

    if ((u >= 1) && (u < width - 1) && (v >= 1) && (v < height - 1)) {
        const int idx0 = v * width + u;
        const int pixel_number = width * height;

        const float nx = (tex2D(D_tex, u - 1, v) - tex2D(D_tex, u + 1, v)) * fx;
        const float ny = (tex2D(D_tex, u, v - 1) - tex2D(D_tex, u, v + 1)) * fy;

        nx_dev[idx0] = nx;
        ny_dev[idx0] = ny;

        const float X0 = tex2D(X_tex, u, v);
        const float Y0 = tex2D(Y_tex, u, v);
        const float Z0 = tex2D(Z_tex, u, v);

        float nz = 0;
        int valid_num = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                const float X1 = tex2D(X_tex, u + j, v + i);
                const float Y1 = tex2D(Y_tex, u + j, v + i);
                const float Z1 = tex2D(Z_tex, u + j, v + i);

                const float X_d = X0 - X1;
                const float Y_d = Y0 - Y1;
                const float Z_d = Z0 - Z1;

                if (Z0 != Z1) {
                    const float nz_tmp = -(nx * X_d + ny * Y_d) / Z_d;
                    if (nz_tmp <= 0) {
                        valid_num++;
                        Volume_dev[pixel_number * valid_num + idx0] = nz_tmp;
                    }
                }
            }
        }
        Volume_dev[idx0] = valid_num;

        if (valid_num == 1) {
            nz = Volume_dev[pixel_number + idx0];
        } else if (valid_num == 2) {
            nz = (Volume_dev[pixel_number + idx0] + Volume_dev[pixel_number * 2 + idx0]) / 2;
        } else if (valid_num > 2) {
            // Bubble sort for median
            for (int m = 1; m < valid_num; m++) {
                for (int n = 0; n < valid_num - m; n++) {
                    const float nz_0 = Volume_dev[pixel_number * (n + 1) + idx0];
                    const float nz_1 = Volume_dev[pixel_number * (n + 2) + idx0];
                    if (nz_0 > nz_1) {
                        Volume_dev[pixel_number * (n + 1) + idx0] = nz_1;
                        Volume_dev[pixel_number * (n + 2) + idx0] = nz_0;
                    }
                }
            }
            if (valid_num % 2 == 0) {
                nz = (Volume_dev[pixel_number * (valid_num / 2) + idx0]
                    + Volume_dev[pixel_number * (valid_num / 2 + 1) + idx0]) / 2;
            } else {
                nz = Volume_dev[pixel_number * ((valid_num + 1) / 2) + idx0];
            }
        }

        if (normalization == POS) {
            float mag = sqrt(nx * nx + ny * ny + nz * nz);
            if (mag != 0) {
                nx_dev[idx0] = nx / mag;
                ny_dev[idx0] = ny / mag;
                nz_dev[idx0] = nz / mag;
            }
        }
        if (visualization == OPEN) {
            nx_dev[idx0] = (1 + nx_dev[idx0]) / 2;
            ny_dev[idx0] = (1 + ny_dev[idx0]) / 2;
            nz_dev[idx0] = (1 + nz_dev[idx0]) / 2;
        }
    }
}

// Basic gradient + Mean aggregation
__global__ void normal_estimation_bg_mean(
    float* nx_dev,
    float* ny_dev,
    float* nz_dev,
    int width,
    int height,
    float fx,
    float fy,
    normalization_type normalization,
    visualization_type visualization) {

    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int u = blockDim.x * blockIdx.x + threadIdx.x;

    if ((u >= 1) && (u < width - 1) && (v >= 1) && (v < height - 1)) {
        const int idx0 = v * width + u;

        // Basic 2-point gradient
        const float nx = (tex2D(D_tex, u - 1, v) - tex2D(D_tex, u + 1, v)) * fx;
        const float ny = (tex2D(D_tex, u, v - 1) - tex2D(D_tex, u, v + 1)) * fy;

        const float X0 = tex2D(X_tex, u, v);
        const float Y0 = tex2D(Y_tex, u, v);
        const float Z0 = tex2D(Z_tex, u, v);

        // Mean aggregation
        float nz_sum = 0;
        int valid_num = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                const float X1 = tex2D(X_tex, u + j, v + i);
                const float Y1 = tex2D(Y_tex, u + j, v + i);
                const float Z1 = tex2D(Z_tex, u + j, v + i);

                const float Z_d = Z0 - Z1;
                if (Z0 != Z1) {
                    const float nz_tmp = -(nx * (X0 - X1) + ny * (Y0 - Y1)) / Z_d;
                    if (nz_tmp <= 0) {
                        nz_sum += nz_tmp;
                        valid_num++;
                    }
                }
            }
        }

        float nz = (valid_num > 0) ? (nz_sum / valid_num) : 0;

        if (normalization == POS) {
            float mag = sqrt(nx * nx + ny * ny + nz * nz);
            if (mag != 0) {
                nx_dev[idx0] = nx / mag;
                ny_dev[idx0] = ny / mag;
                nz_dev[idx0] = nz / mag;
            }
        }
        if (visualization == OPEN) {
            nx_dev[idx0] = (1 + nx_dev[idx0]) / 2;
            ny_dev[idx0] = (1 + ny_dev[idx0]) / 2;
            nz_dev[idx0] = (1 + nz_dev[idx0]) / 2;
        }
    }
}

// Sobel gradient + Median aggregation
__global__ void normal_estimation_sobel_median(
    float* nx_dev,
    float* ny_dev,
    float* nz_dev,
    float* Volume_dev,
    int width,
    int height,
    float fx,
    float fy,
    normalization_type normalization,
    visualization_type visualization) {

    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int u = blockDim.x * blockIdx.x + threadIdx.x;

    if ((u >= 1) && (u < width - 1) && (v >= 1) && (v < height - 1)) {
        const int idx0 = v * width + u;
        const int pixel_number = width * height;

        // Sobel gradient
        const float nx = (2 * (tex2D(D_tex, u - 1, v) - tex2D(D_tex, u + 1, v))
            + tex2D(D_tex, u - 1, v - 1) - tex2D(D_tex, u + 1, v - 1)
            + tex2D(D_tex, u - 1, v + 1) - tex2D(D_tex, u + 1, v + 1)) * fx;

        const float ny = (2 * (tex2D(D_tex, u, v - 1) - tex2D(D_tex, u, v + 1))
            + tex2D(D_tex, u - 1, v - 1) - tex2D(D_tex, u - 1, v + 1)
            + tex2D(D_tex, u + 1, v - 1) - tex2D(D_tex, u + 1, v + 1)) * fy;

        nx_dev[idx0] = nx;
        ny_dev[idx0] = ny;

        const float X0 = tex2D(X_tex, u, v);
        const float Y0 = tex2D(Y_tex, u, v);
        const float Z0 = tex2D(Z_tex, u, v);

        float nz = 0;
        int valid_num = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                const float X1 = tex2D(X_tex, u + j, v + i);
                const float Y1 = tex2D(Y_tex, u + j, v + i);
                const float Z1 = tex2D(Z_tex, u + j, v + i);

                const float X_d = X0 - X1;
                const float Y_d = Y0 - Y1;
                const float Z_d = Z0 - Z1;

                if (Z0 != Z1) {
                    const float nz_tmp = -(nx * X_d + ny * Y_d) / Z_d;
                    if (nz_tmp <= 0) {
                        valid_num++;
                        Volume_dev[pixel_number * valid_num + idx0] = nz_tmp;
                    }
                }
            }
        }
        Volume_dev[idx0] = valid_num;

        if (valid_num == 1) {
            nz = Volume_dev[pixel_number + idx0];
        } else if (valid_num == 2) {
            nz = (Volume_dev[pixel_number + idx0] + Volume_dev[pixel_number * 2 + idx0]) / 2;
        } else if (valid_num > 2) {
            // Bubble sort for median
            for (int m = 1; m < valid_num; m++) {
                for (int n = 0; n < valid_num - m; n++) {
                    const float nz_0 = Volume_dev[pixel_number * (n + 1) + idx0];
                    const float nz_1 = Volume_dev[pixel_number * (n + 2) + idx0];
                    if (nz_0 > nz_1) {
                        Volume_dev[pixel_number * (n + 1) + idx0] = nz_1;
                        Volume_dev[pixel_number * (n + 2) + idx0] = nz_0;
                    }
                }
            }
            if (valid_num % 2 == 0) {
                nz = (Volume_dev[pixel_number * (valid_num / 2) + idx0]
                    + Volume_dev[pixel_number * (valid_num / 2 + 1) + idx0]) / 2;
            } else {
                nz = Volume_dev[pixel_number * ((valid_num + 1) / 2) + idx0];
            }
        }

        if (normalization == POS) {
            float mag = sqrt(nx * nx + ny * ny + nz * nz);
            if (mag != 0) {
                nx_dev[idx0] = nx / mag;
                ny_dev[idx0] = ny / mag;
                nz_dev[idx0] = nz / mag;
            }
        }
        if (visualization == OPEN) {
            nx_dev[idx0] = (1 + nx_dev[idx0]) / 2;
            ny_dev[idx0] = (1 + ny_dev[idx0]) / 2;
            nz_dev[idx0] = (1 + nz_dev[idx0]) / 2;
        }
    }
}

// Sobel gradient + Mean aggregation
__global__ void normal_estimation_sobel_mean(
    float* nx_dev,
    float* ny_dev,
    float* nz_dev,
    int width,
    int height,
    float fx,
    float fy,
    normalization_type normalization,
    visualization_type visualization) {

    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int u = blockDim.x * blockIdx.x + threadIdx.x;

    if ((u >= 1) && (u < width - 1) && (v >= 1) && (v < height - 1)) {
        const int idx0 = v * width + u;

        // Sobel gradient
        const float nx = (2 * (tex2D(D_tex, u - 1, v) - tex2D(D_tex, u + 1, v))
            + tex2D(D_tex, u - 1, v - 1) - tex2D(D_tex, u + 1, v - 1)
            + tex2D(D_tex, u - 1, v + 1) - tex2D(D_tex, u + 1, v + 1)) * fx;

        const float ny = (2 * (tex2D(D_tex, u, v - 1) - tex2D(D_tex, u, v + 1))
            + tex2D(D_tex, u - 1, v - 1) - tex2D(D_tex, u - 1, v + 1)
            + tex2D(D_tex, u + 1, v - 1) - tex2D(D_tex, u + 1, v + 1)) * fy;

        const float X0 = tex2D(X_tex, u, v);
        const float Y0 = tex2D(Y_tex, u, v);
        const float Z0 = tex2D(Z_tex, u, v);

        // Mean aggregation
        float nz_sum = 0;
        int valid_num = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                const float X1 = tex2D(X_tex, u + j, v + i);
                const float Y1 = tex2D(Y_tex, u + j, v + i);
                const float Z1 = tex2D(Z_tex, u + j, v + i);

                const float Z_d = Z0 - Z1;
                if (Z0 != Z1) {
                    const float nz_tmp = -(nx * (X0 - X1) + ny * (Y0 - Y1)) / Z_d;
                    if (nz_tmp <= 0) {
                        nz_sum += nz_tmp;
                        valid_num++;
                    }
                }
            }
        }

        float nz = (valid_num > 0) ? (nz_sum / valid_num) : 0;

        if (normalization == POS) {
            float mag = sqrt(nx * nx + ny * ny + nz * nz);
            if (mag != 0) {
                nx_dev[idx0] = nx / mag;
                ny_dev[idx0] = ny / mag;
                nz_dev[idx0] = nz / mag;
            }
        }
        if (visualization == OPEN) {
            nx_dev[idx0] = (1 + nx_dev[idx0]) / 2;
            ny_dev[idx0] = (1 + ny_dev[idx0]) / 2;
            nz_dev[idx0] = (1 + nz_dev[idx0]) / 2;
        }
    }
}

void load_data(const char* path, float* X, float* Y, float* Z, float* D,
               int width, int height, float fx, float fy, float uo, float vo, int offset) {
    cv::Mat data_mat(cv::Size(width, height), CV_32F);
    std::ifstream bin_file(path, std::ios::binary);
    if (!bin_file) {
        std::cerr << "Failed to open: " << path << std::endl;
        return;
    }
    bin_file.read(reinterpret_cast<char*>(data_mat.data), sizeof(float) * height * width);
    bin_file.close();

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            Z[idx] = offset * data_mat.at<float>(i, j);
            D[idx] = 1.0f / Z[idx];
            X[idx] = Z[idx] * (j + 1 - uo) / fx;
            Y[idx] = Z[idx] * (i + 1 - vo) / fy;
        }
    }
}

void save_normal_bin(const char* path, float* nx, float* ny, float* nz, int size) {
    std::ofstream fs(path, std::ios::binary);
    fs.write(reinterpret_cast<char*>(nx), sizeof(float) * size);
    fs.write(reinterpret_cast<char*>(ny), sizeof(float) * size);
    fs.write(reinterpret_cast<char*>(nz), sizeof(float) * size);
    fs.close();
}

void save_normal_image(const char* path, float* nx, float* ny, float* nz, int width, int height) {
    cv::Mat vis(height, width, CV_16UC3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            if (!isnan(nx[idx]) && !isnan(ny[idx]) && !isnan(nz[idx])) {
                vis.at<cv::Vec3w>(i, j)[0] = (unsigned short)(nx[idx] * 65535);
                vis.at<cv::Vec3w>(i, j)[1] = (unsigned short)(ny[idx] * 65535);
                vis.at<cv::Vec3w>(i, j)[2] = (unsigned short)(nz[idx] * 65535);
            } else {
                vis.at<cv::Vec3w>(i, j) = cv::Vec3w(1, 1, 1);
            }
        }
    }
    cv::imwrite(path, vis);
}

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);

    if (config.help) {
        print_usage(argv[0]);
        return 0;
    }

    // Print configuration
    std::cout << "=== CUDA TFTN Test ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input:  " << config.input_path << std::endl;
    std::cout << "  Output: " << config.output_prefix << ".bin/.png" << std::endl;
    std::cout << "  Size:   " << config.width << "x" << config.height << std::endl;
    std::cout << "  Camera: fx=" << config.fx << ", fy=" << config.fy
              << ", uo=" << config.uo << ", vo=" << config.vo << std::endl;
    std::cout << "  Offset: " << config.offset << std::endl;
    std::cout << "  Kernel: " << config.kernel << std::endl;
    std::cout << "  Aggregation: " << config.aggregation << std::endl;

    normalization_type normalization = POS;
    visualization_type visualization = OPEN;

    const int pixel_number = config.height * config.width;
    const int float_memsize = sizeof(float) * pixel_number;

    float* D = (float*)calloc(pixel_number, sizeof(float));
    float* Z = (float*)calloc(pixel_number, sizeof(float));
    float* X = (float*)calloc(pixel_number, sizeof(float));
    float* Y = (float*)calloc(pixel_number, sizeof(float));
    float* nx = (float*)calloc(pixel_number, sizeof(float));
    float* ny = (float*)calloc(pixel_number, sizeof(float));
    float* nz = (float*)calloc(pixel_number, sizeof(float));

    load_data(config.input_path.c_str(), X, Y, Z, D,
              config.width, config.height, config.fx, config.fy,
              config.uo, config.vo, config.offset);

    // Setup CUDA
    dim3 threads = dim3(Block_x, Block_y);
    dim3 blocks = dim3(idivup(config.width, threads.x), idivup(config.height, threads.y));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray *X_texture, *Y_texture, *Z_texture, *D_texture;

    cudaMallocArray(&X_texture, &desc, config.width, config.height);
    cudaMallocArray(&Y_texture, &desc, config.width, config.height);
    cudaMallocArray(&Z_texture, &desc, config.width, config.height);
    cudaMallocArray(&D_texture, &desc, config.width, config.height);

    float *nx_dev, *ny_dev, *nz_dev, *Volume_dev;
    cudaMalloc((void**)&nx_dev, float_memsize);
    cudaMalloc((void**)&ny_dev, float_memsize);
    cudaMalloc((void**)&nz_dev, float_memsize);
    cudaMalloc((void**)&Volume_dev, float_memsize * 9);

    cudaMemcpyToArray(X_texture, 0, 0, X, float_memsize, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(Y_texture, 0, 0, Y, float_memsize, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(Z_texture, 0, 0, Z, float_memsize, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(D_texture, 0, 0, D, float_memsize, cudaMemcpyHostToDevice);

    cudaBindTextureToArray(X_tex, X_texture, desc);
    cudaBindTextureToArray(Y_tex, Y_texture, desc);
    cudaBindTextureToArray(Z_tex, Z_texture, desc);
    cudaBindTextureToArray(D_tex, D_texture, desc);

    // Kernel dispatch based on kernel type and aggregation method
    std::cout << "Running CUDA kernel (" << config.kernel << " + " << config.aggregation
              << ", iterations=" << config.iterations << ")..." << std::endl;

    // Lambda for kernel dispatch
    auto run_kernel = [&]() {
        if (config.kernel == "sobel") {
            if (config.aggregation == "median") {
                normal_estimation_sobel_median<<<blocks, threads>>>(
                    nx_dev, ny_dev, nz_dev, Volume_dev,
                    config.width, config.height, config.fx, config.fy,
                    normalization, visualization);
            } else {  // mean
                normal_estimation_sobel_mean<<<blocks, threads>>>(
                    nx_dev, ny_dev, nz_dev,
                    config.width, config.height, config.fx, config.fy,
                    normalization, visualization);
            }
        } else {  // basic
            if (config.aggregation == "median") {
                normal_estimation_bg_median<<<blocks, threads>>>(
                    nx_dev, ny_dev, nz_dev, Volume_dev,
                    config.width, config.height, config.fx, config.fy,
                    normalization, visualization);
            } else {  // mean
                normal_estimation_bg_mean<<<blocks, threads>>>(
                    nx_dev, ny_dev, nz_dev,
                    config.width, config.height, config.fx, config.fy,
                    normalization, visualization);
            }
        }
        cudaDeviceSynchronize();
    };

    // Warm-up run
    run_kernel();

    // Benchmark runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < config.iterations; iter++) {
        run_kernel();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_ms = duration.count() / 1000.0 / config.iterations;
    std::cout << "  Total time: " << duration.count() / 1000.0 << " ms (" << config.iterations << " iterations)" << std::endl;
    std::cout << "  Average time per iteration: " << avg_ms << " ms" << std::endl;

    cudaMemcpy(nx, nx_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ny, ny_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(nz, nz_dev, float_memsize, cudaMemcpyDeviceToHost);

    std::string bin_path = config.output_prefix + ".bin";
    std::string png_path = config.output_prefix + ".png";
    save_normal_bin(bin_path.c_str(), nx, ny, nz, pixel_number);
    save_normal_image(png_path.c_str(), nx, ny, nz, config.width, config.height);
    std::cout << "Saved results to " << bin_path << " and " << png_path << std::endl;

    // Cleanup
    cudaFreeArray(X_texture);
    cudaFreeArray(Y_texture);
    cudaFreeArray(Z_texture);
    cudaFreeArray(D_texture);
    cudaFree(nx_dev);
    cudaFree(ny_dev);
    cudaFree(nz_dev);
    cudaFree(Volume_dev);
    free(X); free(Y); free(Z); free(D);
    free(nx); free(ny); free(nz);

    return 0;
}
