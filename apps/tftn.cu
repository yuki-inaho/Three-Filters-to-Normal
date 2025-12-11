// tftn - Three-Filters-to-Normal CLI Tool
// Unified interface for CPU and CUDA surface normal estimation
//
// Usage:
//   tftn --input depth.png --output normal.png [options]
//   tftn --input depth.png --show [options]
//
// Input: uint16 PNG depth image (mm units)
// Output: RGB normal map (16-bit PNG)

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "opencv2/opencv.hpp"

#ifdef USE_CPU
#include "tftn/tftn.h"
#endif

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    std::string input_path;
    std::string output_path;
    int width = 640;               // For .bin files
    int height = 480;              // For .bin files
    float fx = 500.0f;
    float fy = 500.0f;
    float uo = 320.0f;
    float vo = 240.0f;
    float depth_scale = 1.0f;      // Depth multiplier (e.g., offset for .bin files)
    std::string device = "cuda";   // "cpu" or "cuda"
    std::string kernel = "basic";  // "basic" or "sobel"
    std::string aggregation = "mean"; // "mean" or "median"
    bool show = false;
    bool help = false;
};

// ============================================================================
// CUDA Kernels
// ============================================================================

#define Block_x 32
#define Block_y 32

texture<float, 2, cudaReadModeElementType> X_tex;
texture<float, 2, cudaReadModeElementType> Y_tex;
texture<float, 2, cudaReadModeElementType> Z_tex;
texture<float, 2, cudaReadModeElementType> D_tex;

enum normalization_type { POS, NEG };
enum visualization_type { OPEN, CLOSE };

inline int idivup(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// Basic gradient + Median aggregation
__global__ void normal_estimation_bg_median(
    float* nx_dev, float* ny_dev, float* nz_dev, float* Volume_dev,
    int width, int height, float fx, float fy,
    normalization_type normalization, visualization_type visualization) {

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
                        Volume_dev[idx0 + pixel_number * valid_num] = nz_tmp;
                        valid_num++;
                    }
                }
            }
        }

        // Median selection
        if (valid_num > 0) {
            for (int i = 0; i < valid_num - 1; i++) {
                for (int j = i + 1; j < valid_num; j++) {
                    if (Volume_dev[idx0 + pixel_number * i] > Volume_dev[idx0 + pixel_number * j]) {
                        float tmp = Volume_dev[idx0 + pixel_number * i];
                        Volume_dev[idx0 + pixel_number * i] = Volume_dev[idx0 + pixel_number * j];
                        Volume_dev[idx0 + pixel_number * j] = tmp;
                    }
                }
            }
            nz = Volume_dev[idx0 + pixel_number * (valid_num / 2)];
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
    float* nx_dev, float* ny_dev, float* nz_dev,
    int width, int height, float fx, float fy,
    normalization_type normalization, visualization_type visualization) {

    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int u = blockDim.x * blockIdx.x + threadIdx.x;

    if ((u >= 1) && (u < width - 1) && (v >= 1) && (v < height - 1)) {
        const int idx0 = v * width + u;

        const float nx = (tex2D(D_tex, u - 1, v) - tex2D(D_tex, u + 1, v)) * fx;
        const float ny = (tex2D(D_tex, u, v - 1) - tex2D(D_tex, u, v + 1)) * fy;

        const float X0 = tex2D(X_tex, u, v);
        const float Y0 = tex2D(Y_tex, u, v);
        const float Z0 = tex2D(Z_tex, u, v);

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
    float* nx_dev, float* ny_dev, float* nz_dev, float* Volume_dev,
    int width, int height, float fx, float fy,
    normalization_type normalization, visualization_type visualization) {

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

                const float Z_d = Z0 - Z1;
                if (Z0 != Z1) {
                    const float nz_tmp = -(nx * (X0 - X1) + ny * (Y0 - Y1)) / Z_d;
                    if (nz_tmp <= 0) {
                        Volume_dev[idx0 + pixel_number * valid_num] = nz_tmp;
                        valid_num++;
                    }
                }
            }
        }

        // Median selection
        if (valid_num > 0) {
            for (int i = 0; i < valid_num - 1; i++) {
                for (int j = i + 1; j < valid_num; j++) {
                    if (Volume_dev[idx0 + pixel_number * i] > Volume_dev[idx0 + pixel_number * j]) {
                        float tmp = Volume_dev[idx0 + pixel_number * i];
                        Volume_dev[idx0 + pixel_number * i] = Volume_dev[idx0 + pixel_number * j];
                        Volume_dev[idx0 + pixel_number * j] = tmp;
                    }
                }
            }
            nz = Volume_dev[idx0 + pixel_number * (valid_num / 2)];
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
    float* nx_dev, float* ny_dev, float* nz_dev,
    int width, int height, float fx, float fy,
    normalization_type normalization, visualization_type visualization) {

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

// ============================================================================
// Utility Functions
// ============================================================================

void print_usage(const char* prog_name) {
    std::cout << "tftn - Three-Filters-to-Normal Surface Normal Estimation\n\n"
              << "Usage: " << prog_name << " --input <depth_image> [options]\n\n"
              << "Input/Output:\n"
              << "  -i, --input PATH       Input depth image [required]\n"
              << "                         Supported: uint16 PNG (mm), float32 .bin\n"
              << "  -o, --output PATH      Output normal map (16-bit PNG)\n"
              << "  -s, --show             Display result (no save)\n\n"
              << "Image Size (for .bin files):\n"
              << "  -W, --width N          Image width (default: 640)\n"
              << "  -H, --height N         Image height (default: 480)\n\n"
              << "Camera Parameters:\n"
              << "  --fx N                 Focal length X (default: 500)\n"
              << "  --fy N                 Focal length Y (default: 500)\n"
              << "  --uo N                 Principal point X (default: 320)\n"
              << "  --vo N                 Principal point Y (default: 240)\n"
              << "  --scale N              Depth scale/offset (default: 1.0)\n\n"
              << "Algorithm:\n"
              << "  -d, --device DEV       Device: 'cpu' or 'cuda' (default: cuda)\n"
              << "  -k, --kernel TYPE      Gradient kernel: 'basic' or 'sobel' (default: basic)\n"
              << "  -a, --aggregation TYPE nz aggregation: 'mean' or 'median' (default: mean)\n\n"
              << "  -h, --help             Show this help message\n\n"
              << "Examples:\n"
              << "  " << prog_name << " -i depth.png -o normal.png --fx 600 --fy 600\n"
              << "  " << prog_name << " -i depth.bin -o normal.png -W 640 -H 480 --scale 600\n"
              << "  " << prog_name << " -i depth.png --show\n"
              << "  " << prog_name << " -i depth.png -o out.png -k sobel -a median\n";
}

Config parse_args(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        std::string key = arg;
        std::string value = "";

        if (i + 1 < argc && argv[i + 1][0] != '-') {
            value = argv[++i];
        }

        if (key == "-h" || key == "--help") {
            config.help = true;
        } else if (key == "-i" || key == "--input") {
            config.input_path = value;
        } else if (key == "-o" || key == "--output") {
            config.output_path = value;
        } else if (key == "-s" || key == "--show") {
            config.show = true;
            i--;  // No value consumed
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
        } else if (key == "--scale") {
            config.depth_scale = std::stof(value);
        } else if (key == "-d" || key == "--device") {
            if (value != "cpu" && value != "cuda") {
                std::cerr << "Error: Invalid device '" << value << "'. Use 'cpu' or 'cuda'." << std::endl;
                exit(1);
            }
            config.device = value;
        } else if (key == "-k" || key == "--kernel") {
            if (value != "basic" && value != "sobel") {
                std::cerr << "Error: Invalid kernel '" << value << "'. Use 'basic' or 'sobel'." << std::endl;
                exit(1);
            }
            config.kernel = value;
        } else if (key == "-a" || key == "--aggregation") {
            if (value != "mean" && value != "median") {
                std::cerr << "Error: Invalid aggregation '" << value << "'. Use 'mean' or 'median'." << std::endl;
                exit(1);
            }
            config.aggregation = value;
        }
    }

    return config;
}

// Load uint16 PNG depth image and compute X, Y, Z, D arrays
bool load_depth_image(const std::string& path, const Config& config,
                      float* X, float* Y, float* Z, float* D, int& width, int& height) {
    cv::Mat depth = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
        std::cerr << "Error: Cannot load depth image: " << path << std::endl;
        return false;
    }

    if (depth.type() != CV_16UC1) {
        std::cerr << "Error: Depth image must be uint16 (CV_16UC1), got type " << depth.type() << std::endl;
        return false;
    }

    width = depth.cols;
    height = depth.rows;

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            int idx = v * width + u;
            float depth_mm = static_cast<float>(depth.at<uint16_t>(v, u)) * config.depth_scale;
            Z[idx] = depth_mm;
            D[idx] = (depth_mm > 0) ? (1.0f / depth_mm) : 0;
            X[idx] = Z[idx] * (u + 1 - config.uo) / config.fx;
            Y[idx] = Z[idx] * (v + 1 - config.vo) / config.fy;
        }
    }

    return true;
}

// Create visualization image from normal components
cv::Mat create_normal_image(float* nx, float* ny, float* nz, int width, int height) {
    cv::Mat vis(height, width, CV_16UC3);
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            int idx = v * width + u;
            if (!std::isnan(nx[idx]) && !std::isnan(ny[idx]) && !std::isnan(nz[idx])) {
                vis.at<cv::Vec3w>(v, u)[0] = static_cast<uint16_t>(nx[idx] * 65535);
                vis.at<cv::Vec3w>(v, u)[1] = static_cast<uint16_t>(ny[idx] * 65535);
                vis.at<cv::Vec3w>(v, u)[2] = static_cast<uint16_t>(nz[idx] * 65535);
            } else {
                vis.at<cv::Vec3w>(v, u) = cv::Vec3w(0, 0, 0);
            }
        }
    }
    return vis;
}

// ============================================================================
// CUDA Processing
// ============================================================================

void process_cuda(const Config& config, float* X, float* Y, float* Z, float* D,
                  float* nx, float* ny, float* nz, int width, int height) {
    const int pixel_number = width * height;
    const int float_memsize = sizeof(float) * pixel_number;

    dim3 threads(Block_x, Block_y);
    dim3 blocks(idivup(width, threads.x), idivup(height, threads.y));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray *X_texture, *Y_texture, *Z_texture, *D_texture;

    cudaMallocArray(&X_texture, &desc, width, height);
    cudaMallocArray(&Y_texture, &desc, width, height);
    cudaMallocArray(&Z_texture, &desc, width, height);
    cudaMallocArray(&D_texture, &desc, width, height);

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

    normalization_type normalization = POS;
    visualization_type visualization = OPEN;

    // Kernel dispatch
    if (config.kernel == "sobel") {
        if (config.aggregation == "median") {
            normal_estimation_sobel_median<<<blocks, threads>>>(
                nx_dev, ny_dev, nz_dev, Volume_dev,
                width, height, config.fx, config.fy,
                normalization, visualization);
        } else {
            normal_estimation_sobel_mean<<<blocks, threads>>>(
                nx_dev, ny_dev, nz_dev,
                width, height, config.fx, config.fy,
                normalization, visualization);
        }
    } else {
        if (config.aggregation == "median") {
            normal_estimation_bg_median<<<blocks, threads>>>(
                nx_dev, ny_dev, nz_dev, Volume_dev,
                width, height, config.fx, config.fy,
                normalization, visualization);
        } else {
            normal_estimation_bg_mean<<<blocks, threads>>>(
                nx_dev, ny_dev, nz_dev,
                width, height, config.fx, config.fy,
                normalization, visualization);
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(nx, nx_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ny, ny_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(nz, nz_dev, float_memsize, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFreeArray(X_texture);
    cudaFreeArray(Y_texture);
    cudaFreeArray(Z_texture);
    cudaFreeArray(D_texture);
    cudaFree(nx_dev);
    cudaFree(ny_dev);
    cudaFree(nz_dev);
    cudaFree(Volume_dev);
}

// ============================================================================
// CPU Processing (using TFTN library)
// ============================================================================

#ifdef USE_CPU
void process_cpu(const Config& config, float* X, float* Y, float* Z, float* D,
                 float* nx, float* ny, float* nz, int width, int height) {
    // Create range image for TFTN
    cv::Mat range_image(height, width, CV_32F);
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            range_image.at<float>(v, u) = Z[v * width + u];
        }
    }

    // Camera parameters
    tftn::camera camera;
    camera.fx = config.fx;
    camera.fy = config.fy;
    camera.uo = config.uo;
    camera.vo = config.vo;

    // Select TFTN method
    TFTN_METHOD method;
    if (config.kernel == "basic") {
        method = (config.aggregation == "median") ? R_MEDIAN_STABLE_4_8 : R_MEANS_4_8;
    } else {
        method = (config.aggregation == "median") ? R_MEDIAN_SOBEL : R_MEANS_SOBEL;
    }

    // Run TFTN
    cv::Mat result;
    TFTN(range_image, camera, method, &result);

    // Extract normal components
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            int idx = v * width + u;
            cv::Vec3f normal = result.at<cv::Vec3f>(v, u);
            nx[idx] = (1 + normal[0]) / 2;
            ny[idx] = (1 + normal[1]) / 2;
            nz[idx] = (1 + normal[2]) / 2;
        }
    }
}
#endif

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);

    if (config.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (config.input_path.empty()) {
        std::cerr << "Error: Input path is required. Use -i or --input." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (!config.show && config.output_path.empty()) {
        std::cerr << "Error: Either --output or --show must be specified." << std::endl;
        return 1;
    }

    #ifndef USE_CPU
    if (config.device == "cpu") {
        std::cerr << "Error: CPU mode not available (built without USE_CPU)." << std::endl;
        return 1;
    }
    #endif

    // Load depth image
    int width, height;
    cv::Mat depth_check = cv::imread(config.input_path, cv::IMREAD_UNCHANGED);
    if (depth_check.empty()) {
        std::cerr << "Error: Cannot load: " << config.input_path << std::endl;
        return 1;
    }
    width = depth_check.cols;
    height = depth_check.rows;

    const int pixel_number = width * height;
    float* X = new float[pixel_number]();
    float* Y = new float[pixel_number]();
    float* Z = new float[pixel_number]();
    float* D = new float[pixel_number]();
    float* nx = new float[pixel_number]();
    float* ny = new float[pixel_number]();
    float* nz = new float[pixel_number]();

    if (!load_depth_image(config.input_path, config, X, Y, Z, D, width, height)) {
        return 1;
    }

    std::cout << "tftn - Surface Normal Estimation" << std::endl;
    std::cout << "  Input:  " << config.input_path << " (" << width << "x" << height << ")" << std::endl;
    std::cout << "  Device: " << config.device << std::endl;
    std::cout << "  Method: " << config.kernel << " + " << config.aggregation << std::endl;
    std::cout << "  Camera: fx=" << config.fx << " fy=" << config.fy
              << " uo=" << config.uo << " vo=" << config.vo << std::endl;

    // Process
    auto start = std::chrono::high_resolution_clock::now();

    if (config.device == "cuda") {
        process_cuda(config, X, Y, Z, D, nx, ny, nz, width, height);
    } else {
        #ifdef USE_CPU
        process_cpu(config, X, Y, Z, D, nx, ny, nz, width, height);
        #endif
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Time:   " << duration.count() << " ms" << std::endl;

    // Create output image
    cv::Mat normal_image = create_normal_image(nx, ny, nz, width, height);

    // Output
    if (config.show) {
        cv::Mat display;
        normal_image.convertTo(display, CV_8UC3, 255.0 / 65535.0);
        cv::imshow("Normal Map", display);
        std::cout << "Press any key to close..." << std::endl;
        cv::waitKey(0);
    }

    if (!config.output_path.empty()) {
        cv::imwrite(config.output_path, normal_image);
        std::cout << "  Output: " << config.output_path << std::endl;
    }

    // Cleanup
    delete[] X;
    delete[] Y;
    delete[] Z;
    delete[] D;
    delete[] nx;
    delete[] ny;
    delete[] nz;

    return 0;
}
