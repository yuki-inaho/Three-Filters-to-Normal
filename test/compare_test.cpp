// Comparison test: C++ vs CUDA implementation
// Uses the same data and parameters as CUDA

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include "tftn/tftn.h"

// Default parameters (matching CUDA stdafx.h)
struct Config {
    int width = 640;
    int height = 480;
    double fx = 1400.0;
    double fy = 1380.0;  // Corrected from CUDA's 1400
    double uo = 350.0;
    double vo = 200.0;
    int offset = 600;
    std::string input_path = "../matlab_code/torusknot/depth/000001.bin";
    std::string output_prefix = "cpp_normal";
    int method = -1;  // -1 means use kernel/aggregation options
    std::string kernel = "basic";       // "basic" or "sobel"
    std::string aggregation = "median"; // "mean" or "median"
    std::string params_path;            // Optional params.txt path
    bool help = false;
};;

// Convert kernel/aggregation strings to TFTN_METHOD
TFTN_METHOD toTftnMethod(const std::string& kernel, const std::string& agg) {
    if (kernel == "sobel") {
        if (agg == "median") {
            return R_MEDIAN_SOBEL;  // 9
        } else {
            return R_MEANS_SOBEL;   // 8
        }
    } else {  // basic
        if (agg == "median") {
            return R_MEDIAN_STABLE_4_8;  // 6
        } else {
            return R_MEANS_4_8;  // 7
        }
    }
}


// Load camera parameters from params.txt (format: fx fy uo vo [total_frames])
bool loadParamsFromFile(const std::string& path, Config& config) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Warning: Cannot open params file: " << path << std::endl;
        return false;
    }
    double fx, fy, uo, vo;
    if (ifs >> fx >> fy >> uo >> vo) {
        config.fx = fx;
        config.fy = fy;
        config.uo = uo;
        config.vo = vo;
        std::cout << "Loaded params from " << path << ": fx=" << fx
                  << " fy=" << fy << " uo=" << uo << " vo=" << vo << std::endl;
        return true;
    }
    std::cerr << "Warning: Failed to parse params file: " << path << std::endl;
    return false;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "\nOptions:\n"
              << "  -i, --input PATH       Input depth .bin file (default: ../matlab_code/torusknot/depth/000001.bin)\n"
              << "  -o, --output PREFIX    Output file prefix (default: cpp_normal)\n"
              << "  -W, --width N          Image width (default: 640)\n"
              << "  -H, --height N         Image height (default: 480)\n"
              << "  -p, --params PATH      Load camera params from file (format: fx fy uo vo)\n"
              << "  --fx VALUE             Focal length x (default: 1400.0)\n"
              << "  --fy VALUE             Focal length y (default: 1380.0)\n"
              << "  --uo VALUE             Principal point u (default: 350.0)\n"
              << "  --vo VALUE             Principal point v (default: 200.0)\n"
              << "  --offset VALUE         Depth offset multiplier (default: 600)\n"
              << "  -k, --kernel TYPE      Gradient kernel: basic, sobel (default: basic)\n"
              << "  -a, --aggregation TYPE nz aggregation: mean, median (default: median)\n"
              << "  -m, --method N         TFTN method (overrides kernel/aggregation if set)\n"
              << "                         6=R_MEDIAN_STABLE_4_8, 7=R_MEANS_4_8,\n"
              << "                         8=R_MEANS_SOBEL, 9=R_MEDIAN_SOBEL\n"
              << "  -h, --help             Show this help message\n"
              << "\nKernel types:\n"
              << "  basic  - 2-point gradient -> TFTN method 6 (median) or 7 (mean)\n"
              << "  sobel  - Sobel operator -> TFTN method 9 (median) or 8 (mean)\n"
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
            config.fx = std::stod(value);
        } else if (key == "--fy") {
            config.fy = std::stod(value);
        } else if (key == "--uo") {
            config.uo = std::stod(value);
        } else if (key == "--vo") {
            config.vo = std::stod(value);
        } else if (key == "--offset") {
            config.offset = std::stoi(value);
        } else if (key == "-p" || key == "--params") {
            config.params_path = value;
        } else if (key == "-m" || key == "--method") {
            config.method = std::stoi(value);
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
        } else if (arg[0] != '-' && config.input_path == "../matlab_code/torusknot/depth/000001.bin") {
            // Positional argument: treat as input path (backward compatibility)
            config.input_path = arg;
            i--;  // Compensate for the ++i above
        }
    }

    // Load params file if specified (after parsing all args, so explicit --fx etc. can override)
    if (!config.params_path.empty()) {
        loadParamsFromFile(config.params_path, config);
    }

    return config;
}

cv::Mat LoadDepthBin(const std::string& path, int width, int height) {
    cv::Mat mat(cv::Size(width, height), CV_32FC1);
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        std::cerr << "Failed to open: " << path << std::endl;
        return mat;
    }
    fs.read(reinterpret_cast<char*>(mat.data), sizeof(float) * height * width);
    fs.close();
    return mat;
}

void SaveNormalBin(const std::string& path, float* nx, float* ny, float* nz, int size) {
    std::ofstream fs(path, std::ios::binary);
    fs.write(reinterpret_cast<char*>(nx), sizeof(float) * size);
    fs.write(reinterpret_cast<char*>(ny), sizeof(float) * size);
    fs.write(reinterpret_cast<char*>(nz), sizeof(float) * size);
    fs.close();
}

void SaveNormalImage(const std::string& path, float* nx, float* ny, float* nz, int width, int height) {
    cv::Mat vis(height, width, CV_16UC3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            if (!std::isnan(nx[idx]) && !std::isnan(ny[idx]) && !std::isnan(nz[idx])) {
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

    // Determine TFTN method
    TFTN_METHOD tftn_method;
    if (config.method >= 0) {
        // Explicit method specified
        tftn_method = static_cast<TFTN_METHOD>(config.method);
    } else {
        // Use kernel/aggregation to determine method
        tftn_method = toTftnMethod(config.kernel, config.aggregation);
    }

    // Print configuration
    std::cout << "=== C++ TFTN Test ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input:  " << config.input_path << std::endl;
    std::cout << "  Output: " << config.output_prefix << ".bin/.png" << std::endl;
    std::cout << "  Size:   " << config.width << "x" << config.height << std::endl;
    std::cout << "  Camera: fx=" << config.fx << ", fy=" << config.fy
              << ", uo=" << config.uo << ", vo=" << config.vo << std::endl;
    std::cout << "  Offset: " << config.offset << std::endl;
    std::cout << "  Kernel: " << config.kernel << std::endl;
    std::cout << "  Aggregation: " << config.aggregation << std::endl;
    std::cout << "  TFTN Method: " << tftn_method << std::endl;

    // Load depth data
    cv::Mat depth_raw = LoadDepthBin(config.input_path, config.width, config.height);

    // Apply offset (same as CUDA: Z = offset * depth)
    cv::Mat Z_mat = depth_raw * config.offset;

    // Create X, Y, D arrays (same as CUDA logic)
    const int pixel_number = config.height * config.width;
    float* X = new float[pixel_number];
    float* Y = new float[pixel_number];
    float* Z = new float[pixel_number];
    float* D = new float[pixel_number];

    for (int i = 0; i < config.height; i++) {
        for (int j = 0; j < config.width; j++) {
            int idx = i * config.width + j;
            Z[idx] = Z_mat.at<float>(i, j);
            D[idx] = 1.0f / Z[idx];
            X[idx] = Z[idx] * (j + 1 - config.uo) / config.fx;
            Y[idx] = Z[idx] * (i + 1 - config.vo) / config.fy;
        }
    }

    // Create range image for C++ TFTN (3-channel: X, Y, Z)
    cv::Mat range_image(config.height, config.width, CV_32FC3);
    for (int i = 0; i < config.height; i++) {
        for (int j = 0; j < config.width; j++) {
            int idx = i * config.width + j;
            range_image.at<cv::Vec3f>(i, j) = cv::Vec3f(X[idx], Y[idx], Z[idx]);
        }
    }

    // Camera matrix for C++ TFTN
    cv::Matx33d camera(config.fx, 0, config.uo, 0, config.fy, config.vo, 0, 0, 1);

    // Run C++ TFTN
    cv::Mat result;
    std::cout << "Running C++ TFTN (" << config.kernel << " + " << config.aggregation
              << ", method=" << tftn_method << ")..." << std::endl;
    TFTN(range_image, camera, tftn_method, &result);

    // Extract and normalize results
    float* nx_cpp = new float[pixel_number];
    float* ny_cpp = new float[pixel_number];
    float* nz_cpp = new float[pixel_number];

    for (int i = 0; i < config.height; i++) {
        for (int j = 0; j < config.width; j++) {
            int idx = i * config.width + j;
            cv::Vec3f n = result.at<cv::Vec3f>(i, j);
            float mag = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            if (mag > 0) {
                nx_cpp[idx] = n[0] / mag;
                ny_cpp[idx] = n[1] / mag;
                nz_cpp[idx] = n[2] / mag;
            } else {
                nx_cpp[idx] = ny_cpp[idx] = nz_cpp[idx] = 0;
            }
            // Apply visualization transform (same as CUDA: (1+n)/2)
            nx_cpp[idx] = (1 + nx_cpp[idx]) / 2;
            ny_cpp[idx] = (1 + ny_cpp[idx]) / 2;
            nz_cpp[idx] = (1 + nz_cpp[idx]) / 2;
        }
    }

    // Save results
    std::string bin_path = config.output_prefix + ".bin";
    std::string png_path = config.output_prefix + ".png";
    SaveNormalBin(bin_path, nx_cpp, ny_cpp, nz_cpp, pixel_number);
    SaveNormalImage(png_path, nx_cpp, ny_cpp, nz_cpp, config.width, config.height);
    std::cout << "Saved results to " << bin_path << " and " << png_path << std::endl;

    // Cleanup
    delete[] X;
    delete[] Y;
    delete[] Z;
    delete[] D;
    delete[] nx_cpp;
    delete[] ny_cpp;
    delete[] nz_cpp;

    return 0;
}
