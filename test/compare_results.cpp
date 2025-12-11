// Compare C++ and CUDA normal estimation results
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

// Default parameters
struct Config {
    int width = 640;
    int height = 480;
    std::string cpp_path = "cpp_normal.bin";
    std::string cuda_path = "cuda_normal.bin";
    double threshold = 0.01;  // threshold for mismatch
    double fail_percent = 10.0;  // fail if mismatch percentage exceeds this
    bool skip_invalid = true;  // skip invalid pixels (0,0,0) or (0.5,0.5,0.5)
    bool help = false;
};

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options] [cpp_file] [cuda_file]\n"
              << "\nOptions:\n"
              << "  -c, --cpp PATH         C++ normal .bin file (default: cpp_normal.bin)\n"
              << "  -g, --cuda PATH        CUDA normal .bin file (default: cuda_normal.bin)\n"
              << "  -W, --width N          Image width (default: 640)\n"
              << "  -H, --height N         Image height (default: 480)\n"
              << "  -t, --threshold VALUE  Difference threshold (default: 0.01)\n"
              << "  -f, --fail-percent N   Fail if mismatch > N% (default: 10.0)\n"
              << "  -h, --help             Show this help message\n"
              << "\nPositional arguments (backward compatible):\n"
              << "  compare_results [cpp_file] [cuda_file]\n"
              << std::endl;
}

Config parse_args(int argc, char** argv) {
    Config config;
    std::vector<std::string> positional;

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
        } else if (arg[0] == '-' && i + 1 < argc && argv[i + 1][0] != '-') {
            value = argv[++i];
        }

        if (key == "-c" || key == "--cpp") {
            config.cpp_path = value;
        } else if (key == "-g" || key == "--cuda") {
            config.cuda_path = value;
        } else if (key == "-W" || key == "--width") {
            config.width = std::stoi(value);
        } else if (key == "-H" || key == "--height") {
            config.height = std::stoi(value);
        } else if (key == "-t" || key == "--threshold") {
            config.threshold = std::stod(value);
        } else if (key == "-f" || key == "--fail-percent") {
            config.fail_percent = std::stod(value);
        } else if (arg[0] != '-') {
            positional.push_back(arg);
        }
    }

    // Handle positional arguments (backward compatibility)
    if (positional.size() >= 1) {
        config.cpp_path = positional[0];
    }
    if (positional.size() >= 2) {
        config.cuda_path = positional[1];
    }

    return config;
}

// Check if a pixel is invalid (boundary pixel)
// C++ uses (0.5, 0.5, 0.5), CUDA uses (0, 0, 0) for invalid pixels
bool isInvalidPixel(float nx, float ny, float nz) {
    const float eps = 0.001f;
    // Check for (0, 0, 0)
    if (std::abs(nx) < eps && std::abs(ny) < eps && std::abs(nz) < eps) return true;
    // Check for (0.5, 0.5, 0.5)
    if (std::abs(nx - 0.5f) < eps && std::abs(ny - 0.5f) < eps && std::abs(nz - 0.5f) < eps) return true;
    return false;
}

void load_normal_bin(const char* path, float* nx, float* ny, float* nz, int pixel_number) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        std::cerr << "Failed to open: " << path << std::endl;
        return;
    }
    fs.read(reinterpret_cast<char*>(nx), sizeof(float) * pixel_number);
    fs.read(reinterpret_cast<char*>(ny), sizeof(float) * pixel_number);
    fs.read(reinterpret_cast<char*>(nz), sizeof(float) * pixel_number);
    fs.close();
}

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);

    if (config.help) {
        print_usage(argv[0]);
        return 0;
    }

    const int pixel_number = config.width * config.height;

    // Print configuration
    std::cout << "=== Normal Map Comparison ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  C++ file:      " << config.cpp_path << std::endl;
    std::cout << "  CUDA file:     " << config.cuda_path << std::endl;
    std::cout << "  Size:          " << config.width << "x" << config.height << std::endl;
    std::cout << "  Threshold:     " << config.threshold << std::endl;
    std::cout << "  Fail percent:  " << config.fail_percent << "%" << std::endl;

    float* nx_cpp = new float[pixel_number];
    float* ny_cpp = new float[pixel_number];
    float* nz_cpp = new float[pixel_number];
    float* nx_cuda = new float[pixel_number];
    float* ny_cuda = new float[pixel_number];
    float* nz_cuda = new float[pixel_number];

    std::cout << "\nLoading C++ results from: " << config.cpp_path << std::endl;
    load_normal_bin(config.cpp_path.c_str(), nx_cpp, ny_cpp, nz_cpp, pixel_number);

    std::cout << "Loading CUDA results from: " << config.cuda_path << std::endl;
    load_normal_bin(config.cuda_path.c_str(), nx_cuda, ny_cuda, nz_cuda, pixel_number);

    // Compare results
    double total_diff_nx = 0, total_diff_ny = 0, total_diff_nz = 0;
    double max_diff_nx = 0, max_diff_ny = 0, max_diff_nz = 0;
    int valid_count = 0;
    int mismatch_count = 0;

    std::vector<double> all_diffs;

    int skipped_invalid = 0;
    for (int i = 0; i < pixel_number; i++) {
        if (std::isnan(nx_cpp[i]) || std::isnan(nx_cuda[i]) ||
            std::isnan(ny_cpp[i]) || std::isnan(ny_cuda[i]) ||
            std::isnan(nz_cpp[i]) || std::isnan(nz_cuda[i])) {
            continue;
        }

        // Skip invalid/boundary pixels if enabled
        if (config.skip_invalid) {
            if (isInvalidPixel(nx_cpp[i], ny_cpp[i], nz_cpp[i]) ||
                isInvalidPixel(nx_cuda[i], ny_cuda[i], nz_cuda[i])) {
                skipped_invalid++;
                continue;
            }
        }

        double diff_nx = std::abs(nx_cpp[i] - nx_cuda[i]);
        double diff_ny = std::abs(ny_cpp[i] - ny_cuda[i]);
        double diff_nz = std::abs(nz_cpp[i] - nz_cuda[i]);
        double total_diff = std::sqrt(diff_nx*diff_nx + diff_ny*diff_ny + diff_nz*diff_nz);

        all_diffs.push_back(total_diff);

        total_diff_nx += diff_nx;
        total_diff_ny += diff_ny;
        total_diff_nz += diff_nz;

        if (diff_nx > max_diff_nx) max_diff_nx = diff_nx;
        if (diff_ny > max_diff_ny) max_diff_ny = diff_ny;
        if (diff_nz > max_diff_nz) max_diff_nz = diff_nz;

        if (total_diff > config.threshold) {
            mismatch_count++;
        }
        valid_count++;
    }

    if (valid_count > 0) {
        double mismatch_percent = 100.0 * mismatch_count / valid_count;

        std::cout << "\n=== Comparison Results ===" << std::endl;
        std::cout << "Valid pixels: " << valid_count << std::endl;
        if (config.skip_invalid && skipped_invalid > 0) {
            std::cout << "Skipped invalid pixels: " << skipped_invalid << std::endl;
        }
        std::cout << "Pixels with diff > " << config.threshold << ": " << mismatch_count
                  << " (" << mismatch_percent << "%)" << std::endl;

        std::cout << "\nMean absolute difference:" << std::endl;
        std::cout << "  nx: " << (total_diff_nx / valid_count) << std::endl;
        std::cout << "  ny: " << (total_diff_ny / valid_count) << std::endl;
        std::cout << "  nz: " << (total_diff_nz / valid_count) << std::endl;

        std::cout << "\nMax absolute difference:" << std::endl;
        std::cout << "  nx: " << max_diff_nx << std::endl;
        std::cout << "  ny: " << max_diff_ny << std::endl;
        std::cout << "  nz: " << max_diff_nz << std::endl;

        // Percentiles
        std::sort(all_diffs.begin(), all_diffs.end());
        int n = all_diffs.size();
        std::cout << "\nTotal diff (L2 norm) percentiles:" << std::endl;
        std::cout << "  50%: " << all_diffs[n * 50 / 100] << std::endl;
        std::cout << "  90%: " << all_diffs[n * 90 / 100] << std::endl;
        std::cout << "  95%: " << all_diffs[n * 95 / 100] << std::endl;
        std::cout << "  99%: " << all_diffs[n * 99 / 100] << std::endl;
        std::cout << "  max: " << all_diffs[n - 1] << std::endl;

        // Sample some differing pixels
        std::cout << "\nSample differences (first 5 with diff > " << config.threshold << "):" << std::endl;
        int shown = 0;
        for (int i = 0; i < pixel_number && shown < 5; i++) {
            if (std::isnan(nx_cpp[i]) || std::isnan(nx_cuda[i])) continue;
            if (config.skip_invalid &&
                (isInvalidPixel(nx_cpp[i], ny_cpp[i], nz_cpp[i]) ||
                 isInvalidPixel(nx_cuda[i], ny_cuda[i], nz_cuda[i]))) continue;
            double diff = std::sqrt(
                std::pow(nx_cpp[i] - nx_cuda[i], 2) +
                std::pow(ny_cpp[i] - ny_cuda[i], 2) +
                std::pow(nz_cpp[i] - nz_cuda[i], 2));
            if (diff > config.threshold) {
                int y = i / config.width;
                int x = i % config.width;
                std::cout << "  Pixel (" << x << "," << y << "): "
                          << "C++=(" << nx_cpp[i] << "," << ny_cpp[i] << "," << nz_cpp[i] << ") "
                          << "CUDA=(" << nx_cuda[i] << "," << ny_cuda[i] << "," << nz_cuda[i] << ") "
                          << "diff=" << diff << std::endl;
                shown++;
            }
        }

        // Result
        if (mismatch_percent > config.fail_percent) {
            std::cout << "\n[FAIL] Mismatch percentage " << mismatch_percent
                      << "% exceeds threshold " << config.fail_percent << "%" << std::endl;
            delete[] nx_cpp; delete[] ny_cpp; delete[] nz_cpp;
            delete[] nx_cuda; delete[] ny_cuda; delete[] nz_cuda;
            return 1;
        } else {
            std::cout << "\n[PASS] Mismatch percentage " << mismatch_percent
                      << "% is within threshold " << config.fail_percent << "%" << std::endl;
        }
    } else {
        std::cout << "No valid pixels to compare!" << std::endl;
    }

    delete[] nx_cpp; delete[] ny_cpp; delete[] nz_cpp;
    delete[] nx_cuda; delete[] ny_cuda; delete[] nz_cuda;

    return 0;
}
