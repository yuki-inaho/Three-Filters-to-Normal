// Compare C++ and CUDA normal estimation results
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

const int umax = 640;
const int vmax = 480;
const int pixel_number = umax * vmax;

void load_normal_bin(const char* path, float* nx, float* ny, float* nz) {
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
    const char* cpp_path = "cpp_normal.bin";
    const char* cuda_path = "cuda_normal.bin";

    if (argc >= 3) {
        cpp_path = argv[1];
        cuda_path = argv[2];
    }

    float* nx_cpp = new float[pixel_number];
    float* ny_cpp = new float[pixel_number];
    float* nz_cpp = new float[pixel_number];
    float* nx_cuda = new float[pixel_number];
    float* ny_cuda = new float[pixel_number];
    float* nz_cuda = new float[pixel_number];

    std::cout << "Loading C++ results from: " << cpp_path << std::endl;
    load_normal_bin(cpp_path, nx_cpp, ny_cpp, nz_cpp);

    std::cout << "Loading CUDA results from: " << cuda_path << std::endl;
    load_normal_bin(cuda_path, nx_cuda, ny_cuda, nz_cuda);

    // Compare results
    double total_diff_nx = 0, total_diff_ny = 0, total_diff_nz = 0;
    double max_diff_nx = 0, max_diff_ny = 0, max_diff_nz = 0;
    int valid_count = 0;
    int mismatch_count = 0;

    std::vector<double> all_diffs;

    for (int i = 0; i < pixel_number; i++) {
        if (std::isnan(nx_cpp[i]) || std::isnan(nx_cuda[i]) ||
            std::isnan(ny_cpp[i]) || std::isnan(ny_cuda[i]) ||
            std::isnan(nz_cpp[i]) || std::isnan(nz_cuda[i])) {
            continue;
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

        if (total_diff > 0.01) {  // threshold for mismatch
            mismatch_count++;
        }
        valid_count++;
    }

    if (valid_count > 0) {
        std::cout << "\n=== Comparison Results ===" << std::endl;
        std::cout << "Valid pixels: " << valid_count << std::endl;
        std::cout << "Pixels with diff > 0.01: " << mismatch_count
                  << " (" << (100.0 * mismatch_count / valid_count) << "%)" << std::endl;

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
        std::cout << "\nSample differences (first 5 with diff > 0.01):" << std::endl;
        int shown = 0;
        for (int i = 0; i < pixel_number && shown < 5; i++) {
            if (std::isnan(nx_cpp[i]) || std::isnan(nx_cuda[i])) continue;
            double diff = std::sqrt(
                std::pow(nx_cpp[i] - nx_cuda[i], 2) +
                std::pow(ny_cpp[i] - ny_cuda[i], 2) +
                std::pow(nz_cpp[i] - nz_cuda[i], 2));
            if (diff > 0.01) {
                int y = i / umax;
                int x = i % umax;
                std::cout << "  Pixel (" << x << "," << y << "): "
                          << "C++=(" << nx_cpp[i] << "," << ny_cpp[i] << "," << nz_cpp[i] << ") "
                          << "CUDA=(" << nx_cuda[i] << "," << ny_cuda[i] << "," << nz_cuda[i] << ") "
                          << "diff=" << diff << std::endl;
                shown++;
            }
        }
    } else {
        std::cout << "No valid pixels to compare!" << std::endl;
    }

    delete[] nx_cpp; delete[] ny_cpp; delete[] nz_cpp;
    delete[] nx_cuda; delete[] ny_cuda; delete[] nz_cuda;

    return (mismatch_count > valid_count * 0.1) ? 1 : 0;  // Fail if >10% mismatch
}
