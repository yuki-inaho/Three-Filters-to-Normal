// Comparison test: C++ vs CUDA implementation
// Uses the same data and parameters as CUDA

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include "tftn/tftn.h"

// Same parameters as CUDA stdafx.h
const int umax = 640;
const int vmax = 480;
const double fx = 1400;
const double fy = 1400;  // Note: CUDA uses 1400, but correct value is 1380
const double uo = 350;
const double vo = 200;
const int offset = 600;

cv::Mat LoadDepthBin(const std::string& path) {
    cv::Mat mat(cv::Size(umax, vmax), CV_32FC1);
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        std::cerr << "Failed to open: " << path << std::endl;
        return mat;
    }
    fs.read(reinterpret_cast<char*>(mat.data), sizeof(float) * vmax * umax);
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

void SaveNormalImage(const std::string& path, float* nx, float* ny, float* nz) {
    cv::Mat vis(vmax, umax, CV_16UC3);
    for (int i = 0; i < vmax; i++) {
        for (int j = 0; j < umax; j++) {
            int idx = i * umax + j;
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
    std::string data_path = "../matlab_code/torusknot/depth/000001.bin";
    if (argc > 1) {
        data_path = argv[1];
    }

    std::cout << "Loading depth data from: " << data_path << std::endl;

    // Load depth data (same as CUDA)
    cv::Mat depth_raw = LoadDepthBin(data_path);

    // Apply offset (same as CUDA: Z = offset * depth)
    cv::Mat Z_mat = depth_raw * offset;

    // Create X, Y, D arrays (same as CUDA logic)
    const int pixel_number = vmax * umax;
    float* X = new float[pixel_number];
    float* Y = new float[pixel_number];
    float* Z = new float[pixel_number];
    float* D = new float[pixel_number];

    for (int i = 0; i < vmax; i++) {
        for (int j = 0; j < umax; j++) {
            int idx = i * umax + j;
            Z[idx] = Z_mat.at<float>(i, j);
            D[idx] = 1.0f / Z[idx];
            X[idx] = Z[idx] * (j + 1 - uo) / fx;
            Y[idx] = Z[idx] * (i + 1 - vo) / fy;
        }
    }

    // Create range image for C++ TFTN (3-channel: X, Y, Z)
    cv::Mat range_image(vmax, umax, CV_32FC3);
    for (int i = 0; i < vmax; i++) {
        for (int j = 0; j < umax; j++) {
            int idx = i * umax + j;
            range_image.at<cv::Vec3f>(i, j) = cv::Vec3f(X[idx], Y[idx], Z[idx]);
        }
    }

    // Camera matrix for C++ TFTN
    cv::Matx33d camera(fx, 0, uo, 0, fy, vo, 0, 0, 1);

    // Run C++ TFTN (R_MEANS_4 = basic gradient with mean, closest to CUDA's bg_median)
    cv::Mat result;
    std::cout << "Running C++ TFTN (R_MEDIAN_STABLE_8)..." << std::endl;
    TFTN(range_image, camera, R_MEDIAN_STABLE_8, &result);

    // Extract and normalize results
    float* nx_cpp = new float[pixel_number];
    float* ny_cpp = new float[pixel_number];
    float* nz_cpp = new float[pixel_number];

    for (int i = 0; i < vmax; i++) {
        for (int j = 0; j < umax; j++) {
            int idx = i * umax + j;
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

    // Save C++ results
    SaveNormalBin("cpp_normal.bin", nx_cpp, ny_cpp, nz_cpp, pixel_number);
    SaveNormalImage("cpp_normal.png", nx_cpp, ny_cpp, nz_cpp);
    std::cout << "Saved C++ results to cpp_normal.bin and cpp_normal.png" << std::endl;

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
