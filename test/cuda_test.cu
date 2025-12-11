// CUDA test - outputs normal map for comparison
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"

#define uo 350
#define vo 200
#define fx 1400
#define fy 1400
#define vmax 480
#define umax 640
#define offset 600

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

inline int idivup(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void normal_estimation_bg_median(
    float* nx_dev,
    float* ny_dev,
    float* nz_dev,
    float* Volume_dev,
    normalization_type normalization,
    visualization_type visualization) {

    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int u = blockDim.x * blockIdx.x + threadIdx.x;

    if ((u >= 1) && (u < umax - 1) && (v >= 1) && (v < vmax - 1)) {
        const int idx0 = v * umax + u;

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
                        Volume_dev[vmax * umax * valid_num + idx0] = nz_tmp;
                    }
                }
            }
        }
        Volume_dev[idx0] = valid_num;

        if (valid_num == 1) {
            nz = Volume_dev[vmax * umax + idx0];
        } else if (valid_num == 2) {
            nz = (Volume_dev[vmax * umax + idx0] + Volume_dev[vmax * umax * 2 + idx0]) / 2;
        } else if (valid_num > 2) {
            // Bubble sort for median
            for (int m = 1; m < valid_num; m++) {
                for (int n = 0; n < valid_num - m; n++) {
                    const float nz_0 = Volume_dev[vmax * umax * (n + 1) + idx0];
                    const float nz_1 = Volume_dev[vmax * umax * (n + 2) + idx0];
                    if (nz_0 > nz_1) {
                        Volume_dev[vmax * umax * (n + 1) + idx0] = nz_1;
                        Volume_dev[vmax * umax * (n + 2) + idx0] = nz_0;
                    }
                }
            }
            if (valid_num % 2 == 0) {
                nz = (Volume_dev[vmax * umax * (valid_num / 2) + idx0]
                    + Volume_dev[vmax * umax * (valid_num / 2 + 1) + idx0]) / 2;
            } else {
                nz = Volume_dev[vmax * umax * ((valid_num + 1) / 2) + idx0];
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

void load_data(const char* path, float* X, float* Y, float* Z, float* D) {
    cv::Mat data_mat(cv::Size(umax, vmax), CV_32F);
    std::ifstream bin_file(path, std::ios::binary);
    if (!bin_file) {
        std::cerr << "Failed to open: " << path << std::endl;
        return;
    }
    bin_file.read(reinterpret_cast<char*>(data_mat.data), sizeof(float) * vmax * umax);
    bin_file.close();

    for (int i = 0; i < vmax; i++) {
        for (int j = 0; j < umax; j++) {
            int idx = i * umax + j;
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

void save_normal_image(const char* path, float* nx, float* ny, float* nz) {
    cv::Mat vis(vmax, umax, CV_16UC3);
    for (int i = 0; i < vmax; i++) {
        for (int j = 0; j < umax; j++) {
            int idx = i * umax + j;
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
    const char* data_path = "../matlab_code/torusknot/depth/000001.bin";
    if (argc > 1) {
        data_path = argv[1];
    }

    std::cout << "Loading depth data from: " << data_path << std::endl;

    normalization_type normalization = POS;
    visualization_type visualization = OPEN;

    const int pixel_number = vmax * umax;
    const int float_memsize = sizeof(float) * pixel_number;

    float* D = (float*)calloc(pixel_number, sizeof(float));
    float* Z = (float*)calloc(pixel_number, sizeof(float));
    float* X = (float*)calloc(pixel_number, sizeof(float));
    float* Y = (float*)calloc(pixel_number, sizeof(float));
    float* nx = (float*)calloc(pixel_number, sizeof(float));
    float* ny = (float*)calloc(pixel_number, sizeof(float));
    float* nz = (float*)calloc(pixel_number, sizeof(float));

    load_data(data_path, X, Y, Z, D);

    // Setup CUDA
    dim3 threads = dim3(Block_x, Block_y);
    dim3 blocks = dim3(idivup(umax, threads.x), idivup(vmax, threads.y));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray *X_texture, *Y_texture, *Z_texture, *D_texture;

    cudaMallocArray(&X_texture, &desc, umax, vmax);
    cudaMallocArray(&Y_texture, &desc, umax, vmax);
    cudaMallocArray(&Z_texture, &desc, umax, vmax);
    cudaMallocArray(&D_texture, &desc, umax, vmax);

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

    std::cout << "Running CUDA kernel..." << std::endl;
    normal_estimation_bg_median<<<blocks, threads>>>(
        nx_dev, ny_dev, nz_dev, Volume_dev, normalization, visualization);

    cudaDeviceSynchronize();

    cudaMemcpy(nx, nx_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ny, ny_dev, float_memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(nz, nz_dev, float_memsize, cudaMemcpyDeviceToHost);

    save_normal_bin("cuda_normal.bin", nx, ny, nz, pixel_number);
    save_normal_image("cuda_normal.png", nx, ny, nz);
    std::cout << "Saved CUDA results to cuda_normal.bin and cuda_normal.png" << std::endl;

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
