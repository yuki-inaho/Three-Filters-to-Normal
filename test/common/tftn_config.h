#ifndef TFTN_CONFIG_H
#define TFTN_CONFIG_H

#include <string>
#include <fstream>
#include <sstream>

struct CameraParams {
    float fx = 1400.0f;
    float fy = 1380.0f;
    float uo = 350.0f;
    float vo = 200.0f;
    int offset = 600;

    bool loadFromFile(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) return false;
        ifs >> fx >> fy >> uo >> vo;
        return true;
    }
};

enum class GradientKernel { BASIC, SOBEL };
enum class Aggregation { MEAN, MEDIAN };

struct AlgorithmConfig {
    GradientKernel gradient = GradientKernel::BASIC;
    Aggregation aggregation = Aggregation::MEDIAN;
};

#endif
