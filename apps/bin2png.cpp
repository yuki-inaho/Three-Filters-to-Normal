// bin2png - Convert float32 .bin depth to uint16 PNG
//
// Usage:
//   bin2png --input depth.bin --output depth.png -W 640 -H 480 --scale 600
//
// The .bin file contains raw float32 values (inverse depth or range).
// Output PNG is uint16 depth, normalized to fit in 0-65535 range.
// A .meta file is also created with the normalization parameters.

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "opencv2/opencv.hpp"

struct Config {
    std::string input_path;
    std::string output_path;
    int width = 640;
    int height = 480;
    float scale = 600.0f;  // Multiplier for depth values (offset)
    bool help = false;
};

void print_usage(const char* prog_name) {
    std::cout << "bin2png - Convert float32 .bin depth to uint16 PNG\n\n"
              << "Usage: " << prog_name << " --input <depth.bin> --output <depth.png> [options]\n\n"
              << "Options:\n"
              << "  -i, --input PATH    Input .bin file (float32 array) [required]\n"
              << "  -o, --output PATH   Output .png file (uint16) [required]\n"
              << "  -W, --width N       Image width (default: 640)\n"
              << "  -H, --height N      Image height (default: 480)\n"
              << "  --scale N           Depth multiplier/offset (default: 600)\n"
              << "  -h, --help          Show this help\n\n"
              << "Output:\n"
              << "  - depth.png: uint16 normalized depth (0-65535)\n"
              << "  - depth.meta: normalization parameters (depth_scale)\n"
              << "    Use: tftn -i depth.png --scale <depth_scale>\n\n"
              << "Example:\n"
              << "  " << prog_name << " -i depth.bin -o depth.png -W 640 -H 480 --scale 600\n";
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
        } else if (key == "-W" || key == "--width") {
            config.width = std::stoi(value);
        } else if (key == "-H" || key == "--height") {
            config.height = std::stoi(value);
        } else if (key == "--scale") {
            config.scale = std::stof(value);
        }
    }

    return config;
}

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);

    if (config.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (config.input_path.empty() || config.output_path.empty()) {
        std::cerr << "Error: Both --input and --output are required.\n";
        print_usage(argv[0]);
        return 1;
    }

    // Read .bin file
    std::ifstream bin_file(config.input_path, std::ios::binary);
    if (!bin_file) {
        std::cerr << "Error: Cannot open: " << config.input_path << std::endl;
        return 1;
    }

    const int pixel_count = config.width * config.height;
    std::vector<float> data(pixel_count);
    bin_file.read(reinterpret_cast<char*>(data.data()), sizeof(float) * pixel_count);
    bin_file.close();

    // Find min/max depth values
    float min_depth = std::numeric_limits<float>::max();
    float max_depth = std::numeric_limits<float>::lowest();

    for (int i = 0; i < pixel_count; i++) {
        float depth = data[i] * config.scale;
        if (depth > 0 && depth < min_depth) min_depth = depth;
        if (depth > max_depth) max_depth = depth;
    }

    // Calculate normalization: depth_normalized = depth / depth_scale
    // We want max_depth to map to 65535, so depth_scale = max_depth / 65535
    float depth_scale = max_depth / 65535.0f;

    // Convert to uint16 PNG with normalization
    cv::Mat depth_png(config.height, config.width, CV_16UC1);

    for (int v = 0; v < config.height; v++) {
        for (int u = 0; u < config.width; u++) {
            int idx = v * config.width + u;
            float depth = data[idx] * config.scale;

            // Normalize to 0-65535
            float normalized = depth / depth_scale;
            if (normalized < 0) normalized = 0;
            if (normalized > 65535) normalized = 65535;

            depth_png.at<uint16_t>(v, u) = static_cast<uint16_t>(normalized);
        }
    }

    // Save PNG
    cv::imwrite(config.output_path, depth_png);

    // Save meta file with normalization parameters
    std::string meta_path = config.output_path;
    size_t dot_pos = meta_path.rfind('.');
    if (dot_pos != std::string::npos) {
        meta_path = meta_path.substr(0, dot_pos) + ".meta";
    } else {
        meta_path += ".meta";
    }

    std::ofstream meta_file(meta_path);
    meta_file << "# Depth normalization parameters\n";
    meta_file << "# Use with tftn: tftn -i " << config.output_path << " --scale " << depth_scale << "\n";
    meta_file << "depth_scale=" << depth_scale << "\n";
    meta_file << "original_min=" << min_depth << "\n";
    meta_file << "original_max=" << max_depth << "\n";
    meta_file << "input_offset=" << config.scale << "\n";
    meta_file.close();

    std::cout << "Converted: " << config.input_path << " -> " << config.output_path << std::endl;
    std::cout << "  Size: " << config.width << "x" << config.height << std::endl;
    std::cout << "  Input offset: " << config.scale << std::endl;
    std::cout << "  Original depth range: " << min_depth << " - " << max_depth << std::endl;
    std::cout << "  Normalization scale: " << depth_scale << std::endl;
    std::cout << "  Meta file: " << meta_path << std::endl;
    std::cout << "\nTo process with tftn:\n";
    std::cout << "  tftn -i " << config.output_path << " --scale " << depth_scale << " ...\n";

    return 0;
}
