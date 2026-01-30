/*
 * Apple Silicon Optimized Image Resizing Benchmark
 *
 * Demonstrates performance difference between scalar and NEON SIMD
 * implementations of image resizing using stb_image_resize2.h.
 *
 * maintainer: Antigravity
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Control SIMD usage via macro
#ifdef STBIR_NO_SIMD
#define STBIR_NO_SIMD
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

// Constants
const int WIDTH = 3840;  // 4K Width
const int HEIGHT = 2160; // 4K Height
const int CHANNELS = 4;  // RGBA
const int ITERATIONS = 50;

// Helper to generate a checkerboard image if input is missing
std::vector<unsigned char> generate_checkerboard(int w, int h, int c) {
  std::cout << "Generating " << w << "x" << h << " checkerboard pattern..."
            << std::endl;
  std::vector<unsigned char> data(w * h * c);
  int checkSize = 64;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int color = ((x / checkSize) + (y / checkSize)) % 2 == 0 ? 255 : 0;
      int idx = (y * w + x) * c;
      data[idx] = (unsigned char)color;     // R
      data[idx + 1] = (unsigned char)color; // G
      data[idx + 2] = (unsigned char)color; // B
      if (c == 4)
        data[idx + 3] = 255; // A
    }
  }
  return data;
}

int main(int argc, char **argv) {
  std::string inputPath = "input.png";
  std::string outputPath = "output_resized.png";

  int w, h, c;
  unsigned char *imgData =
      stbi_load(inputPath.c_str(), &w, &h, &c, CHANNELS); // Force 4 channels

  std::vector<unsigned char> inputBuffer;

  if (imgData) {
    std::cout << "Loaded " << inputPath << " (" << w << "x" << h << ")"
              << std::endl;
    inputBuffer.assign(imgData, imgData + w * h * CHANNELS);
    stbi_image_free(imgData);
  } else {
    std::cout << "Input file not found, generating synthetic image."
              << std::endl;
    w = WIDTH;
    h = HEIGHT;
    c = CHANNELS;
    inputBuffer = generate_checkerboard(w, h, c);
    // Save the generated input for verification
    stbi_write_png("input_generated.png", w, h, c, inputBuffer.data(), w * c);
    std::cout << "Saved input_generated.png" << std::endl;
  }

  // Target dimensions (50% scale)
  int new_w = w / 2;
  int new_h = h / 2;
  std::vector<unsigned char> outputBuffer(new_w * new_h * c);

  std::cout << "Benchmarking resize from " << w << "x" << h << " to " << new_w
            << "x" << new_h << "..." << std::endl;
  std::cout << "SIMD Status: " <<
#ifdef STBIR_NO_SIMD
      "DISABLED (Scalar)"
#else
      "ENABLED (Auto/NEON)"
#endif
            << std::endl;

  auto start_total = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < ITERATIONS; ++i) {
    // Linear downsample filter
    stbir_resize_uint8_linear(inputBuffer.data(), w, h, 0, outputBuffer.data(),
                              new_w, new_h, 0, (stbir_pixel_layout)c);
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end_total - start_total;

  double total_time_ms = duration.count();
  double avg_time_ms = total_time_ms / ITERATIONS;

  std::cout << "======================================" << std::endl;
  std::cout << "Total Time (" << ITERATIONS << " runs): " << std::fixed
            << std::setprecision(2) << total_time_ms << " ms" << std::endl;
  std::cout << "Average Time per run: " << avg_time_ms << " ms" << std::endl;
  std::cout << "======================================" << std::endl;

  // Save result
  if (stbi_write_png(outputPath.c_str(), new_w, new_h, c, outputBuffer.data(),
                     new_w * c)) {
    std::cout << "Saved successfully: " << outputPath << std::endl;
  } else {
    std::cerr << "Failed to save output image!" << std::endl;
  }

  return 0;
}
