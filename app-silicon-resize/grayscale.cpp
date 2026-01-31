#include <arm_neon.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// --- STB IMAGE KÖNYVTÁRAK ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NO_OPT _Pragma("clang loop vectorize(disable)")

namespace fs = std::filesystem;

// --- SEGÉDFÜGGVÉNY: KÉP GENERÁLÁS ---
std::vector<unsigned char> generate_checkerboard(int w, int h) {
  std::vector<unsigned char> data(w * h * 4);
  int checkSize = 64;

  NO_OPT
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      bool is_white = ((x / checkSize) + (y / checkSize)) % 2 == 0;
      int idx = (y * w + x) * 4;
      // Ha fehér: (255, 255, 255), Ha fekete: (0, 0, 0)
      // Tegyünk bele egy kis színt is, hogy lássuk a grayscale működését!
      if (is_white) {
        data[idx] = 255;     // R
        data[idx + 1] = 100; // G (kicsit zöldes)
        data[idx + 2] = 100; // B
      } else {
        data[idx] = 50;     // R
        data[idx + 1] = 0;  // G
        data[idx + 2] = 50; // B
      }
      data[idx + 3] = 255; // Alpha
    }
  }
  return data;
}

__attribute__((noinline)) void process_grayscale_sisd(unsigned char *img,
                                                      int width, int height) {
  size_t num_pixels = width * height;
  size_t i = 0;

  // std::cout << "SISD feldolgozas inditasa..." << std::endl;

  // Process each pixel sequentially
  for (; i < num_pixels; ++i) {
    int idx = i * 4;
    unsigned char r = img[idx];
    unsigned char g = img[idx + 1];
    unsigned char b = img[idx + 2];

    // Sima C++ matek (integerekkel)
    unsigned char y = (r * 77 + g * 150 + b * 29) >> 8;

    img[idx] = y;
    img[idx + 1] = y;
    img[idx + 2] = y;
  }
}

// --- A LÉNYEG: NEON SIMD GRAYSCALE ---
__attribute__((noinline)) void process_grayscale_simd(unsigned char *img,
                                                      size_t num_pixels) {
  size_t i = 0;

  // Súlyok a szorzáshoz (Fixed Point: 0.299 * 256 ~= 77)
  // uint8-ban tároljuk őket, hogy szorozhassunk velük
  uint8x8_t w_r = vdup_n_u8(77);
  uint8x8_t w_g = vdup_n_u8(150);
  uint8x8_t w_b = vdup_n_u8(29);

  // std::cout << "SIMD feldolgozas inditasa..." << std::endl;

  for (; i <= num_pixels - 8; i += 8) {
    uint8x8x4_t rgba = vld4_u8(img + i * 4);

    // Follow the formula: Y = 0.299 * R + 0.587 * G + 0.114 * B
    // 77 * R + 150 * G + 29 * B
    uint16x8_t sum = vmull_u8(rgba.val[0], w_r);
    sum = vmlal_u8(sum, rgba.val[1], w_g);
    sum = vmlal_u8(sum, rgba.val[2], w_b);
    // 8 bit shift right -> divide by 256
    uint8x8_t gray = vshrn_n_u16(sum, 8);

    uint8x8x4_t result;
    result.val[0] = gray;
    result.val[1] = gray;
    result.val[2] = gray;
    result.val[3] = rgba.val[3];
    vst4_u8(img + i * 4, result);
  }

  // CLEANUP LOOP (Ha a pixelszám nem osztható 8-cal)
  for (; i < num_pixels; ++i) {
    int idx = i * 4;
    unsigned char r = img[idx];
    unsigned char g = img[idx + 1];
    unsigned char b = img[idx + 2];

    // Sima C++ matek (integerekkel)
    unsigned char y = (r * 77 + g * 150 + b * 29) >> 8;

    img[idx] = y;
    img[idx + 1] = y;
    img[idx + 2] = y;
  }
}

// --- MULTI-THREADED WRAPPER ---
void process_grayscale_multithreaded(unsigned char *img, int width, int height,
                                     int num_threads) {
  std::vector<std::thread> workers;
  int rows_per_thread = height / num_threads; // Simplified division

  // Create vertical slices of the image for each thread
  for (int t = 0; t < num_threads; ++t) {
    // Calculate the range of rows for this thread
    int start_y = t * rows_per_thread;
    int end_y = (t == num_threads - 1) ? height : (t + 1) * rows_per_thread;

    // Pointer math: Where does the memory slice start?
    // start_ptr = Image start + (Number of rows * Width * 4 bytes)
    unsigned char *slice_start_ptr = img + (start_y * width * 4);

    // How many pixels are in this slice?
    size_t slice_pixels = (end_y - start_y) * width;

    // Start thread (directly calling the SIMD kernel)
    workers.emplace_back(process_grayscale_simd, slice_start_ptr, slice_pixels);
  }

  // Wait for everyone to finish (JOIN)
  for (auto &worker : workers) {
    worker.join();
  }
}

int main() {
  std::string input_folder = "dataset";
  std::string output_folder = "output_images";

  // Check if the dataset folder exists
  if (!fs::exists(input_folder)) {
    std::cerr << "ERROR: Could not find the './dataset' folder!" << std::endl;
    std::cerr << "Please run the Python script first to download the images."
              << std::endl;
    return 1;
  }

  // Create output folder (optional, if you want to save the images)
  if (!fs::exists(output_folder)) {
    fs::create_directory(output_folder);
  }

  // Collect the files
  std::vector<std::string> image_files;
  for (const auto &entry : fs::directory_iterator(input_folder)) {
    // Only consider PNG and JPG files
    std::string ext = entry.path().extension().string();
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
      image_files.push_back(entry.path().string());
    }
  }

  if (image_files.empty()) {
    std::cerr << "The dataset folder is empty!" << std::endl;
    return 1;
  }

  std::cout << "Processing " << image_files.size() << " images..." << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  double total_sisd_time = 0.0;
  double total_simd_time = 0.0;
  double total_simd_multithreaded_time = 0.0;
  long total_pixels = 0;

  // --- BATCH LOOP ---
  int count = 0;
  for (const auto &filepath : image_files) {
    int w, h, channels;

    // 1. Image loading (NOT part of the measurement!)
    // The '4' parameter forces RGBA format, which is critical for SIMD!
    unsigned char *raw_img = stbi_load(filepath.c_str(), &w, &h, &channels, 4);

    if (!raw_img) {
      std::cerr << "Error loading image: " << filepath << std::endl;
      continue;
    }

    total_pixels += (w * h);

    // Create a copy for measurement (so we don't mess with the raw pointer)
    // The vector automatically handles memory management.
    std::vector<unsigned char> input_data(raw_img, raw_img + (w * h * 4));

    // Free the original stbi buffer because we already copied it to vector
    stbi_image_free(raw_img);

    // Create working copies (so both start with a clean slate)
    std::vector<unsigned char> work_sisd = input_data;
    std::vector<unsigned char> work_simd = input_data;

    // --- MEASUREMENT 1: SISD ---
    auto start = std::chrono::high_resolution_clock::now();
    process_grayscale_sisd(work_sisd.data(), w, h);
    auto end = std::chrono::high_resolution_clock::now();
    total_sisd_time +=
        std::chrono::duration<double, std::milli>(end - start).count();

    // --- MEASUREMENT 2: SIMD ---
    start = std::chrono::high_resolution_clock::now();
    process_grayscale_simd(work_simd.data(), w * h);
    end = std::chrono::high_resolution_clock::now();
    total_simd_time +=
        std::chrono::duration<double, std::milli>(end - start).count();

    // --- MEASUREMENT 3: MULTI-THREADED SIMD ---
    start = std::chrono::high_resolution_clock::now();
    process_grayscale_multithreaded(work_simd.data(), w, h, 4);
    end = std::chrono::high_resolution_clock::now();
    total_simd_multithreaded_time +=
        std::chrono::duration<double, std::milli>(end - start).count();

    // Optional: Save the first image to the output folder for verification
    if (count == 0) {
      std::string out_path = output_folder + "/test_result.png";
      stbi_write_png(out_path.c_str(), w, h, 4, work_simd.data(), w * 4);
    }

    count++;
    // Progress bar
    // if (count % 10 == 0)
    // std::cout << count << " images processed..." << std::endl;
  }

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "RESULTS (" << count << " images):" << std::endl;
  std::cout << "Total pixels:      " << total_pixels / 1000000.0 << " Megapixel"
            << std::endl;
  std::cout << "SISD Total time:   " << total_sisd_time << " ms" << std::endl;
  std::cout << "SIMD Total time:   " << total_simd_time << " ms" << std::endl;
  std::cout << "SIMD Multithreaded Total time:   "
            << total_simd_multithreaded_time << " ms" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "SPEEDUP:           " << total_sisd_time / total_simd_time << "x"
            << std::endl;
  std::cout << "SIMD Multithreaded SPEEDUP:           "
            << total_sisd_time / total_simd_multithreaded_time << "x"
            << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  return 0;
}