#include <arm_neon.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// --- STB IMAGE LIBRARY ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

// --- BARRIER CLASS ---
class Barrier {
public:
  explicit Barrier(std::size_t count)
      : threshold(count), count(count), generation(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    auto gen = generation;
    if (--count == 0) {
      generation++;
      count = threshold;
      cond.notify_all();
    } else {
      cond.wait(lock, [this, gen] { return gen != generation; });
    }
  }

private:
  std::mutex mutex;
  std::condition_variable cond;
  std::size_t threshold;
  std::size_t count;
  std::size_t generation;
};

namespace fs = std::filesystem;

// --- SISD BLUR IMPLEMENTATION (Box Blur 3x3) ---
__attribute__((noinline)) void process_blur_sisd(const unsigned char *src,
                                                 unsigned char *dst, int width,
                                                 int height, int channels,
                                                 int iterations) {
  // Copy src buffer to dst buffer
  std::memcpy(dst, src, width * height * channels);
  // Create a temporary buffer to store the blurred image
  std::vector<unsigned char> temp_buf(width * height * channels);

  for (int iter = 0; iter < iterations; iter++) {
    // Copy dst buffer to temp_buf buffer
    std::memcpy(temp_buf.data(), dst, width * height * channels);

    // Iterate over rows in the image, skip first and last row and column
    for (int y = 1; y < height - 1; ++y) {
      // Iterate over columns in the image, skip first and last row and column
      for (int x = 1; x < width - 1; ++x) {
        // Iterate over channels in the image
        for (int c = 0; c < channels; c++) {
          if (c == 3) {
            int idx = (y * width + x) * channels + c;
            dst[idx] = temp_buf[idx];
            continue;
          }

          int sum = 0;

          for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
              int idx = ((y + ky) * width + (x + kx)) * channels + c;
              sum += temp_buf[idx];
            }
          }

          // Calculate the output index
          int out_idx = (y * width + x) * channels + c;
          // Set the output pixel to the average of the 3x3 kernel
          dst[out_idx] = (unsigned char)(sum / 9);
        }
      }
    }
  }
}

// Optimized division by 3 using 16-bit "Doubling High Multiply".
// Accuracy matches the slower 32-bit vmull version (21846 multiplier),
// but runs in a single instruction without splitting.
static inline uint8x8_t divide_by_3_u16(uint16x8_t sum) {
  // Add 1 before the divsion, so it will be equivalent to (sum + 1) / 3
  // This implements the rounding instead of a simple truncation.
  sum = vaddq_u16(sum, vdupq_n_u16(1));

  // 10923 = ceil(65536 / 6).
  // Since the instruction duplicates (2*x*C), this effectively corresponds to a
  // 21846 multiplier.
  int16x8_t multiplier = vdupq_n_s16(10923);

  // The input is between 0-765, so it can be safely handled as a signed int.
  int16x8_t sum_s16 = vreinterpretq_s16_u16(sum);

  // Instruction: (2 * sum * 10923) >> 16
  // This operation returns the upper 16 bits, with saturation.
  int16x8_t res_s16 = vqdmulhq_s16(sum_s16, multiplier);

  // Convert back and narrow to 8 bits (Unsigned Narrowing)
  return vqmovun_s16(res_s16);
}

static inline uint8x16_t average_3_rows_u8(uint8x16_t top, uint8x16_t mid,
                                           uint8x16_t bot) {
  // Extend to 16 bits and add together (Lower half)
  uint16x8_t sum_low = vaddl_u8(vget_low_u8(top), vget_low_u8(mid));
  sum_low = vaddw_u8(sum_low, vget_low_u8(bot));

  // Extend to 16 bits and add together (Upper half)
  uint16x8_t sum_high = vaddl_u8(vget_high_u8(top), vget_high_u8(mid));
  sum_high = vaddw_u8(sum_high, vget_high_u8(bot));

  // Divide by 3
  uint8x8_t res_low = divide_by_3_u16(sum_low);
  uint8x8_t res_high = divide_by_3_u16(sum_high);

  // Visszaalakítás 8 bites vektorrá
  return vcombine_u8(res_low, res_high);
}

__attribute((noinline)) void
process_blur_simd_vertical_range(const unsigned char *src, unsigned char *dst,
                                 int width, int height, int channels,
                                 int y_start, int y_end) {
  int stride = width * channels;
  // Alpha data and mask
  uint8_t a_data[16] = {0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255};
  uint8x16_t alpha_mask = vld1q_u8(a_data);

  // 1. Handle first row (if in range)
  if (y_start == 0) {
    const unsigned char *p_curr = src;         // 0. row
    const unsigned char *p_bot = src + stride; // 1. row
    unsigned char *p_out = dst;

    for (int x = 0; x < stride; x += channels) {
      for (int c = 0; c < 3; c++) { // Only RGB
        // (Current + Current + Bottom) / 3
        int sum = p_curr[x + c] + p_curr[x + c] + p_bot[x + c];
        p_out[x + c] = sum / 3;
      }
      p_out[x + 3] = 255; // Alpha fix
    }
  }

  // 2. Handle inner rows
  // Intersection of [y_start, y_end) and [1, height-1)
  int inner_start = std::max(1, y_start);
  int inner_end = std::min(height - 1, y_end);

  if (inner_start < inner_end) {
    for (int y = inner_start; y < inner_end; y++) {
      // Get pointers to the previous, current, and next rows
      const unsigned char *p_top = src + (y - 1) * stride;
      const unsigned char *p_mid = src + y * stride;
      const unsigned char *p_bot = src + (y + 1) * stride;
      unsigned char *p_out = dst + y * stride;

      int x = 0;
      for (; x <= stride - 64; x += 64) {
        // Get the top, middle, and bottom rows, 16 pixels per row, 4 channels
        // each
        uint8x16x4_t top = vld4q_u8(p_top + x);
        uint8x16x4_t mid = vld4q_u8(p_mid + x);
        uint8x16x4_t bot = vld4q_u8(p_bot + x);

        uint8x16_t res_r =
            average_3_rows_u8(top.val[0], mid.val[0], bot.val[0]);
        uint8x16_t res_g =
            average_3_rows_u8(top.val[1], mid.val[1], bot.val[1]);
        uint8x16_t res_b =
            average_3_rows_u8(top.val[2], mid.val[2], bot.val[2]);
        uint8x16_t res_a = vdupq_n_u8(255);

        uint8x16x4_t res = {res_r, res_g, res_b, res_a};
        vst4q_u8(p_out + x, res);
      }

      // Smaller Cleanup Loop
      for (; x <= stride - 16; x += 16) {
        uint8x16_t top = vld1q_u8(p_top + x);
        uint8x16_t mid = vld1q_u8(p_mid + x);
        uint8x16_t bot = vld1q_u8(p_bot + x);

        uint16x8_t sum_low = vaddl_u8(vget_low_u8(top), vget_low_u8(mid));
        sum_low = vaddw_u8(sum_low, vget_low_u8(bot));
        uint16x8_t sum_high = vaddl_u8(vget_high_u8(top), vget_high_u8(mid));
        sum_high = vaddw_u8(sum_high, vget_high_u8(bot));

        uint8x16_t res =
            vcombine_u8(divide_by_3_u16(sum_low), divide_by_3_u16(sum_high));
        vst1q_u8(p_out + x, vorrq_u8(res, alpha_mask));
      }

      // Handle the remaining pixels to the right
      for (; x < stride; x += channels) {
        for (int c = 0; c < 3; c++) { // Only RGB
          // (Top + Current + Bottom) / 3
          int sum = p_top[x + c] + p_mid[x + c] + p_bot[x + c];
          p_out[x + c] = sum / 3;
        }
        p_out[x + 3] = 255; // Alpha fix
      }
    }
  }

  // 3. Handle last row (if in range)
  if (y_end == height) {
    const unsigned char *p_curr = src + (height - 1) * stride; // (N-1)th row
    const unsigned char *p_top = src + (height - 2) * stride;  // (N-2)th row
    unsigned char *p_out = dst + (height - 1) * stride;

    for (int x = 0; x < stride; x += channels) {
      for (int c = 0; c < 3; c++) { // Only RGB
        // (Top + Current + Current) / 3
        int sum = p_top[x + c] + p_curr[x + c] + p_curr[x + c];
        p_out[x + c] = sum / 3;
      }
      p_out[x + 3] = 255; // Alpha fix
    }
  }
}

__attribute((noinline)) void
process_blur_simd_horizontal_range(const unsigned char *src, unsigned char *dst,
                                   int width, int height, int channels,
                                   int y_start, int y_end) {
  int stride = width * channels;
  // Alpha data and mask
  uint8_t a_data[16] = {0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255};
  uint8x16_t alpha_mask = vld1q_u8(a_data);

  for (int y = y_start; y < y_end; y++) {
    // The ide aon the horizontal blur is the following:
    // 1. Load the left, middle and right pixels
    const unsigned char *p_src = src + y * stride;
    unsigned char *p_dst = dst + y * stride;

    // Left Side (Pixel 0)
    {
      int x = 0;
      for (int c = 0; c < 3; c++) {
        int sum = p_src[x + c] + p_src[x + c] + p_src[x + c + 4];
        p_dst[x + c] = sum / 3;
      }
      p_dst[x + 3] = 255;
    }

    for (int x = 4; x < 16; x += 4) {
      for (int c = 0; c < 3; c++) {
        int sum = p_src[x - 4 + c] + p_src[x + c] + p_src[x + 4 + c];
        p_dst[x + c] = sum / 3;
      }
      p_dst[x + 3] = 255;
    }

    int x = 16;
    // First 4 pixels (4 pixels * 4 channels = 16 bytes)
    uint8x16_t prev = vld1q_u8(p_src);
    // Next 4 pixels (4 pixels * 4 channels = 16 bytes)
    uint8x16_t curr = vld1q_u8(p_src + 16);

    for (; x <= stride - 32; x += 16) {
      // Next 4 pixels (4 pixels * 4 channels = 16 bytes)
      uint8x16_t next = vld1q_u8(p_src + x + 16);

      // Skip 12 bytes from prev and take 4 bytes from curr
      uint8x16_t left = vextq_u8(prev, curr, 12);
      // Skip 4 bytes from curr and take 12 bytes from next
      uint8x16_t right = vextq_u8(curr, next, 4);

      uint16x8_t sum_low = vaddl_u8(vget_low_u8(left), vget_low_u8(curr));
      sum_low = vaddw_u8(sum_low, vget_low_u8(right));
      uint16x8_t sum_high = vaddl_u8(vget_high_u8(left), vget_high_u8(curr));
      sum_high = vaddw_u8(sum_high, vget_high_u8(right));

      uint8x8_t res_low = divide_by_3_u16(sum_low);
      uint8x8_t res_high = divide_by_3_u16(sum_high);

      uint8x16_t res = vcombine_u8(res_low, res_high);
      // Apply alpha mask
      res = vorrq_u8(res, alpha_mask);
      // Write the result to the output buffer
      vst1q_u8(p_dst + x, res);

      // Go to the next window
      // Window size is 4 pixels (4 pixels * 4 channels = 16 bytes)
      prev = curr;
      curr = next;
    }

    for (; x < stride - 4; x += 4) {
      for (int c = 0; c < 3; c++) { // Only RGB, leave Alpha alone
        int sum = p_src[x - 4 + c] + p_src[x + c] + p_src[x + 4 + c];
        p_dst[x + c] = sum / 3;
      }
      p_dst[x + 3] = 255; // Alpha fix
    }

    // Process last pixel
    {
      int last_x = stride - 4; // Last pixel
      for (int c = 0; c < 3; c++) {
        // (Left + Current + Current) / 3
        int sum = p_src[last_x - 4 + c] + p_src[last_x + c] + p_src[last_x + c];
        p_dst[last_x + c] = sum / 3;
      }
      p_dst[last_x + 3] = 255;
    }
  }
}

__attribute__((noinline)) void process_blur_simd(const unsigned char *src,
                                                 unsigned char *dst, int width,
                                                 int height, int channels,
                                                 int iterations) {
  std::vector<unsigned char> temp_buf(width * height * channels);

  // Initial Pass
  process_blur_simd_vertical_range(src, temp_buf.data(), width, height,
                                   channels, 0, height);
  process_blur_simd_horizontal_range(temp_buf.data(), dst, width, height,
                                     channels, 0, height);

  for (int iter = 1; iter < iterations; iter++) {
    process_blur_simd_vertical_range(dst, temp_buf.data(), width, height,
                                     channels, 0, height);
    process_blur_simd_horizontal_range(temp_buf.data(), dst, width, height,
                                       channels, 0, height);
  }
}

// --- MULTI-THREADED WRAPPER ---
void process_blur_multithreaded(const unsigned char *src, unsigned char *dst,
                                int width, int height, int channels,
                                int num_threads) {
  std::vector<std::thread> threads;
  Barrier barrier(num_threads);

  // Temporary buffer for the whole image (shared across threads)
  std::vector<unsigned char> temp_buf(width * height * channels);

  // Range calculation per thread
  int rows_per_thread = height / num_threads;
  int remainder = height % num_threads;
  int current_y = 0;

  for (int t = 0; t < num_threads; ++t) {
    int start_y = current_y;
    int end_y = start_y + rows_per_thread + (t < remainder ? 1 : 0);
    current_y = end_y;

    threads.emplace_back([=, &barrier, &temp_buf]() {
      int iterations = 50; // Hardcoded or passed as arg if we change sig

      // Iteration 0: src -> temp -> dst
      process_blur_simd_vertical_range(src, temp_buf.data(), width, height,
                                       channels, start_y, end_y);
      barrier.wait(); // Sync execution before Horizontal pass

      process_blur_simd_horizontal_range(temp_buf.data(), dst, width, height,
                                         channels, start_y, end_y);
      barrier.wait(); // Sync execution before next iteration

      // Iterations 1..N: dst -> temp -> dst
      for (int iter = 1; iter < iterations; ++iter) {
        process_blur_simd_vertical_range(dst, temp_buf.data(), width, height,
                                         channels, start_y, end_y);
        barrier.wait();

        process_blur_simd_horizontal_range(temp_buf.data(), dst, width, height,
                                           channels, start_y, end_y);
        barrier.wait();
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}

void process_images_batch_mt(const std::vector<std::string> &files,
                             const std::string &output_folder,
                             int num_threads) {
  std::vector<std::thread> threads;
  std::atomic<int> completed_count{0};
  int total_files = files.size();

  // Thread worker function
  auto worker = [&](int start_idx, int end_idx) {
    for (int i = start_idx; i < end_idx; ++i) {
      int w, h, c;
      // Load image
      unsigned char *raw_img = stbi_load(files[i].c_str(), &w, &h, &c, 4);
      if (!raw_img)
        continue;

      std::vector<unsigned char> src_data(raw_img, raw_img + w * h * 4);
      std::vector<unsigned char> dst_data(w * h * 4);
      stbi_image_free(raw_img);

      // Call the SIMD blur function
      process_blur_simd(src_data.data(), dst_data.data(), w, h, 4, 50);

      // (Optional) Save the image, can be disabled for pure CPU comparison
      // stbi_write_png((output_folder + "/mt_" + filename).c_str(), w, h, 4,
      // dst_data.data(), w * 4);

      completed_count++;
    }
  };

  // Split files between threads
  int files_per_thread = total_files / num_threads;
  int remainder = total_files % num_threads;
  int current_idx = 0;

  for (int t = 0; t < num_threads; ++t) {
    int count = files_per_thread + (t < remainder ? 1 : 0);
    if (count > 0) {
      threads.emplace_back(worker, current_idx, current_idx + count);
      current_idx += count;
    }
  }

  // Wait for threads to finish
  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }
}

int main() {
  std::string input_folder = "dataset";
  // std::string input_folder = "test_images"; // If you want to test with
  // less images
  std::string output_folder = "output_images_blur";

  // Folder checks
  if (!fs::exists(input_folder)) {
    std::cerr << "ERROR: The folder '" << input_folder << "' does not exist!"
              << std::endl;
    return 1;
  }
  if (!fs::exists(output_folder)) {
    fs::create_directory(output_folder);
  }

  // Image file collection
  std::vector<std::string> image_files;
  for (const auto &entry : fs::directory_iterator(input_folder)) {
    std::string ext = entry.path().extension().string();

    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".JPG") {
      image_files.push_back(entry.path().string());
    }
  }

  if (image_files.empty()) {
    std::cerr << "ERROR: The dataset folder is empty!" << std::endl;
    return 1;
  }

  const int ITERATIONS = 50;
  const int SAMPLE_SIZE = 110;
  // If process files is not empty, use it. Otherwise, use the first SAMPLE_SIZE
  // images from the dataset.
  std::vector<std::string> process_files = {"image_109.png"};

  std::vector<std::string> batch_files;
  int limit = std::min(SAMPLE_SIZE, static_cast<int>(image_files.size()));
  for (int i = 0; i < limit; ++i) {
    batch_files.push_back(image_files[i]);
  }

  std::cout << "Processing: " << batch_files.size() << " images..."
            << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  double total_sisd_time = 0.0;
  double total_simd_time = 0.0;
  double total_mt_time = 0.0;
  double total_mt_simd_time = 0.0;
  long total_pixels = 0;

  int count = 0;
  for (const auto &filepath : batch_files) {
    if (!process_files.empty() &&
        filepath.find(process_files[0]) == std::string::npos) {
      continue;
    }

    std::cout << "Processing: " << filepath << std::endl;

    int w, h, channels;
    // Enforce 4 channels (RGBA) for easier alignment (padding)
    unsigned char *raw_img = stbi_load(filepath.c_str(), &w, &h, &channels, 4);
    int proc_channels = 4;

    if (!raw_img) {
      std::cerr << "ERROR: Failed to load image: " << filepath << std::endl;
      continue;
    }

    total_pixels += (w * h);

    // Copy the image data to a vector for easier processing
    std::vector<unsigned char> src_data(raw_img,
                                        raw_img + (w * h * proc_channels));
    stbi_image_free(raw_img);

    // Output buffers (initialize to 0 or the original values for the edges)
    // Initialize with the original values for the edges
    std::vector<unsigned char> dst_sisd = src_data;
    std::vector<unsigned char> dst_simd = src_data;
    std::vector<unsigned char> dst_mt = src_data;

    // --- MEASUREMENT 1: SISD ---
    auto start = std::chrono::high_resolution_clock::now();
    // process_blur_sisd(src_data.data(), dst_sisd.data(), w, h, proc_channels,
    //                   ITERATIONS);
    auto end = std::chrono::high_resolution_clock::now();
    total_sisd_time +=
        std::chrono::duration<double, std::milli>(end - start).count();

    // --- MEASUREMENT 2: SIMD ---
    // Currently empty, or fallback, but measure the call
    start = std::chrono::high_resolution_clock::now();
    process_blur_simd(src_data.data(), dst_simd.data(), w, h, proc_channels,
                      ITERATIONS);
    end = std::chrono::high_resolution_clock::now();
    total_simd_time +=
        std::chrono::duration<double, std::milli>(end - start).count();

    // --- MEASUREMENT 3: MULTI-THREADED ---
    // Currently empty, or fallback, but measure the call
    start = std::chrono::high_resolution_clock::now();
    process_blur_multithreaded(src_data.data(), dst_mt.data(), w, h,
                               proc_channels, 4); // 4 threads
    end = std::chrono::high_resolution_clock::now();
    total_mt_time +=
        std::chrono::duration<double, std::milli>(end - start).count();

    // Save the first image for verification (SISD result)
    if (filepath.find("109") != std::string::npos) {
      std::string out_path = output_folder + "/blur_result_sisd.png";
      // stbi_write_png(out_path.c_str(), w, h, proc_channels, dst_sisd.data(),
      //                w * proc_channels);
      // std::cout << "First image saved (SISD): " << out_path << std::endl;
      out_path = output_folder + "/blur_result_simd.png";
      stbi_write_png(out_path.c_str(), w, h, proc_channels, dst_simd.data(),
                     w * proc_channels);
      std::cout << "First image saved (SIMD): " << out_path << std::endl;
    }

    count++;
  }

  // --- MEASUREMENT 4: MULTI-THREADED SIMD ---
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "MEASUREMENT: MULTI-THREADED SIMD (Batch of "
            << batch_files.size() << ")" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  int num_threads = std::thread::hardware_concurrency();
  process_images_batch_mt(batch_files, output_folder, num_threads);

  auto end = std::chrono::high_resolution_clock::now();
  total_mt_simd_time +=
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "RESULTS (" << count << " images):" << std::endl;
  std::cout << "Total pixels:            " << total_pixels / 1000000.0
            << " Megapixel" << std::endl;
  std::cout << "SISD time:               " << total_sisd_time << " ms"
            << std::endl;
  std::cout << "SIMD time:               " << total_simd_time << " ms"
            << std::endl;
  std::cout << "MT SIMD time (STRIPPED): " << total_mt_time << " ms"
            << std::endl;
  std::cout << "MT SIMD time (BATCH):    " << total_mt_simd_time << " ms"
            << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  if (total_simd_time > 1.0 && total_mt_simd_time > 1.0) {
    std::cout << "SPEEDUP (SISD -> SIMD):      "
              << total_sisd_time / total_simd_time << "x" << std::endl;
    std::cout << "SPEEDUP (SIMD -> MT SIMD):   "
              << total_simd_time / total_mt_simd_time << "x" << std::endl;
    std::cout << "SPEEDUP (SIMD -> MT STRIPPED):   "
              << total_simd_time / total_mt_time << "x" << std::endl;
    std::cout << "TOTAL SPEEDUP (SISD -> MT):  "
              << total_sisd_time / total_mt_simd_time << "x" << std::endl;
    std::cout << "TOTAL SPEEDUP (SISD -> MT STRIPPED):  "
              << total_sisd_time / total_mt_time << "x" << std::endl;
  }

  return 0;
}
