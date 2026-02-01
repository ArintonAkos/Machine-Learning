#include <algorithm>
#include <arm_neon.h>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>

namespace py = pybind11;

// Optimized division by 3 using 16-bit "Doubling High Multiply".
// Accuracy matches the slower 32-bit vmull version (21846 multiplier),
// but runs in a single instruction without splitting.
static inline uint8x8_t divide_by_3_u16(uint16x8_t sum) {
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

// --- GAUSSIAN BLUR IMPLEMENTATION (NEON 11-tap) ---

void blur_horizontal_range_neon(const unsigned char *src, unsigned char *dst,
                                int width, int height, int channels,
                                int y_start, int y_end) {
  int stride = width * channels;
  int radius = 5;

  // Constants for kernel 11 weights: [1, 4, 12, 26, 48, 74, 48, 26, 12, 4, 1]
  uint16x8_t k4 = vdupq_n_u16(4);
  uint16x8_t k12 = vdupq_n_u16(12);
  uint16x8_t k26 = vdupq_n_u16(26);
  uint16x8_t k48 = vdupq_n_u16(48);
  uint16x8_t k74 = vdupq_n_u16(74);
  // Use 128 for rounding, bitshift 8
  uint16x8_t kRound = vdupq_n_u16(128);

  // Iterate over the vertical range provided (Y axis)
  for (int y = y_start; y < y_end; ++y) {

    // Calculate pointers to the start of the current row
    const unsigned char *src_row = src + (y * stride);
    unsigned char *dst_row = dst + (y * stride);

    // 1. Handle Left Boundary
    // We cannot read x-2 at x=0, so we just copy the pixel
    for (int x = 0; x < radius; ++x) {
      for (int c = 0; c < channels; ++c) {
        dst_row[x * channels + c] = src_row[x * channels + c];
      }
    }

    // 2. Handle Right Boundary
    // We cannot read x+2 at the edge, so we copy here as well
    for (int x = width - radius; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        dst_row[x * channels + c] = src_row[x * channels + c];
      }
    }

    // 3. Main NEON Pass
    int x_start = radius;
    int x_end = width - radius;
    // Align vector limit to 4 pixels (16 bytes for RGBA)
    int vec_limit = x_end - ((x_end - x_start) % 4);

    for (int x = x_start; x < vec_limit; x += 4) {
      int idx = x * channels;

      uint16x8_t sum_L = kRound;
      uint16x8_t sum_H = kRound;

      {
        // idx - 5*4 = idx - 20
        uint8x16_t p_left = vld1q_u8(src_row + idx - 20);
        uint16x8_t p_right = vld1q_u8(src_row + idx + 20);

        sum_L = vaddq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)));
        sum_H = vaddq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)));
        sum_L = vaddq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)));
        sum_H = vaddq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)));
      }

      {
        // idx - 4*4 = idx - 16
        uint8x16_t p_left = vld1q_u8(src_row + idx - 16);
        uint16x8_t p_right = vld1q_u8(src_row + idx + 16);

        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k4);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k4);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k4);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k4);
      }

      {
        // idx - 3*4 = idx - 12
        uint8x16_t p_left = vld1q_u8(src_row + idx - 12);
        uint16x8_t p_right = vld1q_u8(src_row + idx + 12);

        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k12);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k12);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k12);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k12);
      }

      {
        // idx - 2*4 = idx - 8
        uint8x16_t p_left = vld1q_u8(src_row + idx - 8);
        uint16x8_t p_right = vld1q_u8(src_row + idx + 8);

        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k26);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k26);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k26);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k26);
      }

      {
        // idx - 1*4 = idx - 4
        uint8x16_t p_left = vld1q_u8(src_row + idx - 4);
        uint16x8_t p_right = vld1q_u8(src_row + idx + 4);

        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k48);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k48);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k48);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k48);
      }

      {
        // idx - 0*4 = idx
        uint8x16_t p_center = vld1q_u8(src_row + idx);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_center)), k74);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_center)), k74);
      }

      // Normalize by dividing by 256 (shift right by 8)
      sum_L = vshrq_n_u16(sum_L, 8);
      sum_H = vshrq_n_u16(sum_H, 8);

      vst1q_u8(dst_row + idx,
               vcombine_u8(vqmovn_u16(sum_L), vqmovn_u16(sum_H)));
    }

    // Handle Leftover pixels (Scalar fallback)
    for (int x = vec_limit; x < x_end; ++x) {
      int idx = x * channels;
      for (int c = 0; c < channels; ++c) {
        uint32_t sum = 128;                                          // Rounding
        sum += 1 * (src_row[idx - 20 + c] + src_row[idx + 20 + c]);  // x-5, x+5
        sum += 4 * (src_row[idx - 16 + c] + src_row[idx + 16 + c]);  // x-4, x+4
        sum += 12 * (src_row[idx - 12 + c] + src_row[idx + 12 + c]); // x-3, x+3
        sum += 26 * (src_row[idx - 8 + c] + src_row[idx + 8 + c]);   // x-2, x+2
        sum += 48 * (src_row[idx - 4 + c] + src_row[idx + 4 + c]);   // x-1, x+1
        sum += 74 * src_row[idx + c];                                // x
        dst_row[idx + c] = (unsigned char)(sum >> 8);
      }
    }
  }
}

void blur_vertical_range_row_neon(const unsigned char *src, unsigned char *dst,
                                  int width, int height, int channels,
                                  int y_start, int y_end) {
  int stride = width * channels;
  int radius = 5;

  // Constants for kernel 11 weights: [1, 4, 12, 26, 48, 74, 48, 26, 12, 4, 1]
  uint16x8_t k4 = vdupq_n_u16(4);
  uint16x8_t k12 = vdupq_n_u16(12);
  uint16x8_t k26 = vdupq_n_u16(26);
  uint16x8_t k48 = vdupq_n_u16(48);
  uint16x8_t k74 = vdupq_n_u16(74);
  // Use 128 for rounding, bitshift 8
  uint16x8_t kRound = vdupq_n_u16(128);

  // Safe boundaries
  int safe_start = std::max(radius, y_start);
  int safe_end = std::min(height - radius, y_end);

  // Left boundary
  for (int y = safe_start; y < safe_end; ++y) {
    unsigned char *dst_row = dst + y * stride;
    const unsigned char *src_ptrs[11];

    for (int i = -5; i <= 5; i++) {
      src_ptrs[i + 5] = src + (y + i) * stride;
    }

    int x = 0;
    for (; x <= stride - 16; x += 16) {
      uint16x8_t sum_L = kRound;
      uint16x8_t sum_H = kRound;

      {
        // (y - 5, y + 5) - Weight: 1
        uint8x16_t p_left = vld1q_u8(src_ptrs[0] + x);
        uint8x16_t p_right = vld1q_u8(src_ptrs[10] + x);

        // Addition
        sum_L = vaddq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)));
        sum_H = vaddq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)));
        sum_L = vaddq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)));
        sum_H = vaddq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)));
      }

      {
        // (y - 4, y + 4) - Weight: 4
        uint8x16_t p_left = vld1q_u8(src_ptrs[1] + x);
        uint8x16_t p_right = vld1q_u8(src_ptrs[9] + x);

        // Addition
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k4);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k4);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k4);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k4);
      }

      {
        // (y - 3, y + 3) - Weight: 12
        uint8x16_t p_left = vld1q_u8(src_ptrs[2] + x);
        uint8x16_t p_right = vld1q_u8(src_ptrs[8] + x);

        // Addition
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k12);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k12);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k12);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k12);
      }

      {
        // (y - 2, y + 2) - Weight: 26
        uint8x16_t p_left = vld1q_u8(src_ptrs[3] + x);
        uint8x16_t p_right = vld1q_u8(src_ptrs[7] + x);

        // Addition
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k26);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k26);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k26);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k26);
      }

      {
        // (y - 1, y + 1) - Weight: 48
        uint8x16_t p_left = vld1q_u8(src_ptrs[4] + x);
        uint8x16_t p_right = vld1q_u8(src_ptrs[6] + x);

        // Addition
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_left)), k48);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_left)), k48);
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_right)), k48);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_right)), k48);
      }

      {
        // (y) - Weight: 74
        uint8x16_t p_center = vld1q_u8(src_ptrs[5] + x);

        // Addition
        sum_L = vmlaq_u16(sum_L, vmovl_u8(vget_low_u8(p_center)), k74);
        sum_H = vmlaq_u16(sum_H, vmovl_u8(vget_high_u8(p_center)), k74);
      }

      sum_L = vshrq_n_u16(sum_L, 8);
      sum_H = vshrq_n_u16(sum_H, 8);

      vst1q_u8(dst_row + x, vcombine_u8(vqmovn_u16(sum_L), vqmovn_u16(sum_H)));
    }

    // Leftover pixels
    for (; x < stride; x++) {
      uint32_t sum = 128; // Rounding

      sum += 1 * (src_ptrs[0][x] + src_ptrs[10][x]);
      sum += 4 * (src_ptrs[1][x] + src_ptrs[9][x]);
      sum += 12 * (src_ptrs[2][x] + src_ptrs[8][x]);
      sum += 26 * (src_ptrs[3][x] + src_ptrs[7][x]);
      sum += 48 * (src_ptrs[4][x] + src_ptrs[6][x]);
      sum += 74 * src_ptrs[5][x];
      dst_row[x] = (unsigned char)(sum >> 8);
    }
  }
}

void process_blur_gaussian(const unsigned char *src, unsigned char *dst,
                           int width, int height, int channels, int iterations,
                           int num_threads = -1) {
  // Create temp buffer for horizontal pass
  std::vector<unsigned char> temp(width * height * channels);
  int stride = width * channels;

  // Horizontal pass
  if (num_threads == -1) {
    num_threads = std::thread::hardware_concurrency();
  }
  int rows_per_thread = height / num_threads;

  for (int i = 0; i < iterations; ++i) {
    const unsigned char *current_input = (i == 0) ? src : dst;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
      int y_start = t * rows_per_thread;
      int y_end = (t == num_threads - 1) ? height : y_start + rows_per_thread;

      threads.emplace_back([=, &temp]() {
        blur_horizontal_range_neon(current_input, temp.data(), width, height,
                                   channels, y_start, y_end);
      });
    }

    // Join threads
    for (auto &t : threads) {
      t.join();
    }

    // Clear threads
    threads.clear();

    // // Vertical pass
    for (int t = 0; t < num_threads; ++t) {
      int y_start = t * rows_per_thread;
      int y_end = (t == num_threads - 1) ? height : y_start + rows_per_thread;

      threads.emplace_back([=, &temp]() {
        blur_vertical_range_row_neon(temp.data(), dst, width, height, channels,
                                     y_start, y_end);
      });
    }

    // Join threads
    for (auto &t : threads) {
      t.join();
    }

    // Clear threads
    threads.clear();

    int radius = 5;
    if (height > radius * 2) {
      std::memcpy(dst, temp.data(), stride * radius);
      std::memcpy(dst + stride * (height - radius),
                  temp.data() + stride * (height - radius), stride * radius);
    }
  }
}

// --- 3. PYTHON BINDING ---
py::array_t<uint8_t> blur_image(py::array_t<uint8_t> input_array,
                                int iterations) {
  // Step 1: Request buffer information.
  // The GIL is currently held, allowing safe access to Python/NumPy internal
  // structures.
  py::buffer_info buf_info = input_array.request();

  if (buf_info.ndim != 3 || buf_info.shape[2] != 4) {
    throw std::runtime_error("The image must be in RGBA (4-channel) format!");
  }

  int height = buf_info.shape[0];
  int width = buf_info.shape[1];
  int channels = 4;
  size_t size = width * height * channels;

  auto ptr = static_cast<uint8_t *>(buf_info.ptr);

  // Step 2: Create a thread-safe local copy.
  // We copy the input data into a C++ std::vector. This is necessary because
  // accessing Python-managed memory (the NumPy pointer) without the GIL is
  // unsafe if the Python Garbage Collector or other threads attempt to modify
  // it.
  std::vector<uint8_t> working_buffer(ptr, ptr + size);

  // Step 3: Release the Global Interpreter Lock (GIL).
  // This block allows other Python threads to execute concurrently while
  // the C++ code performs the heavy computation on the local buffer.
  {
    py::gil_scoped_release release;

    // Perform the SIMD blur operation on the local 'working_buffer'.
    // Note: src and dst pointers point to the same buffer for in-place
    // processing as implemented in the SIMD logic.
    process_blur_simd(working_buffer.data(), working_buffer.data(), width,
                      height, channels, iterations);
  }
  // Step 4: The GIL is automatically re-acquired here when the 'release' object
  // goes out of scope.

  // Step 5: Prepare the output.
  // Allocate a new NumPy array for the result.
  auto result = py::array_t<uint8_t>({height, width, channels});
  py::buffer_info res_info = result.request();

  // Copy the processed data from the C++ vector into the new Python array.
  std::memcpy(res_info.ptr, working_buffer.data(), size);

  return result;
}

py::array_t<uint8_t> blur_gaussian(py::array_t<uint8_t> input_array,
                                   int iterations, int threads = -1) {
  py::buffer_info buf_info = input_array.request();

  if (buf_info.ndim != 3 || buf_info.shape[2] != 4) {
    throw std::runtime_error("The image must be in RGBA (4-channel) format!");
  }

  int height = buf_info.shape[0];
  int width = buf_info.shape[1];
  int channels = 4;
  size_t size = width * height * channels;

  auto ptr = static_cast<uint8_t *>(buf_info.ptr);

  // Working Copy
  std::vector<uint8_t> working_buffer(ptr, ptr + size);

  // Create an output buffer as well, since process_blur_gaussian writes to a
  // distinct dst
  std::vector<uint8_t> dst_buffer(size);

  {
    py::gil_scoped_release release;
    process_blur_gaussian(working_buffer.data(), dst_buffer.data(), width,
                          height, channels, iterations, threads);
  }

  auto result = py::array_t<uint8_t>({height, width, channels});
  py::buffer_info res_info = result.request();
  std::memcpy(res_info.ptr, dst_buffer.data(), size);

  return result;
}

PYBIND11_MODULE(fast_blur_neon, m) {
  m.doc() = "Ultra-fast Box Blur using NEON for Apple Silicon";

  // Binding definition.
  // Note: We do NOT use py::call_guard<py::gil_scoped_release>() here.
  // The GIL is managed manually inside the function to ensure thread safety
  // during memory access.
  m.def("blur", &blur_image, "Blur RGBA image", py::arg("image"),
        py::arg("iterations") = 50);
  m.def("blur_gaussian", &blur_gaussian,
        "Gaussian Blur (11-tap weighted) RGBA image", py::arg("image"),
        py::arg("iterations") = 1, py::arg("threads") = -1);
}