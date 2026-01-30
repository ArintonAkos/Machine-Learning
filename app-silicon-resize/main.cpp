/*
 * Apple Silicon Optimized Benchmark
 * SISD vs NEON SIMD
 */

#include <arm_neon.h>
#include <chrono>
#include <iostream>
#include <vector>

// Force no-vectorization for the baseline
#define NO_OPT _Pragma("clang loop vectorize(disable)")

void warmup_memory(std::vector<float> &a, std::vector<float> &b) {
  const int N = a.size();
  for (int i = 0; i < N; i += 4096) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }
}

float sisd_benchmark(const std::vector<float> &a, const std::vector<float> &b) {
  const int N = a.size();

  // ==========================================
  // 1. SISD BENCHMARK (Baseline)
  // ==========================================
  // We run this 50 times to give Instruments time to capture samples
  auto start_sisd = std::chrono::high_resolution_clock::now();

  for (int k = 0; k < 50; k++) {
    float sisd_sum = 0.0f;
    {
      NO_OPT // Disable vectorization
          for (int i = 0; i < N; ++i) {
        // Do a heavy operation but don't write to memory!
        float val = (a[i] * b[i]) + (a[i] * 0.5f);
        sisd_sum += val;
      }
    }
    // Trick: Use the result
    if (sisd_sum > 0)
      volatile int x = 0;
  }

  auto end_sisd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_sisd =
      end_sisd - start_sisd;

  std::cout << "SISD Total Time (50 runs): " << duration_sisd.count() << " ms"
            << std::endl;
  std::cout << "SISD Avg Time per run:     " << duration_sisd.count() / 50.0
            << " ms" << std::endl;

  return duration_sisd.count() / 50.0;
}

float simd_benchmark(const std::vector<float> &a, const std::vector<float> &b) {
  const int N = a.size();

  // ==========================================
  // 2. SIMD BENCHMARK (NEON) - UNROLLED 4x
  // ==========================================
  auto start_simd = std::chrono::high_resolution_clock::now();

  for (int k = 0; k < 50; k++) {
    // 4 accumulators, so the CPU pipeline is full!
    float32x4_t v_sum0 = vdupq_n_f32(0.0f);
    float32x4_t v_sum1 = vdupq_n_f32(0.0f);
    float32x4_t v_sum2 = vdupq_n_f32(0.0f);
    float32x4_t v_sum3 = vdupq_n_f32(0.0f);

    float32x4_t v_half = vdupq_n_f32(0.5f);

    int i = 0;
    // Step 1. Load 4 blocks at once (4 vectors * 4 elements)
    for (; i <= N - 16; i += 16) {
      float32x4_t va0 = vld1q_f32(&a[i]);
      float32x4_t vb0 = vld1q_f32(&b[i]);

      float32x4_t va1 = vld1q_f32(&a[i + 4]);
      float32x4_t vb1 = vld1q_f32(&b[i + 4]);

      float32x4_t va2 = vld1q_f32(&a[i + 8]);
      float32x4_t vb2 = vld1q_f32(&b[i + 8]);

      float32x4_t va3 = vld1q_f32(&a[i + 12]);
      float32x4_t vb3 = vld1q_f32(&b[i + 12]);

      // Step 2. Calculation (These can all run in parallel!)
      float32x4_t vc0 = vmulq_f32(va0, vb0);
      vc0 = vfmaq_f32(vc0, va0, v_half);

      float32x4_t vc1 = vmulq_f32(va1, vb1);
      vc1 = vfmaq_f32(vc1, va1, v_half);

      float32x4_t vc2 = vmulq_f32(va2, vb2);
      vc2 = vfmaq_f32(vc2, va2, v_half);

      float32x4_t vc3 = vmulq_f32(va3, vb3);
      vc3 = vfmaq_f32(vc3, va3, v_half);

      // Step 3. Sum (No dependency between rows)
      v_sum0 = vaddq_f32(v_sum0, vc0);
      v_sum1 = vaddq_f32(v_sum1, vc1);
      v_sum2 = vaddq_f32(v_sum2, vc2);
      v_sum3 = vaddq_f32(v_sum3, vc3);
    }

    // Step 4. Sum the 4 partial results
    float32x4_t v_final = vaddq_f32(v_sum0, v_sum1);
    v_final = vaddq_f32(v_final, v_sum2);
    v_final = vaddq_f32(v_final, v_sum3);

    float simd_final_sum = vaddvq_f32(v_final);

    // Step 5. Cleanup loop (remaining 0-15 elements)
    for (; i < N; ++i) {
      simd_final_sum += (a[i] * b[i]) + (a[i] * 0.5f);
    }

    if (simd_final_sum > 0)
      volatile int y = 0;
  }

  auto end_simd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_simd =
      end_simd - start_simd;

  std::cout << "SIMD Total Time (50 runs): " << duration_simd.count() << " ms"
            << std::endl;
  std::cout << "SIMD Avg Time per run:     " << duration_simd.count() / 50.0
            << " ms" << std::endl;

  return duration_simd.count() / 50.0;
}

int main() {
  const int N = 50000000; // 50 Million elements (approx 200MB per vector)

  std::cout << "Initializing memory (" << (N * 4 * 3) / (1024 * 1024)
            << " MB)..." << std::endl;

  // 1. Allocation (This is slow, but we don't want to measure this!)
  std::vector<float> a(N, 1.0f);
  std::vector<float> b(N, 2.0f);

  // Warmup memory (ensure pages are committed)
  warmup_memory(a, b);

  std::cout << "Starting Benchmark..." << std::endl;

  // Call SISD benchmark
  float duration_sisd = sisd_benchmark(a, b);

  // Call SIMD benchmark
  float duration_simd = simd_benchmark(a, b);

  double speedup = duration_sisd / duration_simd;
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Speedup: " << speedup << "x" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  return 0;
}