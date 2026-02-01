#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

// --- OPTIMALIZÁLT SHADER (Textúrákkal) ---
const char *shaderSource = R"(
    #include <metal_stdlib>
    using namespace metal;

    kernel void blur_kernel(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]])
    {
        // Ha kilógunk, kilépünk
        if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
            return;
        }

        // Sampler: Ez a hardveres egység kezeli a széleket helyettünk!
        // clamp_to_edge: Ha kimegyünk a képből, az utolsó pixelt ismétli.
        constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);

        float4 sum = float4(0.0);

        // 3x3 Loop - Most már a textúra cache segít!
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                // A 'sample' helyett 'read'-et használunk coord::pixel esetén, 
                // de manuálisan kezeljük a clamp-et, VAGY egyszerűbb:
                // Mivel írható textúránk van, a sima read() gyorsabb, de boundary check kell.
                // DE: A modern Metalban a read() is gyors.
                // A legegyszerűbb, ha manuálisan clampelünk az indexen, 
                // de a hardveres textúra olvasás akkor is gyorsabb a memórialayout miatt.
                
                int2 coords = int2(gid.x, gid.y) + int2(i, j);
                
                // Hardveres clamp (csak biztos ami biztos, bár a sampler tudná normalized coorddal)
                coords.x = clamp(coords.x, 0, int(inTexture.get_width()) - 1);
                coords.y = clamp(coords.y, 0, int(inTexture.get_height()) - 1);
                
                sum += inTexture.read(uint2(coords));
            }
        }

        outTexture.write(sum / 9.0, gid);
    }
)";

int main() {
  std::string input_folder = "dataset";
  std::string output_folder = "output_images_metal";
  if (!fs::exists(output_folder))
    fs::create_directory(output_folder);

  std::vector<std::string> image_files;
  for (const auto &entry : fs::directory_iterator(input_folder)) {
    std::string ext = entry.path().extension().string();
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".JPG") {
      image_files.push_back(entry.path().string());
    }
  }

  if (image_files.empty())
    return 1;

  const int SAMPLE_SIZE = 10;
  const int ITERATIONS = 50;
  std::vector<std::string> batch_files;
  int limit = std::min((int)image_files.size(), SAMPLE_SIZE);
  for (int i = 0; i < limit; i++)
    batch_files.push_back(image_files[i]);

  std::cout << "Metal TEXTURE Benchmark: " << batch_files.size()
            << " images.\n";
  std::cout << "------------------------------------------------\n";

  // 1. Metal Setup
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> commandQueue = [device newCommandQueue];

  NSError *error = nil;
  NSString *srcStr = [NSString stringWithUTF8String:shaderSource];
  id<MTLLibrary> library = [device newLibraryWithSource:srcStr
                                                options:nil
                                                  error:&error];
  if (!library) {
    std::cerr << "Shader Error: " << [[error localizedDescription] UTF8String]
              << "\n";
    return 1;
  }

  id<MTLFunction> blurFunc = [library newFunctionWithName:@"blur_kernel"];
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:blurFunc error:&error];

  double total_gpu_time = 0.0;
  long total_pixels = 0;
  int count = 0;

  for (const auto &filepath : batch_files) {
    int w, h, c;
    unsigned char *raw_img = stbi_load(filepath.c_str(), &w, &h, &c, 4);
    if (!raw_img)
      continue;

    total_pixels += (w * h);

    // --- A LÉNYEG: TEXTÚRÁK LÉTREHOZÁSA ---
    MTLTextureDescriptor *textureDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:w
                                    height:h
                                 mipmapped:NO];
    // Fontos: Engedélyezzük, hogy a shader írja és olvassa is
    textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

    // Textúra A és B (Ping-Pong)
    id<MTLTexture> textureA = [device newTextureWithDescriptor:textureDesc];
    id<MTLTexture> textureB = [device newTextureWithDescriptor:textureDesc];

    // Adatok feltöltése a Texture A-ba
    MTLRegion region = MTLRegionMake2D(0, 0, w, h);
    [textureA replaceRegion:region
                mipmapLevel:0
                  withBytes:raw_img
                bytesPerRow:w * 4];

    stbi_image_free(raw_img);

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pso];

    // Ping-Pong Loop
    for (int i = 0; i < ITERATIONS; ++i) {
      id<MTLTexture> input = (i % 2 == 0) ? textureA : textureB;
      id<MTLTexture> output = (i % 2 == 0) ? textureB : textureA;

      [encoder setTexture:input atIndex:0];
      [encoder setTexture:output atIndex:1];

      // Szálak
      MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
      MTLSize gridSize = MTLSizeMake(w, h, 1);
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    }

    [encoder endEncoding];

    auto start = std::chrono::high_resolution_clock::now();

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    auto end = std::chrono::high_resolution_clock::now();
    total_gpu_time +=
        std::chrono::duration<double, std::milli>(end - start).count();

    // Eredmény letöltése (csak az elsőnél)
    if (count == 0) {
      std::vector<unsigned char> resultData(w * h * 4);
      // Utolsó írás 'bufferA'-ba történt (ha 50 iteráció: páratlan kör (49) ír
      // A-ba)
      [textureA getBytes:resultData.data()
             bytesPerRow:w * 4
              fromRegion:region
             mipmapLevel:0];

      stbi_write_png("metal_texture_result.png", w, h, 4, resultData.data(),
                     w * 4);
    }
    count++;
  }

  std::cout << "------------------------------------------------\n";
  std::cout << "RESULTS (" << count << " images):\n";
  std::cout << "Total pixels: " << total_pixels / 1000000.0 << " Megapixel\n";
  std::cout << "Metal GPU time: " << total_gpu_time << " ms\n";
  std::cout << "------------------------------------------------\n";

  return 0;
}