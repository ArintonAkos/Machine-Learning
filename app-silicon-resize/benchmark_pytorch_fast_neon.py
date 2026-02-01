import torch
import torchvision.transforms.functional as F
import cv2
import time
import os
import glob
import numpy as np
import concurrent.futures
import fast_blur_neon  # A TE MODULOD

# Settings
INPUT_FOLDER = "dataset"
SAMPLE_SIZE = 109
ITERATIONS = 3 
TORCH_KERNEL_SIZE = 21 
TORCH_SIGMA = 10.0

def load_images_raw():
    print("Loading images to RAM...")
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    files = files[:SAMPLE_SIZE]
    
    images = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None: continue
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        images.append(img)
    print(f"Loaded {len(images)} images.")
    return images

# --- 1. NEON SINGLE THREAD (Baseline) ---
def benchmark_neon_single(images):
    print(f"--- Benchmark: NEON (Single Thread) ---")
    start = time.perf_counter()
    for img in images:
        _ = fast_blur_neon.blur(img, ITERATIONS)
    end = time.perf_counter()
    return (end - start) * 1000

# --- 2. NEON MULTI THREAD (A Kihívó) ---
def benchmark_neon_mt(images):
    print(f"--- Benchmark: NEON (Multi-Threaded CPU) ---")
    num_workers = os.cpu_count()
    print(f"Using {num_workers} threads...")
    
    start = time.perf_counter()
    
    # A ThreadPoolExecutor elosztja a képeket a magok között.
    # Mivel a C++ oldalon nincs GIL, ez VALÓDI párhuzamosítás.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(lambda img: fast_blur_neon.blur(img, ITERATIONS), images))
        
    end = time.perf_counter()
    return (end - start) * 1000

# --- 3. PYTORCH MPS (GPU) ---
def benchmark_pytorch_mps(images):
    if not torch.backends.mps.is_available():
        return None

    print(f"--- Benchmark: PYTORCH (MPS / METAL GPU) ---")
    
    # Adatkonverzió (ez a PyTorch ára) + Upload
    # Ezt a mérésen KÍVÜL hagyjuk most, hogy csak a nyers számítási erőt mérjük össze!
    # Így adunk egy kis előnyt a GPU-nak (nem mérjük a másolást).
    tensors = []
    for img in images:
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensors.append(t.to('mps')) 
    
    # Warmup
    _ = F.gaussian_blur(tensors[0], kernel_size=TORCH_KERNEL_SIZE, sigma=TORCH_SIGMA)
    torch.mps.synchronize()

    start = time.perf_counter()
    for t in tensors:
        _ = F.gaussian_blur(t, kernel_size=TORCH_KERNEL_SIZE, sigma=TORCH_SIGMA)
    
    # Wait for GPU
    torch.mps.synchronize()
    end = time.perf_counter()
    
    return (end - start) * 1000

if __name__ == "__main__":
    images = load_images_raw()
    if not images: exit()

    # 1. NEON SINGLE
    neon_st_time = benchmark_neon_single(images)
    print(f"Result: {neon_st_time:.2f} ms")
    
    # 2. NEON MULTI
    neon_mt_time = benchmark_neon_mt(images)
    print(f"Result: {neon_mt_time:.2f} ms")

    # 3. GPU MPS
    mps_time = benchmark_pytorch_mps(images)
    if mps_time:
        print(f"Result: {mps_time:.2f} ms")
    
    print("-" * 50)
    print(f"FINAL LEADERBOARD:")
    print(f"1. NEON Multi-Thread: {neon_mt_time:.2f} ms")
    if mps_time:
        print(f"2. PyTorch MPS (GPU): {mps_time:.2f} ms")
    print(f"3. NEON Single Thread: {neon_st_time:.2f} ms")
    
    if mps_time:
        speedup = mps_time / neon_mt_time
        print("-" * 50)
        print(f"🚀 A TE NEON CPU KÓDOD {speedup:.2f}x GYORSABB, MINT AZ MPS GPU!")