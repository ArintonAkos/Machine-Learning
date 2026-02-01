import ctypes
import numpy as np
import cv2
import time
import os
import glob
import concurrent.futures
import fast_blur_neon  # A TE MODULOD

# --- 1. vImage CTYPES WRAPPER (Az Apple titkos fegyvere) ---
# Betöltjük a rendszerszintű Accelerate keretrendszert
libaccelerate = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Accelerate")

# vImage Buffer struktúra definíciója (ezt várja a C kód)
class vImage_Buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("height", ctypes.c_ulong),
        ("width", ctypes.c_ulong),
        ("rowBytes", ctypes.c_ulong)
    ]

# A Box Blur függvény definíciója
# vImageBoxConvolve_ARGB8888(src, dst, temp, x_radius, y_radius, flags)
vImageBoxConvolve = libaccelerate.vImageBoxConvolve_ARGB8888
vImageBoxConvolve.argtypes = [
    ctypes.POINTER(vImage_Buffer), # src
    ctypes.POINTER(vImage_Buffer), # dst
    ctypes.c_void_p,               # temp buffer
    ctypes.c_uint32,               # x_radius (kernel_size // 2)
    ctypes.c_uint32,               # y_radius
    ctypes.c_uint32,               # kernel_height (csak 0)
    ctypes.c_uint32,               # kernel_width (csak 0)
    ctypes.c_void_p,               # background color
    ctypes.c_uint32                # flags (kvImageEdgeExtend = 1)
]

def run_vimage_blur(img, iterations):
    h, w, c = img.shape
    if c != 4: return # Csak RGBA

    # Adat előkészítése
    # Fontos: A vImage "in-place" nem mindig szeret dolgozni, de megpróbáljuk
    # Létrehozunk egy másolatot az outputnak
    src_data = img.tobytes() # Copy to safe buffer
    dst_data =  ctypes.create_string_buffer(src_data, len(src_data))
    
    # Pufferek definiálása
    src_buf = vImage_Buffer(ctypes.cast(dst_data, ctypes.c_void_p), h, w, w * 4)
    dst_buf = vImage_Buffer(ctypes.cast(dst_data, ctypes.c_void_p), h, w, w * 4) # Ugyanaz a buffer (in-place)

    # Kernel Radius (Box Blur 3x3 -> Radius = 1)
    radius = 1 
    flags = 1 # kvImageEdgeExtend

    # Iterációk
    # A vImage Box Blur egy menetben fut. A te "Stack Blur" hatásodhoz
    # többször kell meghívni egymás után.
    for _ in range(iterations):
        # Temp buffer méretének lekérdezése (opcionális, de biztonságosabb lenne)
        # Most NULL-t adunk át, a vImage majd allokál ha kell (kicsit lassíthat)
        vImageBoxConvolve(ctypes.byref(src_buf), ctypes.byref(dst_buf), None, 
                          0, 0, radius, radius, None, flags)

    return

# --- BENCHMARK BEÁLLÍTÁSOK ---
INPUT_FOLDER = "dataset"
SAMPLE_SIZE = 109
ITERATIONS = 50  # A "Real-Time" beállítás

def load_images():
    print("Loading images...")
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
        
        # A vImage nagyon kényes a "stride"-ra (sorok hossza byte-ban).
        # Biztosítjuk, hogy a memória folytonos legyen.
        img = np.ascontiguousarray(img)
        images.append(img)
    print(f"Loaded {len(images)} images.")
    return images

def benchmark_neon_mt(images):
    print(f"--- Benchmark: YOUR NEON (Multi-Thread) ---")
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(executor.map(lambda img: fast_blur_neon.blur(img, ITERATIONS), images))
    end = time.perf_counter()
    return (end - start) * 1000

def benchmark_vimage_mt(images):
    print(f"--- Benchmark: APPLE vIMAGE (Multi-Thread) ---")
    # A vImage alapból single-threaded (egy képen), 
    # de mivel mi sok képet dolgozunk fel, Pythonból párhuzamosítjuk.
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(executor.map(lambda img: run_vimage_blur(img, ITERATIONS), images))
    end = time.perf_counter()
    return (end - start) * 1000

if __name__ == "__main__":
    images = load_images()
    if not images: exit()

    # 1. TE KÓDOD
    neon_time = benchmark_neon_mt(images)
    print(f"Your NEON Time: {neon_time:.2f} ms")

    # 2. APPLE vIMAGE
    try:
        vimage_time = benchmark_vimage_mt(images)
        print(f"Apple vImage Time: {vimage_time:.2f} ms")
        
        print("-" * 50)
        if neon_time < vimage_time:
             print(f"🏆 NYERTÉL! {vimage_time / neon_time:.2f}x gyorsabb vagy a natív Apple kódnál!")
        else:
             print(f"🥈 A vImage gyorsabb {neon_time / vimage_time:.2f}x-szel.")
             print("(De ne feledd: a tied Linuxon is fut, ez meg csak Mac-en!)")

    except Exception as e:
        print(f"vImage hiba: {e}")