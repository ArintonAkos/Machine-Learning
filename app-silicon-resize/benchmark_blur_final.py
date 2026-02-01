import cv2
import time
import os
import glob
import numpy as np
import concurrent.futures

# ITT IMPORTÁLJUK A TE LEFORDÍTOTT MODULODAT!
import fast_blur_neon 

# Settings
INPUT_FOLDER = "dataset"
ITERATIONS = 50
SAMPLE_SIZE = 109
KERNEL_SIZE = (3, 3)

# Ez a függvény fut le minden egyes szálon (Thread)
def process_file_neon(filepath):
    try:
        # 1. Kép betöltése (OpenCV)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None: return 0
        
        # 2. Konvertálás RGBA-ra (FONTOS: A C++ kód 4 csatornát vár!)
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # 3. A C++ MODUL HÍVÁSA
        # Itt történik a varázslat. A C++ kód lefut ezen a szálon.
        # Mivel a C++ kód most már "Single Threaded", nem akad össze 
        # a többi Python szállal, hanem 100%-on pörgeti az adott magot.
        result = fast_blur_neon.blur(img, ITERATIONS)
        
        return 1 # Success
    except Exception as e:
        print(f"Error: {e}")
        return 0

def run_neon_benchmark_mt(files):
    print(f"Running FAST BLUR NEON Benchmark on {len(files)} images...")
    
    # Megnézzük, hány magunk van (pl. 10 vagy 12)
    num_workers = os.cpu_count()
    print(f"Active Python Threads: {num_workers}")
    print("-" * 50)
    
    start_time = time.perf_counter()
    
    # INDUL A PÁRHUZAMOS FELDOLGOZÁS
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # A map automatikusan szétosztja a fájlokat a szálak között
        results = list(executor.map(process_file_neon, files))
        
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000

if __name__ == "__main__":
    # Fájlok gyűjtése
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    # Csak az első 109 kép (hogy összehasonlítható legyen a korábbival)
    files = files[:SAMPLE_SIZE]

    if not files:
        print("Dataset empty!")
        exit()

    # MÉRÉS
    neon_time = run_neon_benchmark_mt(files)
    
    print("-" * 50)
    print(f"RESULTS ({len(files)} images):")
    print(f"Total Time:      {neon_time:.2f} ms")
    print("-" * 50)