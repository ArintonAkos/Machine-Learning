import cv2
import time
import os
import glob
import concurrent.futures

# Settings
INPUT_FOLDER = "dataset"
ITERATIONS = 50
SAMPLE_SIZE = 109
KERNEL_SIZE = (3, 3) # 3x3 Box Blur

def process_single_image(filepath):
    """
    Ez a függvény fut le minden szálon (Thread) külön-külön.
    """
    try:
        # Load image
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return 0, 0 # Error handling

        # Convert to 4 channels (BGRA) to be fair with C++ RGBA
        if len(img.shape) == 2: # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        h, w, c = img.shape
        pixel_count = w * h

        # --- INDIVIDUAL IMAGE PROCESSING START ---
        # Itt nem mérünk időt globálisan, mert párhuzamosan futunk!
        # Csak a munkát végezzük el.
        
        processed = cv2.blur(img, KERNEL_SIZE)
        
        for _ in range(ITERATIONS - 1):
            processed = cv2.blur(processed, KERNEL_SIZE)

        # Save specific image for verification (thread-safe ish because filenames differ)
        if "108.png" in filepath:
            os.makedirs("output_python", exist_ok=True)
            cv2.imwrite("output_python/opencv_mt_result.png", processed)
            print(f"Verified image saved: output_python/opencv_mt_result.png (from thread)")

        return pixel_count, True

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0, False

def run_multithreaded_benchmark():
    # Collect image files
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    # Limit sample size
    files = files[:SAMPLE_SIZE]

    if not files:
        print("Error: Dataset folder is empty!")
        return

    # Determine number of threads (Logical Cores)
    # Az M4-en ez valószínűleg 10 lesz
    num_threads = os.cpu_count() 
    
    print(f"Processing: {len(files)} images with OpenCV Multi-Threading...")
    print(f"Active Threads: {num_threads}")
    print("-" * 50)

    total_pixels = 0
    
    # --- BATCH MEASUREMENT START ---
    # Most a TELJES időt mérjük (Wall Clock Time), nem a szálak idejét összeadva!
    # Ez a lényege a multithreading mérésnek.
    batch_start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Elindítjuk a szálakat minden fájlra
        futures = [executor.submit(process_single_image, f) for f in files]
        
        # Megvárjuk, amíg mindenki végez
        for future in concurrent.futures.as_completed(futures):
            p_count, success = future.result()
            if success:
                total_pixels += p_count

    batch_end_time = time.perf_counter()
    # --- BATCH MEASUREMENT END ---

    total_time_ms = (batch_end_time - batch_start_time) * 1000

    print("-" * 50)
    print(f"RESULTS ({len(files)} images):")
    print(f"Total pixels:      {total_pixels / 1_000_000:.4f} Megapixel")
    print(f"OpenCV MT Time:    {total_time_ms:.2f} ms") # Ez a teljes futási idő
    print("-" * 50)

if __name__ == "__main__":
    run_multithreaded_benchmark()