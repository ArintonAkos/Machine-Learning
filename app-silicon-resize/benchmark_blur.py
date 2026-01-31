import cv2
import time
import os
import glob
import numpy as np

# Settings
INPUT_FOLDER = "dataset"
ITERATIONS = 50
KERNEL_SIZE = (3, 3) # 3x3 Box Blur

def run_benchmark():
    # Collect image files
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not files:
        print("Error: Dataset folder is empty!")
        return

    print(f"Processing: {len(files)} images with OpenCV...")
    print("-" * 50)

    total_time_ms = 0.0
    total_pixels = 0

    for i, filepath in enumerate(files):
        # Load image (OpenCV loads BGR by default, Alpha is tricky)
        # IMREAD_UNCHANGED: loads Alpha if present
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # To be FAIR to the code (which uses RGBA, i.e., 4 channels):
        # If the image is only 3 channels (JPG), convert it to 4-channel BGRA.
        # This ensures it moves exactly the same number of bytes in memory.
        if len(img.shape) == 2: # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Pixel count
        h, w, c = img.shape
        total_pixels += (w * h)

        # --- MEASUREMENT START ---
        start = time.perf_counter()

        # This is the logic of the code: src -> dst, then dst -> dst
        # In OpenCV, this is simpler because blur returns a new image
        
        # 1. iteration
        processed = cv2.blur(img, KERNEL_SIZE)
        
        # Remaining iterations (ping-ponging in memory)
        for _ in range(ITERATIONS - 1):
            processed = cv2.blur(processed, KERNEL_SIZE)

        end = time.perf_counter()
        # --- MEASUREMENT END ---

        total_time_ms += (end - start) * 1000

        # Save first image for verification
        if i == 0:
            os.makedirs("output_python", exist_ok=True)
            cv2.imwrite("output_python/opencv_result.png", processed)
            print("First image saved: output_python/opencv_result.png")

        # Only measure the first 10 images, like in C++
        if i == 9:
            break

    print("-" * 50)
    print(f"RESULTS ({min(len(files), 10)} images):")
    print(f"Total pixels:      {total_pixels / 1_000_000:.4f} Megapixel")
    print(f"OpenCV time:       {total_time_ms:.2f} ms")
    print("-" * 50)

if __name__ == "__main__":
    run_benchmark()