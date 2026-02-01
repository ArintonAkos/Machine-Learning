from PIL import Image, ImageFilter
import time
import os
import glob

# Settings
INPUT_FOLDER = "dataset"
ITERATIONS = 50 
# Radius 1 jelentése: 1 pixel balra, 1 jobbra + középső = 3 pixel széles.
# Tehát Radius 1 == 3x3 Kernel.
RADIUS = 1 

def run_pillow_benchmark():
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    # Ugyanannyi képet használjunk, mint a C++ teszt (109 db)
    files = files[:109]

    if not files:
        print("Error: Dataset empty!")
        return

    print(f"Processing: {len(files)} images with Pillow BoxBlur (Radius={RADIUS}, Iters={ITERATIONS})...")
    print("-" * 50)

    total_time_ms = 0.0
    total_pixels = 0

    for i, filepath in enumerate(files):
        with Image.open(filepath) as img:
            # Konvertálás RGBA-ra, hogy fair legyen a C++ (4 csatornás) kóddal szemben
            img = img.convert("RGBA")
            img.load() 
            
            w, h = img.size
            total_pixels += (w * h)

            start = time.perf_counter()
            
            # --- A LÉNYEG ---
            # Ugyanazt a "Stack Blur" logikát csináljuk:
            # Sokszor futtatjuk a kis Box Blurt.
            processed = img
            for _ in range(ITERATIONS):
                processed = processed.filter(ImageFilter.BoxBlur(RADIUS))
            
            # Force execution (Pillow lusta kiértékelésű lehet)
            processed.load()

            end = time.perf_counter()
            total_time_ms += (end - start) * 1000
        
        # Mentsük el az elsőt ellenőrzésképp
        if i == 0:
            os.makedirs("output_python", exist_ok=True)
            processed.save("output_python/pillow_box_result.png")
            print("First image saved: output_python/pillow_box_result.png")

    print("-" * 50)
    print(f"RESULTS ({len(files)} images):")
    print(f"Total pixels:      {total_pixels / 1_000_000:.4f} Megapixel")
    print(f"Pillow Time:       {total_time_ms:.2f} ms")
    print("-" * 50)

if __name__ == "__main__":
    run_pillow_benchmark()