import os
import requests
import time

# Beállítások
NUM_IMAGES = 200          # Mennyi képet akarsz? (Kezdésnek 20 elég 4K-ban!)
WIDTH = 3840             # 4K szélesség
HEIGHT = 2160            # 4K magasság
OUTPUT_DIR = "dataset"   # Mappa neve

def download_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Letöltés indul: {NUM_IMAGES} db {WIDTH}x{HEIGHT} kép...")

    for i in range(NUM_IMAGES):
        # A Lorem Picsum szolgáltatást használjuk véletlen képekhez
        url = f"https://picsum.photos/{WIDTH}/{HEIGHT}"
        
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filename = f"{OUTPUT_DIR}/image_{i:03d}.png" # PNG-t kérünk, de JPG is jöhet
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"[{i+1}/{NUM_IMAGES}] Letöltve: {filename}")
            else:
                print(f"Hiba a letöltésnél: {response.status_code}")
        except Exception as e:
            print(f"Hiba: {e}")
            
        # Kicsit várjunk, ne terheljük túl a szervert
        time.sleep(0.5)

    print("Kész! A képek a 'dataset' mappában vannak.")

if __name__ == "__main__":
    download_images()