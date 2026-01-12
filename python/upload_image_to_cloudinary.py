from pathlib import Path
from PIL import Image
import os
import cloudinary
import cloudinary.uploader


cloudinary.config(
    cloud_name="ds84b9f8s",        
    api_key="799828727114421",              
    api_secret="nVNpaFiobyQL655Wu496HjqrtI4",        
    secure=True
)


IMAGE_DIR = Path("/home/tuanh/M1/projet/code/code/python/image")

TARGET_SIZE = 300   

def resize_and_overwrite(image_path: Path, max_size: int = 300):

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        img.save(image_path, format="JPEG", optimize=True, quality=80)


def main():
    print("Dossier:", IMAGE_DIR)

    url_map = []

    for filename in os.listdir(IMAGE_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        path = IMAGE_DIR / filename

        try:
           
            resize_and_overwrite(path, TARGET_SIZE)
            print(f"[RESIZED] {filename}")

           
            upload_result = cloudinary.uploader.upload(
                str(path),
                folder="movies_small",
                public_id=path.stem
            )

            url = upload_result["secure_url"]
            url_map.append((filename, url))
            print(f"   -> [UPLOADED] {filename}: {url}")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    print("\n=== DONE ===")
    print("Total uploaded:", len(url_map))
    print("First 10 results:")
    for fname, url in url_map[:10]:
        print(fname, "->", url)


if __name__ == "__main__":
    main()
