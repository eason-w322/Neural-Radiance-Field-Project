from PIL import Image
import pillow_heif 
import os

input_dir = "/Users/yihaiwei/Desktop/labubu4_images"  
output_dir = "data/labubu4_images"  

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith(".heic"):
        heic_path = os.path.join(input_dir, file)
        jpg_name = os.path.splitext(file)[0] + ".jpg"
        jpg_path = os.path.join(output_dir, jpg_name)

        heif_file = pillow_heif.read_heif(heic_path)
        image = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data
        )
        image.save(jpg_path, "JPEG")

        print(f"✅ Converted {file} → {jpg_name}")

print("🎉 Conversion complete! All images saved to data/calib_images/")