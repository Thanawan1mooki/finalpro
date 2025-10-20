from PIL import Image
import pillow_avif
import glob, os

src_folder = r"C:\projects\banana_classification\train\images"

for avif_file in glob.glob(os.path.join(src_folder, "*.avif")):
    img = Image.open(avif_file)
    new_path = os.path.splitext(avif_file)[0] + ".jpg"
    img.convert("RGB").save(new_path, "JPEG", quality=95)
    print(f"Converted: {os.path.basename(new_path)}")

print("✅ Done converting all .avif to .jpg")
