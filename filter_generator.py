import os
from PIL import Image, ImageOps, ImageSequence

input_dir = 'input_gifs'
output_dir = 'output_filter_gifs'

os.makedirs(output_dir, exist_ok=True)

# Fitawaken Brand Colors (Hex)
# Try swapping #00FFFF (Cyan) with #FF5F1F (Neon Orange) to see what you like!
NEON_TINT = "#00FFFF" 

def process_gifs():
    for filename in os.listdir(input_dir):
        if not filename.endswith('.gif'):
            continue
            
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(in_path) as im:
                frames = []
                # Process each frame of the GIF
                for frame in ImageSequence.Iterator(im):
                    # Convert to RGB to ensure color compatibility
                    rgb_frame = frame.convert('RGB')
                    
                    # 1. Convert to grayscale
                    gray_frame = ImageOps.grayscale(rgb_frame)
                    
                    # 2. Colorize (Black stays black, White becomes our Neon color)
                    cyan_tint = ImageOps.colorize(gray_frame, black="black", white=NEON_TINT)
                    
                    # 3. Mirror the image (Flips left/right for extra obfuscation)
                    flipped = ImageOps.mirror(cyan_tint)
                    
                    frames.append(flipped)
                
                # Save as a new looping GIF
                frames[0].save(
                    out_path,
                    save_all=True,
                    append_images=frames[1:],
                    loop=0,
                    duration=im.info.get('duration', 100) # Keep original speed
                )
            print(f"✅ Filtered: {filename}")
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")

if __name__ == "__main__":
    print("Starting Cyberpunk Filter Processing...")
    process_gifs()
    print("Done!")