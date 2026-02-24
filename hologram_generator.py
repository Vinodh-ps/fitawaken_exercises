import sys
print("1. Script started. Loading core libraries...")

try:
    import os
    import numpy as np
    from PIL import Image, ImageSequence
    print("2. Core libraries loaded. Loading OpenCV...")
    import cv2
    print("3. OpenCV loaded. Loading MediaPipe (this might take a second)...")
    import mediapipe as mp
    print("4. MediaPipe loaded successfully!")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Please run: pip install opencv-python mediapipe numpy Pillow")
    sys.exit(1)

input_dir = 'input_gifs'
output_dir = 'output_hologram_gifs'

os.makedirs(output_dir, exist_ok=True)

def process_gifs():
    print("5. Initializing AI Pose Detection Model...")
    try:
        mp_pose = mp.solutions.pose
        # Initializing inside the function prevents global silent crashes
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        # Colors in BGR (OpenCV format)
        CYAN_BGR = (255, 255, 0)     # Fitawaken Cyan
        ORANGE_BGR = (0, 165, 255)   # Fitawaken Orange
        
        custom_landmarks = mp_drawing.DrawingSpec(color=CYAN_BGR, thickness=2, circle_radius=2)
        custom_connections = mp_drawing.DrawingSpec(color=ORANGE_BGR, thickness=2)
        print("6. AI Model ready! Starting to process GIFs...")
    except Exception as e:
        print(f"❌ ERROR INITIALIZING MEDIAPIPE: {e}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith('.gif')]
    if not files:
        print(f"⚠️ No GIFs found in {input_dir} folder.")
        return

    for filename in files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(in_path) as im:
                frames = []
                
                for frame in ImageSequence.Iterator(im):
                    # Convert PIL to NumPy for OpenCV/MediaPipe
                    frame_rgb = frame.convert('RGB')
                    frame_np = np.array(frame_rgb)
                    
                    # Process with MediaPipe
                    results = pose.process(frame_np)
                    
                    # Create black canvas
                    h, w, _ = frame_np.shape
                    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Draw glowing skeleton
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            black_canvas,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=custom_landmarks,
                            connection_drawing_spec=custom_connections
                        )
                    
                    # Convert back to PIL Image
                    frames.append(Image.fromarray(black_canvas))
                
                # Save the new skeletal GIF
                if frames:
                    frames[0].save(
                        out_path,
                        save_all=True,
                        append_images=frames[1:],
                        loop=0,
                        duration=im.info.get('duration', 100)
                    )
            print(f"✅ Hologram generated: {filename}")
            
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")

if __name__ == "__main__":
    process_gifs()
    print("Done processing all files!")