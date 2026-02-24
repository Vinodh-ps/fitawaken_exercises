import sys

print("1. Starting environment test...")

try:
    import cv2
    print(f"2. OpenCV loaded. Version: {cv2.__version__}")
except Exception as e:
    print(f"❌ Failed to load OpenCV: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print("3. MediaPipe imported successfully.")
except Exception as e:
    print(f"❌ Failed to import MediaPipe: {e}")
    sys.exit(1)

try:
    print("4. Attempting to initialize Pose model...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    print("5. ✅ Pose model initialized successfully! Environment is perfect.")
except Exception as e:
    print(f"❌ Failed to initialize Pose model: {e}")