import cv2

def check_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"Camera {i} is not available")
    return available_cameras

# Mengecek kamera hingga 10 indeks (sesuaikan jika perlu)
available_cameras = check_available_cameras()
print("Available cameras:", available_cameras)
