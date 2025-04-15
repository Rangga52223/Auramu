import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Muat model deteksi emosi
model = load_model('model/deteksi_wajah3.h5')

# Label emosi
label_map_r = {
    0: 'Kemarahan',
    1: 'Jijik',
    2: 'Ketakutan',
    3: 'Kebahagiaan',
    4: 'Biasa Saja',
    5: 'Kesedihan',
    6: 'Kejutan'
}

# Muat Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    # Resize wajah ke ukuran 48x48, lalu normalisasi
    face_resized = cv2.resize(face, (48, 48))
    face_normalized = face_resized / 255.0
    # Tambahkan dimensi untuk batch dan channel
    face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))  # (1, 48, 48, 1)
    return face_reshaped

# Buka kamera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke grayscale (dibutuhkan oleh Haar Cascade)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah di frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Ekstrak area wajah
        face = gray_frame[y:y+h, x:x+w]

        # Preprocess wajah
        processed_face = preprocess_face(face)

        # Prediksi emosi
        prediction = model.predict(processed_face)
        emotion = label_map_r[np.argmax(prediction)]

        # Tampilkan emosi pada layar
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Tampilkan hasil frame
    cv2.imshow('Deteksi Emosi dengan Tracking Wajah', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
