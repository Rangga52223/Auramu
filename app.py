from flask import Flask, request, render_template, send_file  # Tambahkan impor send_file untuk mengirim file sebagai respons
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
import numpy as np
import io
import cv2  # Tambahkan impor OpenCV
import base64  # Tambahkan impor base64 untuk encoding gambar

app = Flask(__name__)
model = load_model('model/deteksi_wajah3.h5')#model nya
#lebel nya
label_map_r = {
    0: 'Kemarahan',
    1: 'Jijik',
    2: 'Ketakutan',
    3: 'Kebahagiaan',
    4: 'Biasa Saja',
    5: 'Kesedihan',
    6: 'Kejutan'
}
#Pesan saya bikin random
messages = {
    "Kemarahan": [
        "Tenangkan hati, semuanya terkendali.",
        "Kendalikan emosi, raih solusi.",
        "Marah takkan selesaikan masalah.",
        "Berhenti sejenak, pikirkan ulang.",
        "Damai dalam hati membawa kekuatan."
    ],
    "Jijik": [
        "Fokus pada hal yang baik.",
        "Hilangkan rasa, pikirkan solusi.",
        "Temukan sisi positif dari situasi.",
        "Kita lebih kuat dari ketidaknyamanan.",
        "Hidup tak selalu indah, terus maju."
    ],
    "Ketakutan": [
        "Takut adalah langkah awal keberanian.",
        "Berani mencoba adalah kemenangan awal.",
        "Hadirkan kekuatan dalam setiap ketakutan.",
        "Langkahi rasa takut, capai tujuan.",
        "Keyakinan mengalahkan segala ketakutan."
    ],
    "Kebahagiaan": [
        "Bagikan senyummu, dunia butuh itu.",
        "Bahagia itu sederhana, syukuri hidup.",
        "Rasakan nikmat kecil, rayakan selalu.",
        "Tawa adalah obat terbaik jiwa.",
        "Hidup bahagia dimulai dari hati."
    ],
    "Biasa Saja": [
        "Lihatlah dengan pikiran terbuka.",
        "Jalan tengah seringkali terbaik.",
        "Netral adalah ruang untuk merenung.",
        "Dengarkan semua sisi sebelum bertindak.",
        "Ketegasan muncul dari netralitas hati."
    ],
    "Kesedihan": [
        "Setiap air mata membawa kekuatan baru.",
        "Sedih hanyalah sementara, bangkitlah.",
        "Cahaya selalu datang setelah gelap.",
        "Ambil waktu untuk sembuh, tetap maju.",
        "Jangan menyerah, bahagia menunggumu."
    ],
    "Kejutan": [
        "Hal baru membawa peluang baru.",
        "Hadapi kejutan dengan hati terbuka.",
        "Tak terduga, namun bisa jadi indah.",
        "Percayalah, segalanya punya maksud baik.",
        "Kejutan adalah awal dari petualangan."
    ]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    motivational_message = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file part"
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No selected file"
            elif file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    #graysacle Processing dan prediksi
                    img_bytes = io.BytesIO(file.read())
                    img = load_img(img_bytes, target_size=(48, 48), color_mode="grayscale")
                    predicted_label, motivational_message = process_image(img)
                    result = f"Auramu: {predicted_label}"
                except Exception as e:
                    result = f"Error processing image: {e}"
            else:
                result = "Invalid file format. Only PNG, JPG allowed."
    
    return render_template('index.html', result=result, motivational_message=motivational_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint untuk memproses gambar, menambahkan kotak di wajah, dan menyisipkan pesan motivasi.
    """
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        try:
            # Membaca file gambar menggunakan OpenCV
            img_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)  # Membaca gambar berwarna
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale
            
            # Deteksi wajah menggunakan Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Jika wajah terdeteksi, proses prediksi
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]  # Potong area wajah
                    face_resized = cv2.resize(face, (48, 48))  # Resize ke ukuran model
                    predicted_label, motivational_message = process_image_cv2(face_resized)

                    # Tambahkan kotak di sekitar wajah
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Tambahkan teks prediksi dan pesan motivasi di atas kotak
                    cv2.putText(img, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(img, motivational_message, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            else:
                return "No face detected in the image", 400

            # Simpan gambar hasil olahan ke buffer
            _, buffer = cv2.imencode('.jpg', img)
            return send_file(
                io.BytesIO(buffer),
                mimetype='image/jpeg',
                as_attachment=False,
                download_name='processed_image.jpg'
            )
        except Exception as e:
            return f"Error processing image: {str(e)}", 500
    else:
        return "Invalid file format. Only PNG, JPG allowed.", 400

def process_image(img):
    """
    Proses gambar dan prediksi label menggunakan model TensorFlow.
    """
    image_array = img_to_array(img)
    image_array = np.expand_dims(image_array, axis=-1) 
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_label = label_map_r[predicted_class]
    motivational_message = random.choice(messages[predicted_label])
    return predicted_label, motivational_message

def process_image_cv2(img):
    """
    Proses gambar menggunakan OpenCV dan prediksi label menggunakan model TensorFlow.
    """
    image_array = np.expand_dims(img, axis=-1)  # Tambahkan dimensi channel
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
    image_array = image_array / 255.0  # Normalisasi
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_label = label_map_r[predicted_class]
    motivational_message = random.choice(messages[predicted_label])
    return predicted_label, motivational_message

if __name__ == '__main__':
    app.run(debug=False)