from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
import numpy as np
import io

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

if __name__ == '__main__':
    app.run(debug=False)
