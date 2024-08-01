import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2

# Fungsi untuk memuat model TensorFlow yang sudah dilatih
def load_trained_model():
    try:
        model = tf.keras.models.load_model('model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        return model
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None


# Fungsi untuk memprediksi gambar
def predict_image(image, model):
    # Load gambar menggunakan PIL (Pillow)
    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

# Fungsi untuk menampilkan halaman Beranda
def show_home():
    st.title('Beranda')
    st.header('Selamat datang di aplikasi prediksi cuaca!')

    # Menambahkan gambar sampel
    st.write("Sampel Gambar")
    st.image("cloudy.jpg", caption="Cloudy", width=600)
    st.image("foggy.jpg", caption="Foggy", width=600)
    st.image("rainy.jpg", caption="Rainy", width=600)
    st.caption(':red[:copyright: Najmah Femalea, 2024]')

# Fungsi untuk menampilkan halaman About
def show_about():
    st.title('Tentang')
    st.write('''Aplikasi ini dibuat untuk memenuhi tugas Penelitian Ilmiah yang berjudul
    'Penerapan Model Xception Dalam Transfer Learning menggunakan Metode Convolutional Neural Network Pada Klasifikasi Cuaca' 
    model deep learning ini berguna untuk memprediksi tiga jenis cuaca berdasarkan keadaan langit yaitu 
    cuaca berawan, cuaca berkabut dan cuaca hujan. Dimana perancangan model menggunakan transfer learning Xception 
    terdiri dari basemodel, satu lapisan pooling dan satu lapisan akhir.
    
    Berikut adalah grafik dari pelatihan dengan melakukan 10 epochs''')

    st.image("grafik hd.png", caption="Grafik Pelatihan", width=700)

    st.write('''Pada iterasi tersebut, diperoleh bahwa akurasi validasi terbaik terjadi pada epoch ke-5 sebesar 94.64%, 
    sedangkan loss validasi terbaik terjadia pada epoch ke-8 sebesar 13.44%.''')

# Fungsi untuk menampilkan halaman Prediksi
def show_prediction(model):
    st.title('Prediksi Cuaca Berdasarkan Gambar')

    uploaded_file = st.file_uploader('Upload gambar', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah.', width=400)
        
        if st.button('Prediksi Cuaca'):
            prediction = predict_image(image, model)
            class_names = ['Cloudy', 'Foggy', 'Rainy']
            st.write(f'Prediksi: {class_names[np.argmax(prediction)]}')
            st.write(f'Probabilitas: {np.max(prediction) * 100:.2f}%')

# Fungsi utama untuk Streamlit app
def main():
    st.sidebar.title('Menu Navigasi')
    menu = st.sidebar.selectbox('Pilih Halaman', ['Beranda', 'Tentang', 'Prediksi'])

    model = load_trained_model()

    if menu == 'Beranda':
        show_home()
    elif menu == 'Tentang':
        show_about()
    elif menu == 'Prediksi':
        show_prediction(model)

if __name__ == '__main__':
    main()