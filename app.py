import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

# Fungsi untuk memuat model TensorFlow yang sudah dilatih
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

# Fungsi untuk memprediksi gambar
def predict_image(image, model):
    size = (150, 150)
    # Convert image to numpy array
    img = np.array(image)
    # Resize using OpenCV
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    # Normalize
    img = img / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    # Predict with the model
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


# Fungsi untuk menampilkan halaman About
def show_about():
    st.title('Tentang')
    st.write('''Aplikasi ini dibuat untuk memenuhi tugas Penelitian Ilmiah yang berjudul
    'Penerapan Metode Convolutional Neural Network Pada Klasifikasi Cuaca' menggunakan 
    model deep learning untuk memprediksi tiga jenis cuaca berdasarkan keadaan langit yaitu 
    cuaca berawan, cuaca berkabut dan cuaca hujan. Dimana perancangan model menggunakan 
    tiga lapisan konvolusi, satu lapisan dropout dan satu lapisan akhir. 
    
    Berikut adalah grafik dari pelatihan dengan melakukan 20 epochs''')

    st.image("train.png", caption="Training and Validasi Accuracy", width=500)
    st.image("loss.png", caption="Training and Validasi Loss", width=500)

    st.write('''Pada iterasi tersebut, didapati bahwa akurasi terbaik untuk data pelatihan dan validasi 
    terjadi pada epoch ke-18, di mana akurasi terbaik untuk data pelatihan mencapai 86%, 
    sedangkan untuk data validasi mencapai 92%.''')

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

    model = load_model()

    if menu == 'Beranda':
        show_home()
    elif menu == 'Tentang':
        show_about()
    elif menu == 'Prediksi':
        show_prediction(model)

if __name__ == '__main__':
    main()
