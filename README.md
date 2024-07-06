# Cyberbullying Detection Project

## Deskripsi Proyek
Proyek ini bertujuan untuk mendeteksi tweet yang mengandung cyberbullying menggunakan model pembelajaran mendalam (deep learning). Aplikasi ini dibangun menggunakan Streamlit sebagai antarmuka pengguna dan TensorFlow untuk model prediksi. Tweet yang dimasukkan oleh pengguna akan diproses dan diklasifikasikan ke dalam salah satu dari lima kategori: religion, age, ethnicity, gender, dan not_cyberbullying.

## Arsitektur Model
Model yang digunakan dalam proyek ini adalah model BiLSTM dengan lapisan embedding. Arsitektur model adalah sebagai berikut:


## Persyaratan


## Instalasi

1. Buat dan aktifkan virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

2. Instal dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download resource NLTK:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

4. Pastikan file `vocab.txt` dan model `cyberbullying_model.h5` tersedia di direktori proyek.

## Cara Menjalankan Aplikasi
1. Jalankan aplikasi Streamlit:
    ```bash
    streamlit run app.py
    ```

2. Buka browser dan akses aplikasi di `http://localhost:8501`.

## Alur Kerja Aplikasi
1. **Data Preparation**: Pengumpulan dan pemrosesan data tweet.
2. **EDA (Exploratory Data Analysis)**: Analisis data untuk memahami distribusi dan karakteristik data.
3. **Preprocessing**: Membersihkan teks dari emoji, URL, mentions, dll.
4. **Lemmatization**: Mengubah kata ke bentuk dasarnya.
5. **Data Splitting**: Membagi data menjadi set pelatihan dan pengujian.
6. **Vectorization**: Mengubah teks menjadi vektor menggunakan TextVectorization.
7. **Oversampling**: Menyeimbangkan data untuk mengatasi masalah data yang tidak seimbang.
8. **Model Architecture Design**: Merancang arsitektur model BiLSTM.
9. **Evaluation**: Mengevaluasi model menggunakan metrik seperti akurasi, precision, recall, dan confusion matrix.
10. **Deployment**: Menggunakan Streamlit untuk membuat antarmuka pengguna dan menghubungkan dengan IBM Cloud Pak for Data untuk deployment.

## Penjelasan Singkat Heatmap
Heatmap confusion matrix menunjukkan performa model dalam mengklasifikasikan setiap kategori. Setiap sel dalam matrix menunjukkan jumlah prediksi yang benar (diagonal) dan salah (off-diagonal) untuk setiap kelas.

## Kontribusi
Pull request sangat diterima untuk memperbaiki proyek ini. Untuk perubahan besar, harap buka isu terlebih dahulu untuk membahas apa yang ingin Anda ubah.

## Lisensi
Proyek ini dilisensikan di bawah MIT License.
