# Cyberbullying Detection Project

## Deskripsi Proyek
Proyek ini bertujuan untuk mendeteksi tweet yang mengandung cyberbullying menggunakan model pembelajaran mendalam (deep learning). Aplikasi ini dibangun menggunakan Streamlit sebagai antarmuka pengguna dan TensorFlow untuk model prediksi. Tweet yang dimasukkan oleh pengguna akan diproses dan diklasifikasikan ke dalam salah satu dari lima kategori: religion, age, ethnicity, gender, dan not_cyberbullying.

## Arsitektur Model

Model yang digunakan dalam proyek ini adalah model BiLSTM dengan lapisan embedding. Berikut adalah arsitektur lengkap model:

| Layer Type      | Output Shape  | Param #      |
|-----------------|---------------|--------------|
| Embedding       | (None, 100, 100) | 1,000,000   |
| Bidirectional (LSTM) | (None, 64) | 34,048       |
| Dropout         | (None, 64)    | 0            |
| Dense           | (None, 16)    | 1,040        |
| Dropout         | (None, 16)    | 0            |
| Dense           | (None, 5)     | 85           |

**Total params:** 1,035,173  
**Trainable params:** 1,035,173  
**Non-trainable params:** 0
### Embedding Layer
Mengubah kata-kata dalam input teks menjadi vektor berdimensi tetap. Ini memetakan kata-kata dari ruang diskret (kata) ke ruang kontinu (vektor).
Parameter:
input_dim=vocab_size: Ukuran kosa kata, yaitu jumlah kata unik yang bisa diterima oleh model.
output_dim=100: Dimensi dari embedding vector. Ini menentukan panjang vektor yang mewakili setiap kata.
input_length=100: Panjang input sequences, yang berarti setiap input akan dipotong atau dipadding untuk menjadi panjang tetap 100.

### Bi-LSTM Layer
Memproses informasi dalam urutan dari dua arah (maju dan mundur) untuk menangkap konteks dari seluruh sequence teks.
Parameter:
32: Jumlah unit LSTM. Ini menentukan dimensi output dari layer LSTM.
return_sequences=False: Menghasilkan output hanya dari langkah waktu terakhir, bukan output untuk setiap langkah waktu.
kernel_regularizer=regularizer: Menambahkan regularisasi L2 untuk mengendalikan overfitting.

### Dropout Layer
Mengurangi overfitting dengan secara acak mematikan sejumlah unit selama pelatihan. Ini membantu model menjadi lebih general dan menghindari ketergantungan pada unit tertentu.
Parameter:
0.5: Tingkat dropout, yaitu 50% dari neuron akan dipadamkan secara acak selama pelatihan.

### Dense Layer
Menyediakan lapisan fully connected yang menghubungkan setiap neuron dari layer sebelumnya ke setiap neuron di layer ini. Ini menambahkan kekuatan non-linearitas ke model.
Parameter:
32: Jumlah neuron dalam layer dense.
activation='relu': Fungsi aktivasi ReLU (Rectified Linear Unit) yang mengaktifkan neuron hanya jika outputnya positif. Ini membantu model belajar representasi yang lebih baik.

### Dense Layer output
Menghasilkan probabilitas klasifikasi untuk setiap kelas. Ini adalah layer output untuk tugas klasifikasi multi-kelas.
Parameter:
num_classes=5: Jumlah kelas output.
activation='softmax': Fungsi aktivasi softmax mengubah output menjadi probabilitas yang dijumlahkan menjadi 1.

## Persyaratan
- streamlit
- tensorflow==2.15.0
- pandas==2.0.3
- numpy==1.25.2
- nltk==3.8.1
- demoji==1.1.0
- langdetect==1.0.9
- contractions==0.1.73

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

