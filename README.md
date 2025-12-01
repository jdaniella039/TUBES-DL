# TUBES-DL: Deep Learning Image Classification

## Deskripsi Proyek
TUBES-DL adalah proyek klasifikasi gambar menggunakan deep learning. Dalam proyek ini, dua model utama dilatih untuk melakukan tugas klasifikasi visual, yaitu **ResNet** (model CNN) dan **Swin Transformer (Swin-T)**. Swin-T dipilih sebagai model utama karena kemampuannya untuk menangkap pola visual yang lebih kompleks dengan efisiensi komputasi yang tinggi. Proyek ini mencakup pipeline lengkap untuk pelatihan model, evaluasi, dan aplikasi inferensi untuk mengklasifikasikan gambar.

## Fitur Utama
- **Model ResNet dan Swin-T**: Dua model deep learning dilatih untuk membandingkan performa klasifikasi gambar.
- **Pipeline End-to-End**: Mulai dari preprocessing data, pelatihan model, evaluasi, hingga deployment aplikasi untuk inferensi.
- **Aplikasi Inferensi**: Menggunakan model yang sudah dilatih untuk mengklasifikasikan gambar baru.
- **Checkpoint Management**: Memungkinkan untuk menyimpan dan memuat model yang telah dilatih dengan mudah.

## Persyaratan
Sebelum menjalankan proyek ini, pastikan bahwa Anda telah menginstal dependensi berikut:
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Pillow

### Instalasi
1. **Clone repository**:
   ```bash
   git clone https://github.com/jdaniella039/TUBES-DL.git
   cd TUBES-DL
   ```

2. **Instal dependensi**:
   Gunakan pip untuk menginstal semua dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

3. **Siapkan Dataset**:
   Tempatkan dataset gambar di folder `data/`. Dataset harus dalam format yang sesuai dengan struktur yang digunakan oleh skrip pelatihan.

## Penggunaan
### Pelatihan Model
Untuk melatih model **ResNet**:
```bash
python train_resnet.py --epochs 50 --batch_size 32
```
Untuk melatih model **Swin-T**:
```bash
python train_swin_t.py --epochs 50 --batch_size 32
```

### Evaluasi Model
Setelah pelatihan selesai, model dapat dievaluasi dengan skrip berikut:
- Untuk **ResNet**:
  ```bash
  python eval_resnet_model.py --model_path path/to/resnet_model.pth
  ```
- Untuk **Swin-T**:
  ```bash
  python eval_swin_t_model.py --model_path path/to/swin_t_model.pth
  ```

### Aplikasi Inferensi
Untuk menjalankan aplikasi inferensi menggunakan model **Swin-T**:
```bash
python app_swin_t.py --image_path path/to/input_image.jpg
```
Aplikasi akan menampilkan gambar input dan prediksi kelas yang dihasilkan oleh model beserta confidence score.

### Manajemen Checkpoint
Untuk memperbaiki atau mengganti nama checkpoint yang rusak, kamu bisa menggunakan skrip utilitas berikut:
- **Fix checkpoint**:  
  ```bash
  python fix_checkpoint.py --checkpoint path/to/checkpoint.pth
  ```
- **Rename checkpoint**:  
  ```bash
  python rename_ckpt.py --dest path/to/renamed_checkpoint.pth --move
  ```

## Struktur Folder
- `data/` - Tempat untuk menyimpan dataset gambar.
- `models/` - Menyimpan file model yang sudah dilatih.
- `scripts/` - Skrip utama untuk pelatihan, evaluasi, dan inferensi.
- `utils/` - Skrip untuk utilitas, seperti perbaikan dan manajemen checkpoint.

## Catatan
- Dataset yang digunakan dalam proyek ini harus disusun dengan struktur yang konsisten agar dapat digunakan dengan baik oleh skrip pelatihan dan evaluasi.
- Pelatihan dapat memakan waktu yang cukup lama tergantung pada spesifikasi perangkat keras yang digunakan.

## Kontribusi
Jika Anda ingin berkontribusi pada proyek ini, silakan fork repository ini dan kirim pull request dengan perubahan yang Anda buat. Semua kontribusi sangat dihargai!

## Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE).
