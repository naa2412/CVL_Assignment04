# Visual Object Tracking — Template Matching vs Optical Flow

Tugas Computer Vision: implementasi dan perbandingan dua metode tracking objek secara manual menggunakan NumPy, tanpa fungsi tracking bawaan OpenCV.

**Nama:** Azhar Maulana  
**NIM:** 24/533487/PA/22582  
**Mata Kuliah:** Computer Vision

---

## Metode yang Diimplementasikan

### 1. Template Matching (NCC)
Normalized Cross-Correlation mengukur kemiripan antara template dan sub-image kandidat dengan sliding window. Template diperbarui tiap frame (adaptive update) agar tahan terhadap perubahan penampilan objek.

$$NCC(i,j) = \frac{1}{N \cdot M} \sum_{m,n} \frac{(I(i+m,j+n) - \bar{I}_{i,j}) \cdot (T(m,n) - \bar{T})}{\sigma_{I_{i,j}} \cdot \sigma_T}$$

### 2. Optical Flow Lucas-Kanade
Lucas-Kanade mengasumsikan brightness constancy dan local motion:

$$I_x u + I_y v + I_t = 0$$

Untuk $n$ piksel dalam window $W$, susun sistem persamaan:

$$\mathbf{A} \mathbf{d} = \mathbf{b}$$

dengan $\mathbf{A} = [I_x, I_y]^T$, $\mathbf{b} = -I_t$

Solusi least-squares:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b}$$

Syarat: $\mathbf{A}^T \mathbf{A}$ harus invertible (tekstur cukup, bukan flat region).

---

## Dataset

Dataset dari **OTB2015** (Object Tracking Benchmark), dua sequence dengan karakteristik berbeda:

| Sequence | Karakteristik |
|----------|--------------|
| **Bolt** | Atlet sprint, displacement besar antar frame, banyak distractor di background |
| **Girl** | Wajah perempuan, gerakan halus, background relatif statis |

Anotasi ground truth dalam format `x, y, w, h` per frame.

---

## Hasil Eksperimen

### Sequence Bolt

| Metrik | Template Matching | Optical Flow LK |
|--------|:-----------------:|:---------------:|
| Mean IoU | 0.0839 | **0.3354** |
| Center Error | 164.77 px | **15.96 px** |
| RMSE | 199.71 px | **18.12 px** |
| Max Error | 339.74 px | **28.84 px** |
| Drift | +7.102 px/frame | +0.587 px/frame |
| Kecepatan | ~2.2 FPS | **~370 FPS** |

### Sequence Girl

| Metrik | Template Matching | Optical Flow LK |
|--------|:-----------------:|:---------------:|
| Mean IoU | **0.8269** | 0.2943 |
| Center Error | **1.63 px** | 19.70 px |
| RMSE | **2.27 px** | 21.44 px |
| Max Error | **6.04 px** | 33.24 px |
| Drift | +0.063 px/frame | +0.445 px/frame |
| Kecepatan | ~15 FPS | **~667 FPS** |

### Kesimpulan

- **Optical Flow LK unggul di Bolt** karena displacement antar frame besar, sedangkan template matching gagal menemukan kembali objek setelah bergerak jauh dari search radius.
- **Template Matching unggul di Girl** karena gerakan halus dan background statis memudahkan pencocokan template; LK justru drift karena keypoint menyebar ke background.
- **LK selalu lebih cepat** dua hingga tiga orde magnitudo dibanding NCC, karena NCC perlu menggeser sliding window di seluruh search region.

---

## Struktur Notebook

```
CVL_Assignment04.ipynb
├── Import & Install Library
├── Download Dataset (OTB2015 via gdown)
├── Ekstraksi Frame dari Video
├── Preview Ground Truth
├── Implementasi Template Matching NCC
├── Implementasi Optical Flow Lucas-Kanade (+ Shi-Tomasi)
├── Fungsi Evaluasi (IoU, Center Error, RMSE, Drift, FPS)
├── Pengujian dan Evaluasi
└── Visualisasi (frame grid + video output)
```

---

## Requirements

```
opencv-python-headless
numpy
matplotlib
imageio[ffmpeg]
gdown
```

Install:
```bash
pip install opencv-python-headless imageio[ffmpeg] gdown
```

Versi yang digunakan: OpenCV 4.13.0, NumPy 2.0.2

---

## Cara Menjalankan

1. Buka notebook di Google Colab
2. Jalankan sel secara berurutan — dataset diunduh otomatis dari Google Drive
3. Output berupa:
   - Tabel metrik di terminal
   - Grid visualisasi frame sample (`.png`)
   - Video hasil tracking (`.mp4`) untuk kedua sequence

---

## Referensi

- Wu, Y., Lim, J., & Yang, M. H. (2013). Online object tracking: A benchmark. *CVPR 2013*. ([OTB2015](http://cvlab.hanyang.ac.kr/tracker_benchmark/))
- Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. *IJCAI*.
- Shi, J., & Tomasi, C. (1994). Good features to track. *CVPR 1994*.
