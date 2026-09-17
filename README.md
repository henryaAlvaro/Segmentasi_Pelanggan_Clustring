# Segmentasi Pelanggan E-Commerce Menggunakan RFM dan K-Means

## Gambaran Umum Proyek

Proyek ini bertujuan melakukan **segmentasi pelanggan pada bisnis e-commerce** berdasarkan perilaku transaksi menggunakan metode **RFM (Recency, Frequency, Monetary)** dan algoritma **K-Means Clustering**.

Hasil segmentasi digunakan untuk memahami karakteristik pelanggan dan menghasilkan insight yang dapat mendukung strategi pemasaran, retensi pelanggan, serta peningkatan nilai pelanggan.

---

## Latar Belakang

Dalam bisnis e-commerce, setiap pelanggan memiliki pola transaksi yang berbeda. Ada pelanggan yang baru melakukan pembelian, sering bertransaksi, atau memberikan kontribusi pendapatan yang besar. Sebaliknya, terdapat pelanggan yang sudah lama tidak melakukan pembelian dan hanya melakukan sedikit transaksi.

Jika seluruh pelanggan diberikan strategi pemasaran yang sama, perusahaan dapat kehilangan peluang untuk melakukan pendekatan yang lebih sesuai dengan karakteristik masing-masing pelanggan.

Oleh karena itu, diperlukan pendekatan berbasis data untuk mengelompokkan pelanggan berdasarkan perilaku transaksi mereka.

---

## Pertanyaan Bisnis

Beberapa pertanyaan yang ingin dijawab melalui proyek ini:

* Pelanggan mana yang masih aktif melakukan pembelian?
* Pelanggan mana yang memiliki frekuensi transaksi tinggi?
* Pelanggan mana yang memberikan nilai transaksi lebih tinggi?
* Apakah pelanggan dapat dikelompokkan berdasarkan perilaku transaksinya?
* Strategi pemasaran seperti apa yang dapat diterapkan pada setiap kelompok pelanggan?

---

## Tujuan Proyek

1. Membersihkan dan mempersiapkan data transaksi.
2. Menghitung nilai **Recency, Frequency, dan Monetary (RFM)** setiap pelanggan.
3. Menganalisis distribusi dan skewness pada variabel RFM.
4. Melakukan transformasi data menggunakan `log1p`.
5. Melakukan standardisasi menggunakan `StandardScaler`.
6. Menentukan jumlah cluster yang sesuai menggunakan **Elbow Method** dan **Silhouette Score**.
7. Melakukan segmentasi pelanggan menggunakan **K-Means Clustering**.
8. Menganalisis karakteristik setiap cluster berdasarkan nilai RFM.
9. Memvisualisasikan cluster menggunakan **PCA**.
10. Menghasilkan business insight dan rekomendasi berdasarkan hasil segmentasi.

---

# Dataset

Dataset yang digunakan adalah **Online Retail Dataset**.

Dataset awal memiliki:

* **541.909 transaksi**
* **8 variabel**
* Informasi invoice
* Informasi pelanggan
* Informasi produk
* Quantity
* Invoice Date
* Unit Price
* Country

Setelah proses pembersihan data:

* **397.884 transaksi valid**
* **4.338 pelanggan**

> Nilai akhir dapat sedikit berbeda apabila dataset diproses ulang menggunakan versi sumber yang berbeda.

---

#  Data Preparation

Tahapan preprocessing yang dilakukan meliputi:

* Memeriksa missing value
* Memeriksa data duplikat
* Memeriksa Quantity bernilai negatif
* Memeriksa UnitPrice bernilai tidak valid
* Menghapus transaksi dengan Quantity ≤ 0
* Menghapus transaksi dengan UnitPrice ≤ 0
* Menghapus transaksi tanpa `CustomerID`
* Mengubah `InvoiceDate` menjadi format datetime

### Pembuatan Revenue

Nilai revenue dihitung menggunakan:

```python
df_clean["Revenue"] = (
    df_clean["Quantity"] *
    df_clean["UnitPrice"]
)
```

Dengan demikian setiap transaksi memiliki nilai pendapatan yang dapat digunakan untuk menghitung Monetary.

---

#  Analisis RFM

RFM digunakan untuk merepresentasikan perilaku transaksi setiap pelanggan.

| Variabel      | Definisi                             | Interpretasi                                 |
| ------------- | ------------------------------------ | -------------------------------------------- |
| **Recency**   | Jumlah hari sejak transaksi terakhir | Semakin kecil semakin baru                   |
| **Frequency** | Jumlah invoice unik                  | Semakin besar semakin sering                 |
| **Monetary**  | Total revenue pelanggan              | Semakin besar semakin tinggi nilai transaksi |

Reference date ditentukan menggunakan:

```python
reference_date = (
    df_clean["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)
```

---

# Analisis Distribusi RFM

Variabel RFM memiliki distribusi yang cenderung **skewed**, terutama Monetary.

Hal ini terjadi karena terdapat pelanggan dengan nilai transaksi yang jauh lebih besar dibandingkan sebagian besar pelanggan lainnya.

Daripada langsung menghapus pelanggan dengan nilai tinggi, dilakukan transformasi:

```python
rfm_log = np.log1p(rfm)
```

Transformasi ini digunakan untuk mengurangi pengaruh distribusi yang sangat menceng sebelum proses clustering.

---

# Standardisasi Data

Setelah transformasi log, data distandardisasi menggunakan:

```python
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_log)
```

Standardisasi diperlukan agar setiap variabel memiliki skala yang sebanding ketika digunakan oleh K-Means.

> Data hasil transformasi dan standardisasi digunakan untuk proses modeling, sedangkan nilai RFM asli tetap digunakan ketika melakukan interpretasi bisnis.

---

# K-Means Clustering

Algoritma yang digunakan adalah **K-Means Clustering**.

K-Means digunakan untuk mengelompokkan pelanggan berdasarkan kemiripan karakteristik RFM.

Proses evaluasi dilakukan pada beberapa nilai `K`, misalnya:

```text
K = 2
K = 3
K = 4
...
K = 10
```

Dua metode evaluasi digunakan:

### 1. Elbow Method

Digunakan untuk melihat perubahan **inertia** ketika jumlah cluster bertambah.

### 2. Silhouette Score

Digunakan untuk mengevaluasi seberapa baik pelanggan berada di dalam cluster masing-masing serta seberapa terpisah cluster tersebut dari cluster lainnya.

Jumlah cluster akhir ditentukan dengan mempertimbangkan **Silhouette Score, Elbow Method, dan karakteristik bisnis setiap cluster**.

> Nilai K final perlu dikonfirmasi setelah notebook revisi dijalankan menggunakan dataset sumber.

---

# Profiling Cluster

Setelah model K-Means terbentuk, setiap cluster dianalisis menggunakan nilai RFM asli.

Analisis mencakup:

* Jumlah pelanggan
* Persentase pelanggan
* Rata-rata Recency
* Rata-rata Frequency
* Rata-rata Monetary
* Median Recency
* Median Frequency
* Median Monetary
* Total Monetary
* Persentase kontribusi revenue

Hal ini dilakukan agar hasil clustering tidak hanya dilihat dari model machine learning, tetapi juga dapat diterjemahkan menjadi informasi bisnis.

---

# Hasil Segmentasi

Hasil 2-cluster yang terdokumentasi dari analisis sebelumnya adalah:

| Cluster   | Jumlah Pelanggan | Persentase | Rata-rata Recency | Rata-rata Frequency | Rata-rata Monetary |
| --------- | ---------------: | ---------: | ----------------: | ------------------: | -----------------: |
| Cluster 0 |            2.671 |     61,57% |            134,14 |                1,67 |             497,74 |
| Cluster 1 |            1.667 |     38,43% |             25,88 |                8,44 |           4.548,26 |

> Nomor cluster bersifat arbitrer. Nama segmentasi harus ditentukan berdasarkan karakteristik RFM, bukan berdasarkan nomor cluster.

### Cluster 0

Karakteristik:

* Recency lebih tinggi
* Frequency lebih rendah
* Monetary lebih rendah
* Jumlah pelanggan lebih besar

Pola tersebut menunjukkan kelompok pelanggan dengan tingkat aktivitas transaksi yang lebih rendah.

### Cluster 1

Karakteristik:

* Recency lebih rendah
* Frequency lebih tinggi
* Monetary lebih tinggi
* Jumlah pelanggan lebih sedikit

Pola tersebut menunjukkan kelompok pelanggan dengan aktivitas transaksi dan nilai transaksi yang lebih tinggi.

---

# Business Insight

## 1. Perilaku pelanggan tidak homogen

Hasil RFM menunjukkan adanya perbedaan perilaku transaksi antar kelompok pelanggan.

Hal ini menunjukkan bahwa pendekatan pemasaran yang sama untuk seluruh pelanggan belum tentu sesuai untuk setiap kelompok.

---

## 2. Sebagian besar pelanggan berada pada kelompok dengan aktivitas transaksi lebih rendah

Pada hasil 2-cluster terdokumentasi, **61,57% pelanggan** berada pada Cluster 0.

Kelompok ini memiliki:

* Rata-rata Recency: **134,14 hari**
* Rata-rata Frequency: **1,67 transaksi**
* Rata-rata Monetary: **497,74**

Pola tersebut dapat menjadi dasar untuk melakukan strategi **reaktivasi pelanggan**.

Contohnya:

* Kampanye win-back
* Pengingat produk
* Penawaran personal
* Rekomendasi produk
* Promo pembelian kembali

---

## 3. Kelompok dengan jumlah pelanggan lebih kecil memiliki aktivitas transaksi lebih tinggi

Cluster 1 mencakup **38,43% pelanggan**.

Namun memiliki:

* Rata-rata Recency: **25,88 hari**
* Rata-rata Frequency: **8,44 transaksi**
* Rata-rata Monetary: **4.548,26**

Kelompok ini menunjukkan pola transaksi yang lebih aktif dibandingkan Cluster 0.

Strategi yang dapat dipertimbangkan:

* Program loyalitas
* Cross-selling
* Upselling
* Penawaran eksklusif
* Rekomendasi produk yang dipersonalisasi

---

## 4. Jumlah pelanggan tidak sama dengan kontribusi revenue

Salah satu hal penting dalam analisis segmentasi adalah membedakan:

```text
Persentase Pelanggan
        ≠
Persentase Revenue
```

Sebuah cluster dapat memiliki jumlah pelanggan yang besar tetapi belum tentu menghasilkan revenue terbesar.

Karena itu, analisis final perlu memperlihatkan:

* Customer Percentage
* Revenue Percentage

secara bersamaan.

---

# Rekomendasi Bisnis

| Karakteristik Pelanggan                           | Strategi                    |
| ------------------------------------------------- | --------------------------- |
| Recency tinggi, Frequency rendah, Monetary rendah | Win-back dan reaktivasi     |
| Recency rendah, Frequency tinggi, Monetary tinggi | Retensi dan loyalitas       |
| Monetary tinggi tetapi Recency tinggi             | Kampanye re-engagement      |
| Recency rendah tetapi Monetary rendah             | Mendorong repeat purchase   |
| Frequency tinggi tetapi Monetary rendah           | Cross-selling dan upselling |

Strategi tersebut merupakan **kerangka rekomendasi berdasarkan pola RFM**. Implementasi aktual tetap perlu mempertimbangkan biaya kampanye, margin produk, dan hasil kampanye sebelumnya.

---

# Visualisasi PCA

PCA digunakan untuk mengurangi dimensi data RFM menjadi dua komponen sehingga cluster dapat divisualisasikan dalam ruang 2D.

PCA digunakan hanya untuk:

> **Visualisasi dan eksplorasi hasil clustering**

PCA tidak digunakan sebagai algoritma utama untuk membentuk cluster.

Pada analisis sebelumnya:

* PC1 menjelaskan sekitar **75,10%** variasi
* PC2 menjelaskan sekitar **18,76%**
* Total sekitar **93,86%** variasi

---

# Alur Proyek

```text
Data Transaksi
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Revenue
      ↓
RFM Calculation
      ↓
Analisis Distribusi
      ↓
Log Transformation
      ↓
StandardScaler
      ↓
K-Means Evaluation
      ↓
Elbow + Silhouette
      ↓
Final K-Means
      ↓
Cluster Profiling
      ↓
PCA Visualization
      ↓
Business Insight
      ↓
Business Recommendation
```

---

# Teknologi yang Digunakan

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# Struktur Repository

```text
customer-segmentation-rfm-kmeans/
│
├── Dataset/
│   └── Online_Retail.xlsx
│
├── Notebook/
│   └── customer_segmentation_rfm_kmeans.ipynb
│
├── images/
│   ├── customer_distribution.png
│   ├── rfm_profile.png
│   ├── rfm_relative_comparison.png
│   ├── elbow_curve.png
│   ├── silhouette_curve.png
│   └── pca_clusters.png
│
├── README.md
│
└── requirements.txt
```

---

#  Cara Menjalankan Project

### 1. Clone repository

```bash
git clone https://github.com/USERNAME/customer-segmentation-rfm-kmeans.git
```

Masuk ke folder:

```bash
cd customer-segmentation-rfm-kmeans
```

### 2. Install library

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

### 3. Jalankan Jupyter Notebook

```bash
jupyter notebook
```

Kemudian buka:

```text
Notebook/customer_segmentation_rfm_kmeans.ipynb
```

Jalankan notebook dari awal hingga akhir.

---

# Keterbatasan

Beberapa keterbatasan dalam proyek ini:

* RFM hanya menggambarkan perilaku berdasarkan transaksi.
* Faktor demografis pelanggan belum digunakan.
* Preferensi produk belum dimasukkan ke dalam clustering.
* K-Means membutuhkan penentuan jumlah cluster.
* Transformasi dan standardisasi dapat memengaruhi hasil clustering.
* Customer dengan nilai transaksi ekstrem tetap perlu diperhatikan ketika melakukan interpretasi bisnis.
* Cluster ID tidak memiliki makna bisnis secara langsung.
* Hasil segmentasi perlu divalidasi kembali menggunakan data atau periode transaksi yang berbeda.

---

# Pengembangan Selanjutnya

Project ini dapat dikembangkan dengan:

* Customer Lifetime Value (CLV)
* Preferensi kategori produk
* Analisis produk yang sering dibeli
* Geographic segmentation
* Hierarchical Clustering
* Perbandingan dengan algoritma clustering lainnya
* Dashboard interaktif menggunakan Plotly
* Dashboard menggunakan Streamlit
* Analisis efektivitas campaign berdasarkan segmentasi
* Sistem rekomendasi produk berdasarkan segmentasi pelanggan

---
