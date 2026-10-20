# IF3170_G01_Tandborste

Program optimasi penyusunan paket dalam truk 3D menggunakan Hill-Climbing, Simulated Annealing, dan Genetic Algorithm untuk Tugas Besar 1 IF3170 Inteligensi Artifisial.

## Struktur Folder

```text
.
├── main.py             # Program utama
├── requirements.txt    # Dependensi Python
├── src/
│   ├── algorithms/     # Implementasi algoritma pencarian
│   ├── core/           # Model, state, batasan, dan fungsi objektif
│   ├── utils/          # Pembacaan input dan eksekusi pencarian
│   └── visualization/  # Visualisasi 3D dan grafik metrik
├── test_cases/         # Berkas input JSON
├── output/             # Hasil pencarian dan visualisasi
└── docs/               # Dokumen dan laporan
```

## Persiapan

Gunakan Python 3.10 atau lebih baru. Jalankan dari direktori utama proyek:

```bash
python -m pip install -r requirements.txt
```

## Konfigurasi dan Input

Simpan input JSON di `test_cases/`, mengikuti format `small.json`: objek `truck` berisi dimensi dan kapasitas truk, sedangkan daftar `packages` berisi atribut paket. Tersedia kasus `small.json`, `medium.json`, dan `large.json`.

Parameter algoritma diatur melalui argumen perintah. Gunakan `--input` untuk memilih berkas input, `--seed` untuk mengatur seed acak, dan `--objective default` atau `--objective bonus` untuk memilih fungsi objektif. Lihat seluruh parameter dengan:

```bash
python main.py --help
```

## Menjalankan Program

```bash
python main.py --input test_cases/small.json --algo steepest --seed 42 --save-plot --save-metrics
```

Pilihan `--algo`: `steepest`, `sideways`, `stochastic`, `restart`, `sa`, atau `ga`. Tambahkan `--plot` untuk membuka visualisasi di browser.

### Flag Umum

Semua flag bersifat opsional. Flag visualisasi digunakan tanpa nilai; flag lainnya diikuti nilai.

| Flag | Nilai bawaan | Keterangan |
|---|---|---|
| `-h`, `--help` | — | Tampilkan bantuan dan keluar. |
| `-i`, `--input` | `test_cases/small.json` | Lokasi input JSON. |
| `-a`, `--algo` | `steepest` | Pilihan algoritma pada tabel berikut. |
| `--objective` | `default` | `default`: total nilai paket; `bonus`: urgensi ETA dan rasio volume. |
| `--seed` | Tidak ditentukan | Seed bilangan acak berupa integer. |
| `-o`, `--output-dir` | `output` | Folder hasil eksekusi. |
| `--plot` | Nonaktif | Buka visualisasi 3D di browser. |
| `--save-plot` | Nonaktif | Simpan visualisasi 3D sebagai HTML. |
| `--save-metrics` | Nonaktif | Simpan grafik metrik sebagai PNG. |

### Flag per Algoritma

| Algoritma (`--algo`) | Flag dan nilai bawaan | Keterangan |
|---|---|---|
| `steepest` / `steepest_ascent` | `--max-iterations 300`, `--sample-neighbors 25` | Batas iterasi dan jumlah tetangga yang disampel per iterasi. |
| `sideways` / `sideways_move` | `--max-iterations 300`, `--sample-neighbors 25`, `--max-sideways 50` | Batas iterasi, sampel tetangga, dan batas langkah sideways berturut-turut. |
| `stochastic` | `--max-iterations 300` | Batas iterasi. |
| `restart` / `random_restart` | `--restarts 5`, `--iter-per-restart 100` | Batas restart dan iterasi per restart. |
| `sa` / `simulated_annealing` | `--t0 200.0`, `--alpha 0.95`, `--t-min 0.01`, `--iter-per-temp 10`, `--stuck-limit 50` | Suhu awal, faktor pendinginan, suhu akhir, iterasi per suhu, dan ambang penolakan berturut-turut untuk pencatatan kondisi stuck. |
| `ga` / `genetic_algorithm` | `--pop-size 30`, `--generations 50` | Ukuran populasi dan jumlah generasi. |

Alias flag: `--max-iter` sama dengan `--max-iterations`; `--max-restarts` sama dengan `--restarts`. Gunakan parameter sesuai algoritma yang dipilih; parameter algoritma lain diabaikan.

Contoh setiap algoritma dengan seluruh parameter khususnya:

```bash
python main.py --algo steepest --max-iterations 300 --sample-neighbors 25
python main.py --algo sideways --max-iterations 300 --sample-neighbors 25 --max-sideways 50
python main.py --algo stochastic --max-iterations 300
python main.py --algo restart --restarts 5 --iter-per-restart 100
python main.py --algo sa --t0 200.0 --alpha 0.95 --t-min 0.01 --iter-per-temp 10 --stuck-limit 50
python main.py --algo ga --pop-size 30 --generations 50
```

## Hasil Program

Hasil pencarian otomatis disimpan sebagai JSON di `output/`. Opsi `--save-plot` menyimpan visualisasi 3D sebagai HTML, sedangkan `--save-metrics` menyimpan grafik metrik sebagai PNG. Gunakan `--output-dir <folder>` untuk mengganti lokasi penyimpanan.

## Pembagian Tugas

| NIM | Tugas |
|---|---|
| 13524078 | Simulated Annealing, Genetic Algorithm, fungsi objektif alternatif, serta pengujian dan analisis eksperimen. |
| 13524086 | Representasi state, batasan, fungsi objektif, Hill-Climbing, serta pengujian dan analisis eksperimen. |
| 13524094 | Visualisasi dan integrasi program, serta penyusunan dan finalisasi laporan. |
