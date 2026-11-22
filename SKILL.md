---
name: motion-papercut
description: "Bikin video motion explainer gaya papercraft (kertas guntingan): intro papan kertas gantung, bukit sobekan kertas, huruf stiker, tag, kartu polaroid, counter, grafik, strip takeaway, musik + SFX kertas otomatis. Dari topik, naskah, dan/atau voice over (VO ditranskrip offline dengan Whisper base via onnxruntime). Format bisa diatur."
---

# Motion Papercut (explainer kertas guntingan + musik + SFX)

Kamu tidak menggambar frame sendiri. Kamu menulis SATU spec JSON lalu merendernya dengan mesin `papercut.py`.
Mesin mengunci konsep desain: semua elemen jadi potongan kertas bertekstur dengan bayangan lembut, intro papan kertas
gantung (kata judul jatuh + konfeti, lalu chip topik tergantung bertali di pojok), huruf stiker, satu take kamera,
musik prosedural + SFX kertas. Jangan pernah mengubah gaya itu untuk "memperbagus". Yang berubah per video hanya
**isi, objek, preset latar, dan pengaturan format**.
Balas dalam bahasa user (default Bahasa Indonesia), singkat dan praktis.

Output: MP4 (default 9:16, 720x1280, 30 fps) + spec.json + preset.json (+ SRT kalau ada naskah).

Buat task list. Kerjakan langsung; satu-satunya pertanyaan yang boleh adalah konfirmasi format (langkah 1) dan naskah kalau tidak bisa didapat.

## 0. Siapkan mesin (sekali per percakapan)

Mesin render ada di folder `engine/` di dalam skill ini.

1. `P=<scratchpad>/papercut-<slug>`; `mkdir -p $P/job`.
2. Cari folder skill: `find / -path "*motion-papercut*" -name papercut.py 2>/dev/null | head -1`. Salin folder `engine`
   yang memuat file itu: `cp -r <folder skill>/engine $P/`. Mesin = `E=$P/engine/papercut.py`.
   Contoh spec lengkap: `engine/examples/ai_belajar.json` (9:16, tanpa narasi, timeline detik) dan
   `engine/examples/daun_16x9.json` (16:9, naskah + caption).
3. Kalau `papercut.py` tidak ditemukan: minta user mengunggah ulang skill dari
   https://github.com/mujays/motion-papercut (Code → Download ZIP). Jangan menulis ulang mesin sendiri.
4. Baca `SPEC_REFERENCE.md` sampai habis sebelum menulis spec (katalog objek, aksi, layout per rasio, aturan gaya).
5. Butuh Python 3 + numpy, Pillow, opencv (`cv2`), ffmpeg. Kalau `cv2` tidak ada: `pip install --break-system-packages opencv-python-headless`.

## 1. FORMAT DULU, sebelum generate apa pun

User bisa menempel blok seperti ini (urutan bebas, boleh sebagian):

```
FORMAT VIDEO
Rasio      : 9:16            (9:16 | 16:9 | 1:1 | 4:5 | portrait | landscape | square)
Kualitas   : 720p            (720p | 1080p)
FPS        : 30              (24 | 30 | 60)
Subtitle   : otomatis        (otomatis = nyala kalau ada narasi | ya | tidak)
  Posisi   : bawah           (bawah | tengah | atas)
  Kata     : 3               (1–5 kata per tampilan)
  Highlight: ya              (kata kunci coral: ya | tidak)
  Ukuran   : sedang          (kecil | sedang | besar)
Judul      : ya              (papan kertas gantung → chip topik ; "pop" = gaya huruf pop lama)
Outro      : ya              (strip takeaway "A ➜ B")
CTA        : FOLLOW UNTUK LAINNYA   (teks sendiri | tidak)
Musik      : otomatis        (otomatis | tidak | nama file musik sendiri)
  Mood     : playful         (playful | upbeat | calm | curious)
  Volume   : normal          (pelan −4 | normal 0 | keras +3)
SFX        : ya, normal      (ya | tidak ; pelan | normal | keras)
Watermark  : tidak           (@akun + kanan-atas/kiri-atas/kanan-bawah/kiri-bawah | tidak)
Latar      : night           (night | dusk | space | deep | day | kraft | mint | peach)
Bahasa     : Indonesia       (Indonesia | Inggris)
Input      : topik           (audio | audio+naskah | audio+srt | naskah | topik)
```

Aturan:
- **Ada blok** → baca, isi yang kosong dengan default, tampilkan pengaturan final dalam SATU baris ringkas, lanjut tanpa bertanya.
- **Tidak ada blok** → tampilkan default sebagai daftar pendek, tanya SEKALI: "Pakai pengaturan ini, atau ada yang mau diubah?"
  (pakai AskUserQuestion kalau ada). Kalau user jelas tidak sedang menunggu, pakai default dan sebutkan.
- **User melampirkan preset JSON** → itulah pengaturannya.
- Simpan pengaturan final ke `$P/job/preset.json` (objek `format`). Pakai lagi untuk video berikutnya di percakapan ini.

Pemetaan ke `"format"` di spec:

| Blok | spec.format |
|---|---|
| Rasio | `"aspect": "9:16"` (juga "16:9", "1:1", "4:5") |
| Kualitas / FPS | `"quality": 720` / `1080`, `"fps": 30` |
| Subtitle otomatis | jangan isi `captions` (mesin: nyala kalau ada audio narasi, mati kalau tidak) |
| Subtitle ya/tidak + detail | `"captions": {"show": true, "position": "bottom", "words": 3, "highlight": true, "size": "sedang", "uppercase": false}` / `"captions": false` |
| Judul / Outro | `"title": true`, `"outro": true` (chip topik: `meta.chip` atau `"chip": false`) |
| CTA | `"cta": "TEKS"`; `"cta": false` = tanpa |
| Musik | `"music": "auto"` / `false` / `"bgm.mp3"`; `"music_mood": "playful"`; `"music_volume_db": 0` |
| SFX | `"sfx": true`, `"sfx_volume_db"`: pelan −6, normal 0, keras +3 |
| Watermark | `"watermark": "@akun", "watermark_pos": "bottom-right"` |
| Latar | `spec.background` = preset (default `night`; pilih sesuai topik bila user bilang otomatis) |
| Bahasa | `spec.lang` = "id"/"en"; semua teks di video ditulis dalam bahasa itu |

## 2. Input → mode

Simpan file user ke `$P/job/` (`voice.mp3`, `captions.srt`, `bgm.mp3`). Konversi audio lain ke mp3 dengan ffmpeg.

- **Ada audio voice over (dengan atau tanpa naskah)** → SELALU transkrip dulu dengan **Whisper "base"** (model
  OpenAI, versi ONNX dari proyek sherpa-onnx). Model berjalan offline di sini lewat `onnxruntime`
  (`engine/whisper_ort.py`), tanpa torch dan tanpa paket sherpa-onnx:
  ```bash
  pip install --break-system-packages -q onnxruntime
  mkdir -p $P/asr && curl -sSL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2 | tar xj -C $P/asr
  python3 $E $P/job/voice.mp3 --transcribe $P/asr/sherpa-onnx-whisper-base --lang id -o $P/job/transcript.txt \
          [--compare $P/job/naskah.txt]
  ```
  `--lang` = bahasa VO (`id`, `en`, … atau `auto`). Kalau user memberi naskah, simpan dulu sebagai `naskah.txt` dan
  pakai `--compare`: mesin menampilkan persentase kecocokan kata dan daftar bedanya.
  Cara memakai hasilnya:
  - **Ada naskah** → teks caption tetap dari naskah (ejaan benar). Baca daftar beda:
    salah dengar Whisper (mis. "korofil"/"klorofil", "pahun"/"pohon", kata nempel/terpisah) → abaikan.
    Kalimat/frasa yang benar-benar tidak diucapkan, ditambah, atau diganti di VO → sesuaikan naskah dengan yang
    DIUCAPKAN (ejaan tetap rapi) supaya sinkronnya pas, lalu beri tahu user bagian mana yang berbeda.
  - **Tanpa naskah** → transkrip jadi naskah setelah kamu koreksi: ejaan, nama, istilah, angka, tanda baca akhir
    kalimat (jeda kalimat dipakai untuk sinkron). Tunjukkan hasil koreksinya ke user dalam satu blok singkat.
  Hasil akhirnya jadi `"script"` + `"audio"`: caption dan beat sinkron otomatis, suara ikut dimix, musik otomatis
  dipelankan saat ada suara.
  Whisper base cepat (±10–15 detik untuk VO 30–50 detik) tapi untuk bahasa Indonesia sering salah eja nama/istilah;
  kalau transkrip tanpa naskah terlalu berantakan, pakai `sherpa-onnx-whisper-small` (±640 MB, lebih akurat) dengan
  perintah yang sama. Kalau download model gagal, minta naskah atau SRT. Jangan pernah mengarang transkrip.
- **Audio + SRT** → `"audio"` + `"srt"`, tetap isi `"script"` dengan teks polos.
- **Naskah saja** → `"script"`, timing diperkirakan; video punya musik + SFX tapi tanpa suara narasi. Sebutkan ini dan
  tawarkan sinkron ulang setelah user merekam VO dari naskah yang persis sama. Caption default mati (stiker yang bercerita) kecuali user minta.
- **Topik saja** → tulis naskah pendek dulu (45–90 detik, ±100–200 kata): dibuka pertanyaan penasaran, mekanisme
  langkah demi langkah dengan angka/perbandingan konkret, ditutup takeaway "A → B". Tampilkan, lalu lanjut (mode naskah).
- **Mode timeline (tanpa naskah sama sekali, mis. "video 15 detik seperti referensi")** → beat pakai detik (`"at": 4.3`) dan isi `"duration"`.

## 3. Rancang spec (`$P/job/spec.json`)

1. Masukkan `"format"` dari langkah 1 dan `"design_aspect"` = rasio yang dirancang.
2. Pecah cerita jadi 3–6 adegan. Tiap adegan = satu benda kertas besar yang jelas (prop/svg) + alat bercerita:
   angka → `counter` + `count`; data mengalir → `conveyor` + `flow` kartu `photo`; tebak/cek → `bubble` + `ring` + `mark`;
   perbaikan → panah merah `arrow` + `draw`; progres → `bars` + `grow` + counter persen + `burst`; istilah → `label` (1–2 kata).
3. Satu dunia: `landscape` di belakang (horizon 9:16 ≈ y 880–920; 16:9 ≈ camY+190), adegan disusun menyamping
   (±940 satuan per adegan di 9:16), pindah adegan dengan `camera` pan. Bulan/matahari pakai `parallax` 0.3–0.4.
4. Intro = papan kertas gantung (default; `meta.title_style: "pop"` untuk gaya lama). Papan pergi di ±2,8 dtk, atau
   tepat di akhir kalimat pertama kalau ada narasi. Sampai saat itu langit harus kosong; objek adegan pertama muncul setelahnya. Jangan taruh objek penting di
   kiri atas (chip), di zona caption (kalau nyala), atau di bawah 78 % layar saat outro.
5. Beat: anchor dari kata naskah (disalin persis, berurutan) atau detik. Perubahan tiap 1,5–3 dtk; tidak boleh diam > 5 dtk.
6. `meta.title` = hook 1–2 baris pendek; `meta.outro` = `["sebab", "akibat"]` (jadi "A ➜ B"); `meta.cta`.
7. `highlight` = 10–30 kata kunci (mode narasi/caption).
8. Ilustrasi yang tidak ada di pustaka prop → `svg` gaya kertas (isian flat, tanpa outline/teks, ≤ 10 bentuk, palet di referensi).

## 4. Cek → pratinjau → render

```bash
python3 $E $P/job/spec.json --check      # perbaiki semua "!"; baris pertama = format & mode
python3 $E $P/job/spec.json --preview 1,3,5,8,11,14,17,20   # -> $P/job/spec_preview.png
```
Buka `spec_preview.png` dengan Read dan PERIKSA: objek menabrak judul/chip/caption/strip outro? label keluar layar
atau bertumpuk? panah menembus benda? adegan kosong? framing salah untuk rasionya? Perbaiki, pratinjau lagi (maks 2 putaran). Lalu:
```bash
python3 $E $P/job/spec.json -o $P/job/video.mp4
```
Waktu render ±1,7× durasi di 720p (±3× di 1080p/60 fps). Video panjang (> 60 dtk): `--part 1/3`, `--part 2/3`,
`--part 3/3`, lalu `--finalize 3` (dengan `-o` yang sama).
Versi rasio lain ("buat juga versi 1:1"): salin spec, ubah `format.aspect` saja, biarkan `design_aspect`; mesin
otomatis menyesuaikan kamera. Pratinjau, ubah posisi hanya kalau perlu.

## 5. Serahkan

Kirim dengan SendUserFile: `video.mp4` (utama), `spec.json`, `preset.json`, dan `video.srt` kalau ada naskah.
Satu–dua kalimat: durasi, format (mis. "9:16 · 720p · tanpa caption · musik playful"), mode, adegan. Tawarkan revisi.

## 6. Revisi

Ubah hanya yang diminta, jalankan `--check`, render ulang. Sisanya identik supaya seri konsisten. Edit kode mesin
hanya kalau error, lalu beri tahu user apa yang diubah.
