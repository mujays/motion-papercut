# PAPERCUT ENGINE — Spec Reference (baca sampai habis sebelum menulis spec)

Mesin mengubah SATU spec JSON jadi MP4 bergaya **papercraft / kertas guntingan** (default 9:16, 720x1280, 30 fps).

**Konsep desain yang dikunci (jangan diubah):** semua elemen = potongan kertas (warna flat + tekstur serat halus +
tepi potongan sedikit terang + bayangan lembut). Langit kertas bertekstur dengan bintang kertas kecil, lapisan bukit
dengan tepi sobekan putih + pohon cemara, huruf stiker tebal, intro **papan kertas gantung** (kartu krem bertali jatuh
dan berayun, kata judul jatuh satu per satu dengan pantulan squash, konfeti kertas, lalu papan ditarik ke atas dan chip
topik tergantung bertali di kiri atas), tag kertas, kartu polaroid, counter
angka, grafik batang, starburst, strip takeaway di akhir. SATU take kamera (pan/zoom, bukit punya parallax).
Musik latar prosedural + SFX kertas. Narasi opsional (sinkron otomatis + caption strip kertas).

**Bebas per topik:** latar (preset), apa yang digambar, warna objek, isi teks.

## 1. Kerangka spec

```json
{
  "lang": "id",
  "background": "night",              // night | dusk | space | deep | day | kraft | mint | peach  (atau {"top":"#hex","bottom":"#hex","land":[4 warna],"tree":"#hex","stars":80})
  "format": {"aspect": "9:16", "quality": 720, "fps": 30,
             "captions": {"show": true, "position": "bottom", "words": 3, "highlight": true, "size": "sedang", "uppercase": false},
             "title": true, "outro": true, "cta": "FOLLOW UNTUK LAINNYA", "chip": true,
             "music": "auto", "music_mood": "playful", "music_volume_db": 0,
             "sfx": true, "sfx_volume_db": 0, "watermark": "@akun", "watermark_pos": "top-right"},
  "design_aspect": "9:16",
  "meta": {"title": ["Bagaimana AI", "belajar?"],          // hook 1–2 baris (≤ 14 huruf/baris enak dibaca)
           "outro": ["Banyak contoh", "tebakan makin jitu"], // 2 bagian → "A ➜ B" di strip kertas; 1 string = teks biasa
           "cta": "FOLLOW UNTUK LAINNYA",
           "chip": true,                                     // true = teks judul jadi chip; "teks lain"; false = tanpa chip
           "title_style": "sign",                            // "sign" (default: papan gantung) | "pop" (huruf pop lalu terbang jadi chip)
           "title_dur": null, "outro_dur": 3.6},             // title_dur kosong = otomatis (narasi: papan pergi saat kalimat pertama selesai)
  "script": "naskah lengkap…",        // wajib kalau ada audio/caption; opsional di mode timeline
  "audio": "voice.mp3",               // opsional → mode narasi (sinkron otomatis; WAJIB bersama "script")
  "srt": "captions.srt",              // opsional: timing persis
  "highlight": ["klorofil", "matahari"],
  "duration": 23,                     // mode timeline (tanpa naskah): panjang video. Kalau kosong dihitung dari beat terakhir
  "camera": {"x": 360, "y": 640, "zoom": 1.0},
  "world": [ … ],
  "beats": [ … ]
}
```

**Musik:** `"music": "auto"` = musik prosedural (mode tanpa narasi: keras ±−10 dB; dengan narasi: pelan −23 dB dan
otomatis "ducking" saat ada suara). `"music": "bgm.mp3"` = file sendiri (di-loop, fade di akhir). `false` = tanpa musik.
Mood: `playful` (default, ceria pluck + music box + shaker), `upbeat`, `calm`, `curious`.

**Caption:** default NYALA kalau ada audio narasi, MATI kalau tidak ada narasi (cerita dibawa stiker).
Bisa dipaksa dengan `format.captions`. Caption = strip kertas krem, teks navy, kata kunci coral.

## 2. Mode waktu
| input | mode | anchor beat |
|---|---|---|
| audio + script | `audio` | kata dari naskah (`"at": "mengikis batuan"`) — naskah dari transkrip Whisper yang sudah dikoreksi |
| audio + srt | `srt` | kata naskah |
| script saja | `script` | kata naskah (timing diperkirakan, tanpa suara narasi) |
| tanpa script | `timeline` | **detik** (`"at": 4.3`). Isi `duration` |

Anchor lain: `"start"`, `"end"`, `"outro"`, angka detik, `"^frasa"` (akhir frasa), sufiks `"+0.4s"`/`"-0.3s"`, `"#12"` (indeks kata).

### 2b. Transkrip VO (sama persis dengan motion-2d-explainer)
* Model: Whisper "base" (OpenAI), versi ONNX dari proyek sherpa-onnx, dijalankan offline lewat `onnxruntime`
  (`whisper_ort.py`), tanpa torch dan tanpa paket sherpa-onnx.
* `python papercut.py voice.mp3 --transcribe <folder-model> --lang id -o job/transcript.txt [--compare job/naskah.txt]`
  → audio dipotong di jeda jadi bagian ≤ 25 dtk, ditranskrip, ditulis ke `transcript.txt`. `--compare` menampilkan
  persen kecocokan kata dan daftar beda (`replace/delete/insert  audio: '…' | naskah: '…'`). `--lang auto` = deteksi bahasa.
* Hasil akhirnya selalu `"script"` (naskah benar/terkoreksi) + `"audio"`; sinkron caption & beat memakai penyelaras jeda
  di `explainer.py` (jeda kalimat = jangkar, jadi tanda baca akhir kalimat penting).

## 3. Koordinat, kamera, layout
* Satuan dunia = piksel kanvas 720x1280. Kamera `{x,y}` = titik dunia di tengah layar; zoom 1 di 9:16 memperlihatkan
  x 0…720, y 0…1280. Area terlihat per rasio pada zoom 1: 9:16 → 720×1280 · 4:5 → 858×1073 · 1:1 → 960×960 · 16:9 → 1280×720.
* **Layout 9:16 (standar):** `landscape` dengan horizon y ≈ 880–920. Langit y 100–800 untuk cerita.
  Zona judul (papan gantung): layar y 15–45 % dari 0 dtk sampai papan ditarik (±2,8 dtk; mode narasi = akhir kalimat
  pertama). Munculkan objek adegan pertama setelahnya. Chip tergantung di kiri atas (layar y 0–9 %) sepanjang video. Strip outro + CTA di layar y 78–92 % saat outro.
  Kalau caption nyala: layar y 74–84 % dipakai caption.
* **16:9:** horizon ≈ camY + 190 (mis. kamera y 640 → horizon 830). Cerita disusun menyamping di sepanjang x, kamera
  pan ke kanan antar adegan. Caption di ±85 % tinggi → objek penting di atas ±75 %.
* **1:1 / 4:5:** horizon ≈ camY + 200…240.
* Pan antar adegan: taruh adegan berikutnya 900–1000 satuan ke kanan (9:16) lalu `camera focus [x+940, 640]`.
  Buat `landscape` cukup lebar (`x1` ≥ x adegan terakhir + 700).
* Zoom normal 1.0–1.6. Lebih dari 2 jarang perlu (gaya ini datar seperti diorama).

## 4. Objek dunia
Setiap objek boleh punya: `id`, `x`, `y` (pusat), `hidden: true` (muncul lewat `show`), `enter` (gaya muncul default),
`rotate` (derajat), `scale`, `sx`/`sy` (skala sumbu), `alpha`, `bob` (melayang px), `wobble` (goyang derajat),
`parallax` (0–1; 1 = ikut kamera penuh, 0.3 = jauh seperti bulan), `layer` (urutan gambar; default 0),
`screen: true` + `pos: [fx, fy]` (HUD, fraksi layar 0–1, tidak ikut kamera).
Semua angka bisa dianimasikan dengan `set`.

| type | parameter (default) | catatan |
|---|---|---|
| **landscape** | y=900 (horizon), x0=-700, x1=1420, layers=4, trees=true, colors [4], seed | jadi 4 lapis bukit: belakang (parallax .55), tengah + cemara (.78), depan sobek + rumput, latar depan sobek. Taruh PERTAMA (setelah bulan/matahari) |
| hill | x0, x1, y, amp=40, wave=520, bottom, color, torn, trees, tree_color, grass, edge, parallax, seed | satu lapis bukit manual |
| shape | shape rect\|rrect\|circle\|ellipse\|poly\|blob\|path, w, h, r, radius, points, fill, sw (path), torn | potongan kertas bebas (blob = kurva halus) |
| prop | prop, w=120, flip, recolor {"#hex":"warna"} | pustaka: brain cat dog pine tree cloud moon sun star house person heart bulb coin book phone gear rocket drop leaf clock magnifier laptop flag trophy box atom car fish planet (+ lightbulb battery cell chart cup food house2 tree2 sun2 person2) |
| svg | svg "<svg viewBox='0 0 100 100'>…</svg>", w | ilustrasi apa pun (§4b) |
| **text** | text ("\n" = baris baru), size=64, colors "cycle" \| "teal" \| ["coral","yellow"], outline, font (default stiker \| "heavy" \| "semi"), letters=true, wave (bob per huruf), jitter=1 | huruf stiker; `show` = pop huruf satu per satu |
| **tag** | text, size=24, bg cream, ink ink, shape pill\|tag\|rect | label kertas kecil |
| **photo** | w=110, icon (nama prop) atau svg, bg lightblue, tag "teks", tag_color red, frame cream | kartu polaroid + tag kecil |
| bubble | text, size=34, tail left\|right\|down\|none, bg, ink | balon bicara/tebakan |
| arrow | x,y (pangkal), to [x,y] atau points [[x,y]…], bend=0.18 (+ = melengkung ke KIRI arah jalan: panah ke kanan melengkung ke atas, panah ke bawah melengkung ke kanan), width=12, color yellow, head, draw 0–1 | strip panah kertas; `draw` = animasi menggambar |
| ring | r / rx, ry, width=10, color yellow, draw | lingkaran sorot |
| mark | kind check\|cross, size=60, color | centang hijau / silang merah |
| burst | r=80, points=16, inner=.74, color yellow, text, text_color coral, spin=6 | starburst lencana |
| **counter** | value, size=56, color yellow, prefix, suffix "contoh", suffix_color teal, suffix_size, decimals, sep ("." id / "," en), align left\|center\|right | angka stiker; animasi dengan aksi `count` |
| **bars** | x,y = sudut kiri-bawah, w=440, h=320, values [0–1…], colors [yellow, teal], highlight_last=true (coral), label, grow 0–1 | grafik batang; animasi `grow` |
| conveyor | x,y = ujung kiri, w=600, h=22, color, roller_color yellow, speed=60, legs=3 | ban berjalan, titik kuning bergerak |
| dials | n=3, r=22, color, angle, angles [..], jitter=3 | panel kenop/"parameter model" |
| strip | text ("A -> B" atau ["A","B"]), size=30, ink, hl, bg | strip kertas takeaway (biasanya `screen`) |

Warna: `coral yellow teal pink cream ink(navy) red green blue lightblue lavender purple slate forest moss white orange
brown grey sky gold mint peach kraft dark rose sand` atau `"#RRGGBB"`.

### 4b. SVG gaya kertas
* viewBox bebas, objek dipusatkan di x,y dengan lebar `w`.
* HANYA isian flat. Tanpa outline gelap, tanpa gradient, tanpa teks. Tiap bentuk = satu lembar kertas; mesin otomatis
  memberi bayangan kecil antar lapis, jadi urutan bentuk = urutan tumpukan kertas (belakang dulu).
* Stroke boleh untuk "strip kertas" (garis tebal, `stroke-linecap="round"`).
* Palet: coral #E8775F, kuning #F2C552, toska #74C6B4, pink #EE9AA8, krem #F4EEDC, navy #3B3A5C, merah #D65448,
  hijau #5EAA6A, biru muda #A8D4E6, lavender #7D76AE, oranye #EE964A, coklat #966848, kulit #F2C4A0.
* ≤ 10 bentuk. Siluet besar yang jelas.

## 5. Beats

```json
{"at": 4.3, "do": [aksi, aksi…]}        // atau "at": "kata dari naskah"
```
Setiap aksi boleh `delay` (detik) dan `sfx` (nama SFX atau null untuk diam).

| aksi | field | SFX otomatis |
|---|---|---|
| camera | focus [x,y] / id, zoom, dur=1.3, ease io\|out\|back\|linear\|in\|bounce | whoosh lembut |
| show | id / [ids], stagger=0.12, dur, enter pop\|slide-up\|slide-down\|slide-left\|slide-right\|drop\|grow\|fade | pop / tap (tag) / paper (slide) / stamp (burst) / correct\|wrong (mark) / pop-pop huruf (text) |
| hide | id / [ids], dur=0.35, exit shrink\|fade\|slide-* | – |
| set | id, props {kunci: nilai}, dur=0.8, ease | (isi `sfx`) |
| draw | id (arrow/ring), dur=0.7 | swish |
| grow | id (bars), dur=2, to=1 | pop per batang |
| count | id (counter), to, dur=1.6, ease out\|linear | tik-tik melambat + ding |
| pulse | target id, scale=0.16 | softpop |
| label | label "teks", target id/[x,y], offset [dx,dy] (default [0,-70]), hold=2.6 / until, rotate=-3, style pill\|tag\|rect, bg, ink | tap |
| bigtext | bigtext "teks", pos [fx,fy] (default [0.5,0.42]), size=92, colors, hold=2 | stamp |
| takeaway | takeaway "A -> B", y=0.84, hold=3 | paper |
| flow | items [objek…] atau template id, from, to, every=0.5, count=6 / until, dur=2.4, arc, enter, vanish=true | tap |
| swap | swap [id_lama, id_baru] | pop |
| shake | amp=8, dur=0.45 | – |
| sfx | sfx nama, gain dB | – |

SFX: pop softpop tap paper stamp snip tick ding chime correct wrong whoosh swish sparkle bubble riser blip rise typing boom.

## 6. Aturan gaya (wajib = konsistensi seri)
1. Satu dunia, satu take. Adegan berpindah dengan pan kamera menyamping (atau zoom), tidak ada potongan.
2. Buka dengan judul hook di papan gantung di langit kosong; objek adegan pertama muncul SETELAH papan ditarik.
3. Ada perubahan tiap 1,5–3 dtk (show, label, count, draw, flow, camera). Tidak boleh diam > 5 dtk.
4. Tiap konsep = satu "benda kertas" besar yang jelas (otak, pohon, rumah…) + tag kecil 1–2 kata.
5. Angka = `counter` yang menghitung naik/turun. Perbandingan/progres = `bars` + counter persen + `burst` di puncak.
   Benar/salah = bubble + ring + mark. "Banyak data mengalir" = `flow` kartu photo di atas conveyor.
6. Tutup dengan strip takeaway "A ➜ B" + CTA (otomatis dari `meta.outro`).
7. Tanpa foto, tanpa emoji, tanpa logo/karakter terkenal. Ilustrasi sendiri pakai prop/svg gaya kertas.
8. Huruf stiker untuk kata pendek (≤ 3 kata). Teks panjang → tag/strip/bubble.

## 7. Jalankan
```bash
python papercut.py job/spec.json --check                 # anchor, id, jeda > 5 dtk
python papercut.py job/spec.json --preview 1,3,6,9,12    # lembar kontak PNG — LIHAT
python papercut.py job/spec.json -o job/video.mp4        # render (±1,7× durasi di 720p)
python papercut.py job/spec.json -o job/video.mp4 --part 1/3 … --finalize 3   # video panjang
python papercut.py voice.mp3 --transcribe <folder-model> --lang id -o job/transcript.txt [--compare job/naskah.txt]
```
Contoh lengkap: `examples/ai_belajar.json` (9:16, timeline, tanpa narasi) dan `examples/daun_16x9.json` (16:9, naskah + caption).
