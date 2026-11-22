"""Offline speech-to-text: OpenAI Whisper "base" (ONNX export from the sherpa-onnx project) run with onnxruntime.

Model files (from https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2):
  base-encoder.int8.onnx  (or base-encoder.onnx)
  base-decoder.int8.onnx  (or base-decoder.onnx)
  base-tokens.txt
Needs only: numpy + onnxruntime (pip install onnxruntime). No torch, no sherpa-onnx package.

Usage:
  from whisper_ort import WhisperORT
  w = WhisperORT("/path/to/sherpa-onnx-whisper-base")
  text = w.transcribe(samples_16k_float32, lang="id")     # <= 30 s per call
"""
import base64
import glob
import os

import numpy as np

SR = 16000
N_FFT = 400
HOP = 160
CHUNK = 30 * SR          # Whisper window: 30 s -> 3000 mel frames -> 1500 encoder frames


# ----------------------------------------------------------------------------- features (== openai/whisper audio.py)
def _hz_to_mel(f):
    f = np.asanyarray(f, float)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asanyarray(m, float)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filters(n_mels=80, sr=SR, n_fft=N_FFT):
    """librosa.filters.mel(sr, n_fft, n_mels) with Slaney scale + Slaney norm (what Whisper uses)."""
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    return (w * enorm[:, None]).astype(np.float32)


_MEL = None


def log_mel(audio, n_mels=80):
    """audio: float32 16 kHz, padded/trimmed to 30 s -> (n_mels, 3000)."""
    global _MEL
    if _MEL is None or _MEL.shape[0] != n_mels:
        _MEL = mel_filters(n_mels)
    a = np.zeros(CHUNK, np.float32)
    a[:min(len(audio), CHUNK)] = audio[:CHUNK]
    pad = N_FFT // 2
    x = np.pad(a, (pad, pad), mode="reflect")                      # torch.stft(center=True)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic hann
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(x[idx] * window, axis=1)
    mag = (np.abs(spec) ** 2).T[:, :-1]                             # drop last frame -> 3000
    m = _MEL @ mag
    logs = np.log10(np.maximum(m, 1e-10))
    logs = np.maximum(logs, logs.max() - 8.0)
    return ((logs + 4.0) / 4.0).astype(np.float32)


# ----------------------------------------------------------------------------- model
class WhisperORT:
    def __init__(self, model_dir, threads=2):
        import onnxruntime as ort

        def pick(kind):
            c = sorted(glob.glob(os.path.join(model_dir, f"*{kind}*.onnx")), key=lambda p: ("int8" not in p, p))
            if not c:
                raise FileNotFoundError(f"no *{kind}*.onnx in {model_dir}")
            return c[0]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(pick("encoder"), so, providers=prov)
        self.dec = ort.InferenceSession(pick("decoder"), so, providers=prov)
        meta = self.enc.get_modelmeta().custom_metadata_map
        self.n_mels = int(meta.get("n_mels", 80))
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])
        self.sot = int(meta["sot"])
        self.eot = int(meta["eot"])
        self.transcribe_tok = int(meta["transcribe"])
        self.no_ts = int(meta["no_timestamps"])
        self.blank = int(meta.get("blank_id", 220))
        codes = meta["all_language_codes"].split(",")
        toks = [int(t) for t in meta["all_language_tokens"].split(",")]
        self.lang2tok = dict(zip(codes, toks))
        tok_file = glob.glob(os.path.join(model_dir, "*tokens.txt"))[0]
        self.id2bytes = {}
        with open(tok_file, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    self.id2bytes[int(p[1])] = base64.b64decode(p[0])

    def detect_language(self, ck, cv):
        """Optional: pick the most likely language token after <sot>."""
        logits = self._step(np.array([[self.sot]], np.int64), *self._empty_cache(), ck, cv, 0)[0]
        cand = np.array(list(self.lang2tok.values()))
        best = cand[np.argmax(logits[0, -1, cand])]
        return {v: k for k, v in self.lang2tok.items()}[int(best)]

    def _empty_cache(self):
        shape = (self.n_text_layer, 1, self.n_text_ctx, self.n_text_state)
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def _step(self, tokens, k_cache, v_cache, ck, cv, offset):
        return self.dec.run(None, {
            "tokens": tokens, "in_n_layer_self_k_cache": k_cache, "in_n_layer_self_v_cache": v_cache,
            "n_layer_cross_k": ck, "n_layer_cross_v": cv, "offset": np.array([offset], np.int64)})

    def transcribe(self, audio, lang="id", max_tokens=220):
        mel = log_mel(np.asarray(audio, np.float32), self.n_mels)[None]
        ck, cv = self.enc.run(None, {"mel": mel})
        if lang in (None, "", "auto"):
            lang = self.detect_language(ck, cv)
        lang_tok = self.lang2tok.get(lang, self.lang2tok.get("en"))
        prompt = [self.sot, lang_tok, self.transcribe_tok, self.no_ts]
        k_cache, v_cache = self._empty_cache()
        logits, k_cache, v_cache = self._step(np.array([prompt], np.int64), k_cache, v_cache, ck, cv, 0)
        offset = len(prompt)
        out = []
        last = logits[0, -1].copy()
        last[self.blank] = -np.inf                  # suppress a leading blank (Whisper's SuppressBlank)
        for _ in range(max_tokens):
            nxt = int(np.argmax(last))
            if nxt == self.eot or nxt >= self.eot:  # eot or any special token ends the segment
                break
            out.append(nxt)
            # stop on obvious repetition loops
            if len(out) > 24 and out[-12:] == out[-24:-12]:
                out = out[:-12]
                break
            logits, k_cache, v_cache = self._step(np.array([[nxt]], np.int64), k_cache, v_cache, ck, cv, offset)
            offset += 1
            if offset >= self.n_text_ctx - 1:
                break
            last = logits[0, -1]
        return b"".join(self.id2bytes.get(t, b"") for t in out).decode("utf-8", errors="ignore").strip()
