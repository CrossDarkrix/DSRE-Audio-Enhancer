import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QTextCursor, QDragEnterEvent, QDropEvent, QKeySequence, QAction


def _decode_subprocess_output(data: bytes) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    for enc in ("utf-8", "cp932", "utf-16-le", "utf-16-be"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _run_subprocess(cmd, check=False, capture_stdout=False, capture_stderr=False):
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )
    result.stdout_text = _decode_subprocess_output(result.stdout if capture_stdout else b"")
    result.stderr_text = _decode_subprocess_output(result.stderr if capture_stderr else b"")

    if check and result.returncode != 0:
        err = subprocess.CalledProcessError(result.returncode, cmd)
        err.stdout = result.stdout
        err.stderr = result.stderr
        err.stdout_text = result.stdout_text
        err.stderr_text = result.stderr_text
        raise err

    return result

def add_ffmpeg_to_path():
    ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    if os.path.isdir(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir


add_ffmpeg_to_path()


def get_ffmpeg_executable() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local = os.path.join(os.path.dirname(__file__), "ffmpeg", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    raise FileNotFoundError("FFmpeg not found. Place ffmpeg.exe in the 'ffmpeg' folder next to this script.")


def get_ffprobe_executable() -> str:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    local = os.path.join(os.path.dirname(__file__), "ffmpeg", "ffprobe.exe")
    if os.path.exists(local):
        return local
    raise FileNotFoundError("FFprobe not found. Place ffprobe.exe in the 'ffmpeg' folder next to this script.")

def ensure_ch_first(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y[np.newaxis, :]
    if y.ndim == 2:
        # If likely (samples, channels), transpose to (channels, samples)
        if y.shape[0] > y.shape[1]:
            return y.T
        return y
    raise ValueError(f"Unsupported audio shape: {y.shape}")


def ensure_sf_shape(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2:
        # If likely (channels, samples), transpose to (samples, channels)
        if y.shape[0] <= y.shape[1]:
            return y.T
        return y
    raise ValueError(f"Unsupported audio shape: {y.shape}")


def sanitize_audio(x: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    if x is None:
        if fallback is not None:
            return fallback.copy().astype(np.float32)
        raise ValueError("Audio data is None")

    x = np.asarray(x)
    if x.size == 0:
        if fallback is not None:
            return fallback.copy().astype(np.float32)
        raise ValueError("Audio data is empty")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1000.0:
        x = x / peak
    return x


def audio_peak(x: np.ndarray) -> float:
    if x is None or x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def audio_rms(x: np.ndarray) -> float:
    if x is None or x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))

def ffprobe_audio_info(file_path: str) -> Dict[str, Any]:
    ffprobe = get_ffprobe_executable()
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        file_path,
    ]
    result = _run_subprocess(cmd, capture_stdout=True)
    data = json.loads(result.stdout)

    audio_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio":
            audio_stream = s
            break

    if not audio_stream:
        raise ValueError(f"No audio stream found: {file_path}")

    return {
        "sample_rate": int(audio_stream.get("sample_rate", 0) or 0),
        "channels": int(audio_stream.get("channels", 0) or 0),
        "duration": float(audio_stream.get("duration", data.get("format", {}).get("duration", 0)) or 0.0),
        "codec_name": audio_stream.get("codec_name", ""),
        "bit_rate": int(audio_stream.get("bit_rate", data.get("format", {}).get("bit_rate", 0)) or 0),
    }


def load_audio_ffmpeg(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Decode audio via FFmpeg to float32 WAV at target_sr.
    Returns audio as (channels, samples), sr.
    """
    ffmpeg = get_ffmpeg_executable()
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()

    cmd = [
        ffmpeg, "-y",
        "-i", file_path,
        "-vn", "-sn", "-dn",
        "-map", "0:a:0",
        "-ar", str(target_sr),
        "-c:a", "pcm_f32le",
        tmp_wav.name,
    ]

    try:
        _run_subprocess(cmd, check=True, capture_stderr=True)
        y, sr = sf.read(tmp_wav.name, always_2d=True)
        y = y.T.astype(np.float32, copy=False)
        return sanitize_audio(y), sr
    finally:
        try:
            os.remove(tmp_wav.name)
        except Exception:
            pass


def extract_cover_image(in_path: str) -> Optional[str]:
    ffmpeg = get_ffmpeg_executable()
    cover_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cover_tmp.close()

    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-an",
        "-map", "0:v:0",
        "-frames:v", "1",
        cover_tmp.name,
    ]

    try:
        result = _run_subprocess(cmd)
        if result.returncode == 0 and os.path.exists(cover_tmp.name) and os.path.getsize(cover_tmp.name) > 0:
            return cover_tmp.name
    except Exception:
        pass

    try:
        os.remove(cover_tmp.name)
    except Exception:
        pass
    return None


def save_with_metadata(in_path: str, y_out: np.ndarray, sr: int, out_path: str, fmt: str = "ALAC", normalize: bool = True) -> str:
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if y_out is None or y_out.size == 0:
        raise ValueError("Empty audio data provided")
    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    ffmpeg = get_ffmpeg_executable()
    fmt = fmt.upper()
    if fmt not in ("ALAC", "FLAC", "MP3"):
        raise ValueError(f"Unsupported format: {fmt}")

    data = ensure_sf_shape(sanitize_audio(y_out)).astype(np.float32, copy=False)
    if normalize:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.0:
            data /= peak
    else:
        data = np.clip(data, -1.0, 1.0)

    if np.max(np.abs(data)) < 1e-10:
        raise ValueError("Audio data is essentially silent - cannot save")

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    cover_tmp = None

    try:
        sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

        codec_map = {"ALAC": "alac", "FLAC": "flac", "MP3": "libmp3lame"}
        ext_map = {"ALAC": "m4a", "FLAC": "flac", "MP3": "mp3"}
        sample_fmt_map = {"ALAC": "s32p", "FLAC": "s32", "MP3": "s16p"}
        out_path = os.path.splitext(out_path)[0] + "." + ext_map[fmt]

        cover_tmp = extract_cover_image(in_path)

        if fmt == "MP3":
            if cover_tmp:
                cmd = [
                    ffmpeg, "-y",
                    "-i", tmp_wav.name,
                    "-i", in_path,
                    "-i", cover_tmp,
                    "-map", "0:a",
                    "-map", "2:v",
                    "-map_metadata", "1",
                    "-id3v2_version", "3",
                    "-c:a", codec_map[fmt],
                    "-sample_fmt", sample_fmt_map[fmt],
                    "-b:a", "320k",
                    "-c:v", "mjpeg",
                    out_path,
                ]
            else:
                cmd = [
                    ffmpeg, "-y",
                    "-i", tmp_wav.name,
                    "-i", in_path,
                    "-map", "0:a",
                    "-map_metadata", "1",
                    "-c:a", codec_map[fmt],
                    "-sample_fmt", sample_fmt_map[fmt],
                    "-b:a", "320k",
                    out_path,
                ]
        else:
            if cover_tmp:
                cmd = [
                    ffmpeg, "-y",
                    "-i", tmp_wav.name,
                    "-i", in_path,
                    "-i", cover_tmp,
                    "-map", "0:a",
                    "-map", "2:v",
                    "-disposition:v", "attached_pic",
                    "-map_metadata", "1",
                    "-c:a", codec_map[fmt],
                    "-sample_fmt", sample_fmt_map[fmt],
                    "-c:v", "copy",
                    out_path,
                ]
            else:
                cmd = [
                    ffmpeg, "-y",
                    "-i", tmp_wav.name,
                    "-i", in_path,
                    "-map", "0:a",
                    "-map_metadata", "1",
                    "-c:a", codec_map[fmt],
                    "-sample_fmt", sample_fmt_map[fmt],
                    out_path,
                ]

        _run_subprocess(cmd, check=True, capture_stdout=True, capture_stderr=True)

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            raise RuntimeError(f"Output file was not created correctly: {out_path}")
        return out_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg command failed: {' '.join(cmd)}\nSTDERR:\n{e.stderr}\nSTDOUT:\n{e.stdout}")
    finally:
        try:
            os.remove(tmp_wav.name)
        except Exception:
            pass
        if cover_tmp and os.path.exists(cover_tmp):
            try:
                os.remove(cover_tmp)
            except Exception:
                pass

def apply_iir_filter(b, a, x):
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("apply_iir_filter expects 1D array")
    if len(a) == 0 or a[0] == 0:
        raise ValueError("Invalid IIR coefficients")

    b = b / a[0]
    a = a / a[0]

    y = np.zeros_like(x, dtype=np.float64)
    nb, na = len(b), len(a)
    for n in range(len(x)):
        acc = 0.0
        for i in range(nb):
            if n - i >= 0:
                acc += b[i] * x[n - i]
        for i in range(1, na):
            if n - i >= 0:
                acc -= a[i] * y[n - i]
        y[n] = acc
    return y.astype(np.float32)


def filtfilt_np(b, a, x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < max(len(a), len(b)) * 3:
        return apply_iir_filter(b, a, x)
    pad = min(len(x) - 1, max(len(a), len(b)) * 3)
    front = 2 * x[0] - x[1:pad + 1][::-1]
    back = 2 * x[-1] - x[-pad - 1:-1][::-1]
    xp = np.concatenate([front, x, back])
    y = apply_iir_filter(b, a, xp)
    y = apply_iir_filter(b, a, y[::-1])[::-1]
    return y[pad:pad + len(x)].astype(np.float32)


def design_peaking_eq(freq, gain_db, q, sr):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return np.array([b0, b1, b2], dtype=np.float64), np.array([a0, a1, a2], dtype=np.float64)


def bandpass_fft(x, sr, low_hz, high_hz, transition_ratio=0.15):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.astype(np.float32)

    low_hz = max(1.0, float(low_hz))
    nyquist = sr / 2.0
    high_hz = min(float(high_hz), nyquist - 1.0)
    if low_hz >= high_hz:
        return np.zeros_like(x, dtype=np.float32)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = np.zeros_like(freqs, dtype=np.float64)

    bw = max(20.0, (high_hz - low_hz) * transition_ratio)
    low1 = max(0.0, low_hz - bw)
    low2 = low_hz
    high1 = high_hz
    high2 = min(nyquist, high_hz + bw)

    rising = (freqs >= low1) & (freqs < low2)
    if np.any(rising):
        t = (freqs[rising] - low1) / max(1e-12, (low2 - low1))
        mask[rising] = 0.5 - 0.5 * np.cos(np.pi * t)

    passband = (freqs >= low2) & (freqs <= high1)
    mask[passband] = 1.0

    falling = (freqs > high1) & (freqs <= high2)
    if np.any(falling):
        t = (freqs[falling] - high1) / max(1e-12, (high2 - high1))
        mask[falling] = 0.5 + 0.5 * np.cos(np.pi * t)

    y = np.fft.irfft(X * mask, n=n)
    return y.astype(np.float32)

def generate_harmonics(signal_band, fundamental_freq, sr, num_harmonics=5, harmonic_strength=0.3):
    signal_band = sanitize_audio(signal_band)
    if len(signal_band) == 0:
        return signal_band

    enhanced = signal_band.astype(np.float32).copy()
    for h in range(2, num_harmonics + 2):
        harmonic_freq = fundamental_freq * h
        if harmonic_freq < sr / 2:
            phase_increment = 2 * np.pi * harmonic_freq / sr
            if not np.isfinite(phase_increment):
                continue
            harmonic_oscillator = np.sin(phase_increment * np.arange(len(signal_band), dtype=np.float64)).astype(np.float32)
            if not np.all(np.isfinite(harmonic_oscillator)):
                continue
            harmonic_content = signal_band * harmonic_oscillator * (harmonic_strength / h)
            if not np.all(np.isfinite(harmonic_content)):
                continue
            enhanced += harmonic_content
    return sanitize_audio(enhanced, fallback=signal_band)


def multiband_exciter(x, sr, harmonic_intensity=0.6, progress_cb=None, abort_cb=None):
    x = ensure_ch_first(x).astype(np.float32, copy=False)
    enhanced = np.zeros_like(x, dtype=np.float32)
    nyquist = sr // 2
    base_strength_scale = float(np.clip(harmonic_intensity, 0.1, 1.5))

    bands = []
    band_definitions = [
        {"name": "Sub Bass", "low": 20, "high": 80, "gain": 1.10, "harmonics": 3, "strength": 0.08},
        {"name": "Bass", "low": 80, "high": 250, "gain": 1.20, "harmonics": 4, "strength": 0.14},
        {"name": "Low Mid", "low": 250, "high": 800, "gain": 1.25, "harmonics": 5, "strength": 0.18},
        {"name": "Mid", "low": 800, "high": 2500, "gain": 1.35, "harmonics": 6, "strength": 0.22},
        {"name": "High Mid", "low": 2500, "high": 8000, "gain": 1.45, "harmonics": 4, "strength": 0.25},
        {"name": "Presence", "low": 8000, "high": 16000, "gain": 1.50, "harmonics": 3, "strength": 0.18},
        {"name": "Air", "low": 16000, "high": min(20000, nyquist - 1000), "gain": 1.35, "harmonics": 2, "strength": 0.12},
    ]
    for band in band_definitions:
        if band["low"] < nyquist and band["high"] < nyquist and band["high"] > band["low"]:
            bands.append(band)

    if not bands:
        return x.astype(np.float32)

    for ch in range(x.shape[0]):
        if abort_cb and abort_cb():
            break
        channel_enhanced = x[ch].copy()
        for i, band in enumerate(bands):
            if abort_cb and abort_cb():
                break
            if progress_cb:
                progress = int((i + ch * len(bands)) * 100 / (len(bands) * x.shape[0]))
                progress_cb(progress, f"Processing band {band['name']}")
            try:
                band_signal = bandpass_fft(x[ch], sr, band["low"], band["high"])
                band_signal = sanitize_audio(band_signal, fallback=np.zeros_like(x[ch], dtype=np.float32))
                if audio_peak(band_signal) < 1e-8:
                    continue
                center_freq = (band["low"] + band["high"]) / 2
                strength = band["strength"] * base_strength_scale
                harmonics_added = generate_harmonics(band_signal, center_freq, sr, band["harmonics"], strength)
                saturated = np.tanh(harmonics_added * 1.35).astype(np.float32) * 0.82
                band_enhanced = saturated * band["gain"]
                if not np.all(np.isfinite(band_enhanced)):
                    continue
                channel_enhanced = channel_enhanced + band_enhanced * 0.22
            except Exception:
                continue
        enhanced[ch] = sanitize_audio(channel_enhanced, fallback=x[ch])
    return enhanced


def psychoacoustic_enhancer(x, sr, strength=1.0, progress_cb=None, abort_cb=None):
    x = ensure_ch_first(x).astype(np.float32, copy=False)
    enhanced = np.zeros_like(x, dtype=np.float32)
    scale = float(np.clip(strength, 0.3, 1.5))

    critical_bands = [
        {"freq": 1000, "boost": 1.0 * scale, "q": 1.5},
        {"freq": 2500, "boost": 1.8 * scale, "q": 2.0},
        {"freq": 4000, "boost": 2.0 * scale, "q": 1.8},
        {"freq": 6000, "boost": 1.5 * scale, "q": 1.2},
        {"freq": 10000, "boost": 1.0 * scale, "q": 0.8},
    ]

    for ch in range(x.shape[0]):
        if abort_cb and abort_cb():
            break
        channel_enhanced = x[ch].copy()
        for i, band in enumerate(critical_bands):
            if abort_cb and abort_cb():
                break
            if progress_cb:
                progress = int((i + ch * len(critical_bands)) * 100 / (len(critical_bands) * x.shape[0]))
                progress_cb(progress, f"Psychoacoustic enhancement at {band['freq']}Hz")
            if band["freq"] >= sr // 2:
                continue
            try:
                b, a = design_peaking_eq(freq=band["freq"], gain_db=band["boost"], q=band["q"], sr=sr)
                filtered = filtfilt_np(b, a, x[ch])
                filtered = sanitize_audio(filtered, fallback=x[ch])
                blend_factor = 0.22
                channel_enhanced = channel_enhanced * (1.0 - blend_factor) + filtered * blend_factor
            except Exception:
                continue
        enhanced[ch] = sanitize_audio(channel_enhanced, fallback=x[ch])
    return enhanced


def stereo_width_enhancer(x, width_factor=1.15):
    x = ensure_ch_first(x).astype(np.float32, copy=False)
    if x.shape[0] != 2:
        return x
    left, right = x[0], x[1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    side_enhanced = side * float(np.clip(width_factor, 1.0, 1.8))
    return np.array([mid + side_enhanced, mid - side_enhanced], dtype=np.float32)


def dynamic_range_enhancer(x, ratio=1.12, attack_ms=5, release_ms=50, sr=44100):
    x = ensure_ch_first(x).astype(np.float32, copy=False)
    attack_samples = max(1, int(attack_ms * sr / 1000))
    release_samples = max(1, int(release_ms * sr / 1000))
    enhanced = np.zeros_like(x, dtype=np.float32)

    for ch in range(x.shape[0]):
        signal_ch = x[ch]
        envelope = np.abs(signal_ch)
        smoothed_env = np.zeros_like(envelope)
        if len(envelope) == 0:
            enhanced[ch] = signal_ch
            continue
        current_env = envelope[0]
        for i in range(len(envelope)):
            if envelope[i] > current_env:
                current_env += (envelope[i] - current_env) / attack_samples
            else:
                current_env -= (current_env - envelope[i]) / release_samples
            smoothed_env[i] = current_env
        threshold = 0.08
        gain = np.ones_like(smoothed_env)
        above = smoothed_env > threshold
        gain[above] = (smoothed_env[above] / threshold) ** (ratio - 1.0)
        gain = np.clip(gain, 1.0, 1.8)
        enhanced[ch] = signal_ch * gain
    return sanitize_audio(enhanced, fallback=x)


def enhanced_audio_algorithm(
    x: np.ndarray,
    sr: int,
    enhancement_strength: float = 0.7,
    harmonic_intensity: float = 0.6,
    stereo_width: float = 1.15,
    dynamic_enhancement: float = 1.12,
    progress_cb=None,
    abort_cb=None,
) -> np.ndarray:
    x = ensure_ch_first(x)
    x = sanitize_audio(x)

    if x is None or x.size == 0:
        raise ValueError("Input audio data is empty or None")
    if audio_peak(x) < 1e-10:
        raise ValueError("Input audio data appears to be silent")

    if progress_cb:
        progress_cb(0, "Starting enhancement process")

    if progress_cb:
        progress_cb(10, "Applying multi-band harmonic excitement")
    enhanced = multiband_exciter(
        x, sr,
        harmonic_intensity=harmonic_intensity,
        progress_cb=lambda p, desc: progress_cb(10 + p // 4, desc) if progress_cb else None,
        abort_cb=abort_cb,
    )
    enhanced = sanitize_audio(enhanced, fallback=x)
    if abort_cb and abort_cb():
        return x
    USE_PSYCHO = False  # 高速化用
    if USE_PSYCHO:
        if progress_cb:
            progress_cb(35, "Applying psychoacoustic enhancement")
        psycho_enhanced = psychoacoustic_enhancer(
            enhanced, sr,
            strength=enhancement_strength,
            progress_cb=lambda p, desc: progress_cb(35 + p // 4, desc) if progress_cb else None,
            abort_cb=abort_cb,
        )
    else:
        psycho_enhanced = enhanced

    psycho_enhanced = sanitize_audio(psycho_enhanced, fallback=x)

    if abort_cb and abort_cb():
        return x

    if progress_cb:
        progress_cb(60, "Enhancing dynamic range")
    dynamic_enhanced = dynamic_range_enhancer(
        psycho_enhanced,
        ratio=float(np.clip(dynamic_enhancement, 1.0, 1.5)),
        sr=sr,
    )
    dynamic_enhanced = sanitize_audio(dynamic_enhanced, fallback=x)
    if abort_cb and abort_cb():
        return x

    if progress_cb:
        progress_cb(75, "Enhancing stereo width")
    stereo_enhanced = stereo_width_enhancer(dynamic_enhanced, stereo_width) if x.shape[0] == 2 else dynamic_enhanced
    stereo_enhanced = sanitize_audio(stereo_enhanced, fallback=x)
    if abort_cb and abort_cb():
        return x

    if progress_cb:
        progress_cb(90, "Final processing and normalization")
    if audio_peak(stereo_enhanced) < 1e-10:
        final = x.copy()
    else:
        blend_factor = float(np.clip(enhancement_strength, 0.1, 0.8))
        final = x * (1.0 - blend_factor) + stereo_enhanced * blend_factor
    final = sanitize_audio(final, fallback=x)
    peak = audio_peak(final)
    if peak > 0.95:
        final *= 0.95 / peak
    if audio_peak(final) < 1e-10:
        final = x.copy()
    if progress_cb:
        progress_cb(100, "Enhancement complete")
    return sanitize_audio(final, fallback=x)

class DragDropListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DropOnly)
        self.setDefaultDropAction(QtCore.Qt.DropAction.CopyAction)
        self.placeholder_item = QtWidgets.QListWidgetItem(self.tr("Drag and drop audio files here..."))
        self.placeholder_item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        self.addItem(self.placeholder_item)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def _is_audio_path(self, file_path: str) -> bool:
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aif', '.aac', '.wma', '.mka'}
        _, ext = os.path.splitext(file_path.lower())
        return os.path.isfile(file_path) and ext in audio_extensions

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if self._is_audio_path(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        if self.count() == 1 and self.item(0) == self.placeholder_item:
            self.takeItem(0)

        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if self._is_audio_path(file_path):
                if not self.findItems(file_path, QtCore.Qt.MatchFlag.MatchExactly):
                    self.addItem(file_path)

        if self.count() == 0:
            self.addItem(self.placeholder_item)
        event.acceptProposedAction()


class DSREWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)
    sig_file_progress = QtCore.Signal(int, int, str)
    sig_step_progress = QtCore.Signal(int, str)
    sig_overall_progress = QtCore.Signal(int, int)
    sig_file_done = QtCore.Signal(str, str)
    sig_error = QtCore.Signal(str, str)
    sig_finished = QtCore.Signal()
    sig_retry_available = QtCore.Signal(str, str)
    sig_processing_stats = QtCore.Signal(dict)

    def __init__(self, files, output_dir, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self._abort = False
        self.processing_stats = {
            'total_files': len(files),
            'processed_files': 0,
            'failed_files': 0,
            'total_size_mb': 0.0,
            'processed_size_mb': 0.0,
            'start_time': None,
            'estimated_remaining': 0.0,
        }

    def abort(self):
        self._abort = True

    def get_file_size_mb(self, file_path: str) -> float:
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            return 0.0

    def estimate_processing_time(self, file_size_mb: float) -> float:
        return max(1.0, file_size_mb * 0.5)

    def process_audio_chunked(self, y: np.ndarray, sr: int, chunk_seconds: float = 10.0, overlap_seconds: float = 0.05) -> np.ndarray:
        if y.ndim == 1:
            y = y[np.newaxis, :]

        total_samples = y.shape[1]
        chunk_size = max(2048, int(sr * chunk_seconds))
        overlap = max(256, int(sr * overlap_seconds))

        if total_samples <= chunk_size:
            return enhanced_audio_algorithm(
                y, sr,
                enhancement_strength=float(self.params["decay"]),
                harmonic_intensity=float(self.params["m"]) / 16.0,
                stereo_width=1.15,
                dynamic_enhancement=1.12,
                progress_cb=None,
                abort_cb=lambda: self._abort,
            )

        out = np.zeros_like(y, dtype=np.float32)
        weight = np.zeros((1, total_samples), dtype=np.float32)
        step = max(1, chunk_size - overlap)

        for start in range(0, total_samples, step):
            if self._abort:
                break
            end = min(total_samples, start + chunk_size)
            chunk = y[:, start:end]
            if chunk.size == 0:
                continue

            processed_chunk = enhanced_audio_algorithm(
                chunk, sr,
                enhancement_strength=float(self.params["decay"]),
                harmonic_intensity=float(self.params["m"]) / 16.0,
                stereo_width=1.15,
                dynamic_enhancement=1.12,
                progress_cb=None,
                abort_cb=lambda: self._abort,
            )

            chunk_len = processed_chunk.shape[1]
            fade = np.ones(chunk_len, dtype=np.float32)
            if start > 0:
                fade_in = min(overlap, chunk_len)
                fade[:fade_in] = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
            if end < total_samples:
                fade_out = min(overlap, chunk_len)
                fade[-fade_out:] = np.minimum(fade[-fade_out:], np.linspace(1.0, 0.0, fade_out, dtype=np.float32))

            out[:, start:end] += processed_chunk * fade[np.newaxis, :]
            weight[:, start:end] += fade[np.newaxis, :]

        weight[weight == 0] = 1.0
        return (out / weight).astype(np.float32)

    def load_audio_with_recovery(self, file_path: str, target_sr: int):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        try:
            DEBUG_PROBE = False
            if DEBUG_PROBE:
                info = ffprobe_audio_info(file_path)
                self.sig_log.emit(f"Source info: codec={info.get('codec_name', '')}, sr={info.get('sample_rate', 0)}Hz, ch={info.get('channels', 0)}, dur={info.get('duration', 0):.2f}s")
        except Exception as probe_err:
            self.sig_log.emit(f"FFprobe warning: {probe_err}")

        self.sig_log.emit(f"Loading audio via FFmpeg: {os.path.basename(file_path)}")
        y, sr = load_audio_ffmpeg(file_path, target_sr=target_sr)
        if y is None or y.size == 0:
            raise ValueError("Empty audio data loaded")
        self.sig_log.emit(f"Loaded successfully: {y.shape}, {sr}Hz")
        return y, sr

    def categorize_error(self, error: Exception) -> str:
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['permission denied', 'access denied', 'disk full', 'no space']):
            return "fatal"
        if any(keyword in error_str for keyword in ['file not found', 'no such file', 'network', 'timeout', 'connection']):
            return "io"
        if any(keyword in error_str for keyword in ['memory', 'out of memory', 'allocation']):
            return "memory"
        if any(keyword in error_str for keyword in ['format', 'codec', 'sample rate', 'bitrate']):
            return "format"
        if any(keyword in error_str for keyword in ['ffmpeg', 'encoder', 'decoder', 'ffprobe']):
            return "ffmpeg"
        return "retry"

    def run(self):
        total = len(self.files)
        done = 0
        self.processing_stats['start_time'] = time.time()
        self.processing_stats['total_size_mb'] = sum(self.get_file_size_mb(f) for f in self.files)

        self.sig_overall_progress.emit(done, total)
        self.sig_processing_stats.emit(self.processing_stats.copy())

        for idx, in_path in enumerate(self.files, start=1):
            if self._abort:
                break

            fname = os.path.basename(in_path)
            file_size_mb = self.get_file_size_mb(in_path)
            self.sig_file_progress.emit(idx, total, fname)
            self.sig_step_progress.emit(0, fname)
            self.sig_log.emit(f"Processing {fname} ({file_size_mb:.1f}MB, est. {self.estimate_processing_time(file_size_mb):.1f}s)")

            retry_count = 0
            max_retries = 3

            while retry_count <= max_retries:
                try:
                    target_sr = int(self.params["target_sr"])
                    y, sr = self.load_audio_with_recovery(in_path, target_sr=target_sr)
                    if y.ndim == 1:
                        y = y[np.newaxis, :]

                    def step_cb(cur, desc):
                        self.sig_step_progress.emit(int(cur), fname)

                    self.sig_log.emit(f"Input audio shape: {y.shape}, sample rate: {sr} Hz")
                    self.sig_log.emit(f"Input audio range: {np.min(y):.4f} to {np.max(y):.4f}")
                    self.sig_log.emit(f"Input audio RMS: {audio_rms(y):.6f}")
                    self.sig_log.emit(f"Enhancement parameters: strength={self.params['decay']}, harmonics={self.params['m']}")

                    if audio_peak(y) < 1e-10:
                        raise RuntimeError("Input audio file appears to be silent or corrupted")

                    if file_size_mb > 150:
                        self.sig_log.emit(f"Using chunked processing for large file: {fname}")
                        y_out = self.process_audio_chunked(y, sr)
                    else:
                        self.sig_log.emit("Starting Enhanced Audio Processing...")
                        y_out = enhanced_audio_algorithm(
                            y, sr,
                            enhancement_strength=float(self.params["decay"]),
                            harmonic_intensity=float(self.params["m"]) / 16.0,
                            stereo_width=1.15,
                            dynamic_enhancement=1.12,
                            progress_cb=step_cb,
                            abort_cb=lambda: self._abort,
                        )

                    self.sig_log.emit(f"Output audio shape: {y_out.shape}, sample rate: {sr} Hz")
                    self.sig_log.emit(f"Output audio range: {np.min(y_out):.4f} to {np.max(y_out):.4f}")
                    self.sig_log.emit(f"Output audio RMS: {audio_rms(y_out):.6f}")

                    if audio_peak(y_out) < 1e-10:
                        self.sig_log.emit("ERROR: Output audio is silent! Using original audio instead.")
                        y_out = y.copy()

                    if y.shape == y_out.shape:
                        diff = np.abs(y_out - y)
                        max_diff = float(np.max(diff))
                        mean_diff = float(np.mean(diff))
                        rms_original = audio_rms(y)
                        rms_enhanced = audio_rms(y_out)
                        enhancement_ratio = rms_enhanced / (rms_original + 1e-12)
                        self.sig_log.emit("Enhancement Results:")
                        self.sig_log.emit(f"  Max difference: {max_diff:.6f}")
                        self.sig_log.emit(f"  Mean difference: {mean_diff:.6f}")
                        self.sig_log.emit(f"  RMS enhancement ratio: {enhancement_ratio:.3f}")
                        if max_diff < 0.001:
                            self.sig_log.emit("WARNING: Very small enhancement detected!")
                        else:
                            self.sig_log.emit("SUCCESS: Significant audio enhancement applied!")

                    os.makedirs(self.output_dir, exist_ok=True)
                    base, _ = os.path.splitext(fname)
                    out_ext = {'ALAC': 'm4a', 'FLAC': 'flac', 'MP3': 'mp3'}.get(self.params['format'], 'm4a')
                    out_path = os.path.join(self.output_dir, f"{base}_enhanced.{out_ext}")
                    out_path = save_with_metadata(in_path, y_out, sr, out_path, fmt=self.params['format'])

                    self.sig_log.emit(f"File saved: {out_path}")
                    self.sig_file_done.emit(in_path, out_path)
                    self.processing_stats['processed_files'] += 1
                    self.processing_stats['processed_size_mb'] += file_size_mb
                    break

                except Exception as e:
                    err = "".join(traceback.format_exception_only(type(e), e)).strip()
                    retry_count += 1
                    error_type = self.categorize_error(e)
                    if retry_count <= max_retries and error_type != "fatal":
                        self.sig_log.emit(f"[Retry {retry_count}/{max_retries}] {fname}: {err}")
                        time.sleep(2 if error_type in ("io", "ffmpeg") else 1)
                    else:
                        self.sig_error.emit(fname, err)
                        self.sig_log.emit(f"[Error] {fname}: {err}")
                        self.processing_stats['failed_files'] += 1
                        if error_type != "fatal":
                            self.sig_retry_available.emit(fname, err)
                        break

            done += 1
            self.sig_overall_progress.emit(done, total)
            self.sig_step_progress.emit(100, fname)
            self.sig_processing_stats.emit(self.processing_stats.copy())

        self.sig_finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.tr("DSRE v3.0"))
        icon_path = os.path.join(os.path.dirname(__file__), "logo.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1200, 800)

        self.dark_mode = False
        self.recent_files = []
        self.max_recent_files = 10
        self.failed_files = []
        self.worker: Optional[DSREWorker] = None
        self.config_file = os.path.join(os.path.dirname(__file__), "dsre_enhanced_config.json")

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.create_menu_bar()

        self.status_bar = self.statusBar()
        self.status_bar.showMessage(self.tr("Ready - SciPy-Free Audio Processing Algorithm Loaded"))

        self.list_files = DragDropListWidget()
        self.list_files.setToolTip(self.tr("Drag and drop audio files here for enhancement processing"))
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.btn_add = QtWidgets.QPushButton(self.tr("Add Input Files"))
        self.btn_clear = QtWidgets.QPushButton(self.tr("Clear Input List"))
        self.btn_remove_selected = QtWidgets.QPushButton(self.tr("Remove Selected"))
        self.btn_outdir = QtWidgets.QPushButton(self.tr("Select Output Directory"))
        self.le_outdir = QtWidgets.QLineEdit()
        self.le_outdir.setPlaceholderText(self.tr("Output folder"))
        self.le_outdir.setText(os.path.abspath("enhanced_output"))

        self.sb_m = QtWidgets.QSpinBox()
        self.sb_m.setRange(1, 32)
        self.sb_m.setValue(16)
        self.sb_m.setToolTip(self.tr("Harmonic intensity (1-32): Higher values add more harmonic richness"))

        self.dsb_decay = QtWidgets.QDoubleSpinBox()
        self.dsb_decay.setRange(0.1, 1.0)
        self.dsb_decay.setSingleStep(0.05)
        self.dsb_decay.setValue(0.7)
        self.dsb_decay.setToolTip(self.tr("Enhancement strength (0.1-1.0): Controls overall enhancement intensity"))

        self.sb_sr = QtWidgets.QSpinBox()
        self.sb_sr.setRange(44100, 192000)
        self.sb_sr.setSingleStep(22050)
        self.sb_sr.setValue(96000)
        self.sb_sr.setToolTip(self.tr("Target sample rate: Uses FFmpeg resampling during loading"))

        self.pb_file = QtWidgets.QProgressBar()
        self.pb_all = QtWidgets.QProgressBar()
        self.lbl_now = QtWidgets.QLabel(self.tr("Control"))
        self.lbl_stats = QtWidgets.QLabel(self.tr("Ready to process - SciPy-Free Algorithm Active"))
        self.lbl_stats.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        self.lbl_eta = QtWidgets.QLabel("")
        self.lbl_eta.setStyleSheet("QLabel { color: #666; font-size: 10px; }")

        self.btn_start = QtWidgets.QPushButton(self.tr("Start Enhanced Processing"))
        self.btn_cancel = QtWidgets.QPushButton(self.tr("Cancel Processing"))
        self.btn_cancel.setEnabled(False)
        self.btn_retry = QtWidgets.QPushButton(self.tr("Retry Failed Files"))
        self.btn_retry.setEnabled(False)
        self.btn_test_dark = QtWidgets.QPushButton(self.tr("Dark Mode"))
        self.btn_test_dark.clicked.connect(self.toggle_dark_mode)
        self.btn_retry.setStyleSheet("QPushButton { background-color: #ff9800; color: white; }")

        self.te_log = QtWidgets.QTextEdit()
        self.te_log.setReadOnly(True)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        lbl_files = QtWidgets.QLabel(self.tr("Input Audio Files"))
        lbl_files.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        left_layout.addWidget(lbl_files)
        left_layout.addWidget(self.list_files)
        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)

        middle_widget = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout()
        lbl_ops = QtWidgets.QLabel(self.tr("Enhancement Operations"))
        lbl_ops.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        middle_layout.addWidget(lbl_ops)

        vbtn = QtWidgets.QVBoxLayout()
        vbtn.addWidget(self.btn_add)
        vbtn.addWidget(self.btn_clear)
        vbtn.addWidget(self.btn_remove_selected)
        vbtn.addSpacing(10)
        vbtn.addWidget(QtWidgets.QLabel(self.tr("Enhanced Output Directory")))
        vbtn.addWidget(self.le_outdir)
        vbtn.addWidget(self.btn_outdir)
        vbtn.addSpacing(20)
        vbtn.addWidget(self.lbl_now)
        vbtn.addWidget(self.btn_start)
        vbtn.addWidget(self.btn_cancel)
        vbtn.addWidget(self.btn_retry)
        vbtn.addWidget(self.btn_test_dark)
        vbtn.addStretch(1)

        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.addItems(["ALAC", "FLAC", "MP3"])
        self.cb_format.setToolTip(self.tr("Output format: ALAC (lossless), FLAC (lossless), MP3 (lossy)"))
        vbtn.addWidget(QtWidgets.QLabel(self.tr("Output Format")))
        vbtn.addWidget(self.cb_format)

        middle_layout.addLayout(vbtn)
        middle_widget.setLayout(middle_layout)
        main_splitter.addWidget(middle_widget)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        lbl_params = QtWidgets.QLabel(self.tr("Enhancement Parameters"))
        lbl_params.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        right_layout.addWidget(lbl_params)

        form = QtWidgets.QFormLayout()
        form.addRow(self.tr("Harmonic Intensity (1-32):"), self.sb_m)
        form.addRow(self.tr("Enhancement Strength (0.1-1.0):"), self.dsb_decay)
        form.addRow(self.tr("Target Sample Rate (Hz):"), self.sb_sr)
        right_layout.addLayout(form)
        right_layout.addSpacing(20)

        vprog = QtWidgets.QVBoxLayout()
        vprog.addWidget(QtWidgets.QLabel(self.tr("Current File Enhancement Progress")))
        vprog.addWidget(self.pb_file)
        vprog.addWidget(QtWidgets.QLabel(self.tr("Overall Processing Progress")))
        vprog.addWidget(self.pb_all)
        vprog.addWidget(self.lbl_stats)
        vprog.addWidget(self.lbl_eta)
        vprog.addStretch(1)
        right_layout.addLayout(vprog)
        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([300, 300, 400])

        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vertical_splitter.addWidget(main_splitter)

        log_widget = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.addWidget(QtWidgets.QLabel(self.tr("Enhanced Processing Log")))
        log_layout.addWidget(self.te_log)
        log_widget.setLayout(log_layout)
        vertical_splitter.addWidget(log_widget)
        vertical_splitter.setSizes([600, 200])

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(vertical_splitter)
        self.central_widget.setLayout(main_layout)

        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_clear.clicked.connect(self.on_clear_files)
        self.btn_remove_selected.clicked.connect(self.on_remove_selected)
        self.btn_outdir.clicked.connect(self.on_choose_outdir)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_retry.clicked.connect(self.on_retry_failed)
        self.list_files.itemSelectionChanged.connect(self.update_button_states)

        self.load_config()
        self.sb_m.valueChanged.connect(self.save_config)
        self.dsb_decay.valueChanged.connect(self.save_config)
        self.sb_sr.valueChanged.connect(self.save_config)
        self.le_outdir.textChanged.connect(self.save_config)
        self.cb_format.currentTextChanged.connect(self.save_config)

        self.append_log("DSRE v3.0 - SciPy-Free Audio Processing Suite")
        self.append_log("=" * 60)
        self.append_log(self.tr("NEW FEATURES:"))
        self.append_log(self.tr("• SciPy removed completely"))
        self.append_log(self.tr("• librosa removed completely"))
        self.append_log(self.tr("• resampy removed completely"))
        self.append_log(self.tr("• FFmpeg-based loading and resampling"))
        self.append_log(self.tr("• NumPy-only filtering and EQ"))
        self.append_log(self.tr("• Crossfaded chunked processing for large files"))
        self.append_log("=" * 60)
        self.append_log(self.tr("Software by: Qu Le Fan (Enhanced by AI)"))
        self.append_log(self.tr("Feedback: Le_Fan_Qv@outlook.com"))
        self.append_log(self.tr("Discussion Group: 323861356 (QQ)"))
        self.append_log(self.tr("Ready for enhanced audio processing!"))

        self.apply_theme()
        self.btn_test_dark.setText(self.tr("Light Mode") if self.dark_mode else self.tr("Dark Mode"))
        self.update_button_states()

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu(self.tr('&File'))

        add_files_action = QAction(self.tr('&Add Files...'), self)
        add_files_action.setShortcut(QKeySequence.StandardKey.Open)
        add_files_action.triggered.connect(self.on_add_files)
        file_menu.addAction(add_files_action)

        clear_files_action = QAction(self.tr('&Clear All'), self)
        clear_files_action.setShortcut('Ctrl+L')
        clear_files_action.triggered.connect(self.on_clear_files)
        file_menu.addAction(clear_files_action)
        file_menu.addSeparator()

        self.recent_menu = file_menu.addMenu(self.tr('&Recent Files'))
        self.update_recent_files_menu()
        file_menu.addSeparator()

        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        process_menu = menubar.addMenu(self.tr('&Processing'))
        start_action = QAction(self.tr('&Start Enhanced Processing'), self)
        start_action.setShortcut('F5')
        start_action.triggered.connect(self.on_start)
        process_menu.addAction(start_action)

        cancel_action = QAction(self.tr('&Cancel Processing'), self)
        cancel_action.setShortcut('Escape')
        cancel_action.triggered.connect(self.on_cancel)
        process_menu.addAction(cancel_action)

        retry_action = QAction(self.tr('&Retry Failed Files'), self)
        retry_action.setShortcut('Ctrl+R')
        retry_action.triggered.connect(self.on_retry_failed)
        process_menu.addAction(retry_action)

        help_menu = menubar.addMenu(self.tr('&Help'))
        about_action = QAction(self.tr('&About'), self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.save_config()
        self.btn_test_dark.setText(self.tr("Light Mode") if self.dark_mode else self.tr("Dark Mode"))

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow { background-color: #2b2b2b; color: #ffffff; }
                QWidget { background-color: #2b2b2b; color: #ffffff; }
                QListWidget { background-color: #3c3c3c; color: #ffffff; border: 2px dashed #666666; border-radius: 5px; }
                QListWidget::item { background-color: transparent; padding: 5px; border-bottom: 1px solid #555555; }
                QListWidget::item:hover { background-color: #4a4a4a; }
                QListWidget::item:selected { background-color: #0078d4; color: white; }
                QPushButton { background-color: #404040; color: #ffffff; border: 1px solid #666666; padding: 5px; border-radius: 3px; }
                QPushButton:hover { background-color: #505050; }
                QPushButton:pressed { background-color: #606060; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #666666; padding: 5px; }
                QProgressBar { background-color: #3c3c3c; border: 1px solid #666666; text-align: center; }
                QProgressBar::chunk { background-color: #0078d4; }
                QTextEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #666666; }
                QLabel { color: #ffffff; }
                QMenuBar { background-color: #2b2b2b; color: #ffffff; border-bottom: 1px solid #666666; }
                QMenuBar::item { background-color: transparent; padding: 4px 8px; }
                QMenuBar::item:selected { background-color: #404040; }
                QMenu { background-color: #3c3c3c; color: #ffffff; border: 1px solid #666666; }
                QMenu::item { padding: 4px 20px; }
                QMenu::item:selected { background-color: #404040; }
                QStatusBar { background-color: #2b2b2b; color: #ffffff; border-top: 1px solid #666666; }
                QSplitter::handle { background-color: #666666; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #ffffff; color: #333333; }
                QWidget { background-color: #ffffff; color: #333333; }
                QListWidget { border: 2px dashed #aaa; border-radius: 5px; background-color: #f9f9f9; min-height: 200px; }
                QListWidget::item { padding: 5px; border-bottom: 1px solid #eee; color: #333; background-color: transparent; }
                QListWidget::item:hover { background-color: #e3f2fd; color: #333; }
                QListWidget::item:selected { background-color: #2196f3; color: white; border: 1px solid #1976d2; }
                QListWidget::item:selected:hover { background-color: #1976d2; color: white; }
                QPushButton { background-color: #f0f0f0; color: #333333; border: 1px solid #cccccc; padding: 5px; border-radius: 3px; }
                QPushButton:hover { background-color: #e0e0e0; }
                QPushButton:pressed { background-color: #d0d0d0; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #ffffff; color: #333333; border: 1px solid #cccccc; padding: 5px; }
                QProgressBar { background-color: #f0f0f0; border: 1px solid #cccccc; text-align: center; }
                QProgressBar::chunk { background-color: #2196f3; }
                QTextEdit { background-color: #ffffff; color: #333333; border: 1px solid #cccccc; }
                QLabel { color: #333333; }
                QMenuBar { background-color: #f0f0f0; color: #333333; border-bottom: 1px solid #cccccc; }
                QMenuBar::item { background-color: transparent; padding: 4px 8px; }
                QMenuBar::item:selected { background-color: #e0e0e0; }
                QMenu { background-color: #ffffff; color: #333333; border: 1px solid #cccccc; }
                QMenu::item { padding: 4px 20px; }
                QMenu::item:selected { background-color: #e0e0e0; }
                QStatusBar { background-color: #f0f0f0; color: #333333; border-top: 1px solid #cccccc; }
                QSplitter::handle { background-color: #cccccc; }
            """)

    def show_about(self):
        text = self.tr(
            "DSRE v3.0 - SciPy-Free Audio Processing Suite\n\n"
            "Enhanced with multi-band harmonic excitement,\n"
            "psychoacoustic enhancement, dynamic range processing,\n"
            "and FFmpeg-based decoding/resampling.\n\n"
            "Features:\n"
            "• SciPy removed completely\n"
            "• librosa removed completely\n"
            "• resampy removed completely\n"
            "• FFmpeg-based loading / resampling\n"
            "• NumPy-only filtering / EQ\n"
            "• Crossfaded chunked processing for large files\n\n"
            "Original Software by: Qu Le Fan\n"
            "Enhanced Algorithm by: AI Assistant\n"
            "Feedback: Le_Fan_Qv@outlook.com\n"
            "Discussion Group: 323861356 (QQ)"
        )
        QtWidgets.QMessageBox.about(self, self.tr("About DSRE Enhanced"), text)

    def _real_file_count(self):
        count = 0
        for i in range(self.list_files.count()):
            if self.list_files.item(i).flags() != QtCore.Qt.ItemFlag.NoItemFlags:
                count += 1
        return count

    def on_add_files(self):
        filters = (
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aiff *.aif *.aac *.wma *.mka);;"
            "All Files (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, self.tr("Select Audio Files for Enhancement"), "", filters)
        if self.list_files.count() == 1 and self.list_files.item(0).flags() == QtCore.Qt.ItemFlag.NoItemFlags:
            self.list_files.takeItem(0)
        for f in files:
            if f and not self.list_files.findItems(f, QtCore.Qt.MatchFlag.MatchExactly):
                self.list_files.addItem(f)
                self.add_to_recent_files(f)
        if self.list_files.count() == 0:
            self.list_files.addItem(self.list_files.placeholder_item)
        self.update_button_states()

    def on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select Enhanced Output Directory"), self.le_outdir.text() or "")
        if d:
            self.le_outdir.setText(d)

    def on_clear_files(self):
        self.list_files.clear()
        self.list_files.addItem(self.list_files.placeholder_item)
        self.lbl_stats.setText(self.tr("Ready for enhanced processing"))
        self.lbl_eta.setText("")
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.te_log.clear()
        self.append_log(self.tr("DSRE v3.0 - SciPy-Free Audio Processing Suite"))
        self.append_log(self.tr("Ready for enhanced audio processing!"))
        self.update_button_states()

    def update_button_states(self):
        has_selection = len(self.list_files.selectedItems()) > 0
        has_items = self._real_file_count() > 0
        self.btn_remove_selected.setEnabled(has_selection)
        self.btn_start.setEnabled(has_items and self.worker is None)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)

    def on_remove_selected(self):
        selected_items = self.list_files.selectedItems()
        if not selected_items:
            return
        for item in reversed(selected_items):
            self.list_files.takeItem(self.list_files.row(item))
        if self.list_files.count() == 0:
            self.list_files.addItem(self.list_files.placeholder_item)
        self.update_button_states()

    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.sb_m.setValue(config.get('m', 16))
                self.dsb_decay.setValue(config.get('decay', 0.7))
                self.sb_sr.setValue(config.get('target_sr', 96000))
                self.le_outdir.setText(config.get('output_dir', os.path.abspath("enhanced_output")))
                format_map = {'ALAC': 0, 'FLAC': 1, 'MP3': 2}
                self.cb_format.setCurrentIndex(format_map.get(config.get('format', 'ALAC'), 0))
                self.recent_files = config.get('recent_files', [])
                self.update_recent_files_menu()
                self.dark_mode = config.get('dark_mode', False)
        except Exception as e:
            self.append_log(f"Failed to load config: {e}")

    def save_config(self):
        try:
            config = {
                'm': self.sb_m.value(),
                'decay': self.dsb_decay.value(),
                'target_sr': self.sb_sr.value(),
                'output_dir': self.le_outdir.text(),
                'format': self.cb_format.currentText(),
                'recent_files': self.recent_files,
                'dark_mode': self.dark_mode,
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.append_log(f"Failed to save config: {e}")

    def add_to_recent_files(self, file_path: str):
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.update_recent_files_menu()

    def update_recent_files_menu(self):
        if not hasattr(self, 'recent_menu'):
            return
        self.recent_menu.clear()
        if not self.recent_files:
            action = QAction(self.tr("No recent files"), self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return
        for file_path in self.recent_files:
            action = QAction(os.path.basename(file_path), self)
            action.setToolTip(file_path)
            action.triggered.connect(lambda checked=False, path=file_path: self.load_recent_file(path))
            self.recent_menu.addAction(action)

    def load_recent_file(self, file_path: str):
        if os.path.exists(file_path):
            if self.list_files.count() == 1 and self.list_files.item(0).flags() == QtCore.Qt.ItemFlag.NoItemFlags:
                self.list_files.takeItem(0)
            if not self.list_files.findItems(file_path, QtCore.Qt.MatchFlag.MatchExactly):
                self.list_files.addItem(file_path)
        else:
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
                self.update_recent_files_menu()
            QtWidgets.QMessageBox.warning(self, self.tr("File Not Found"), self.tr(f"The file {file_path} no longer exists."))
        self.update_button_states()

    def params(self):
        return dict(
            m=self.sb_m.value(),
            decay=self.dsb_decay.value(),
            target_sr=self.sb_sr.value(),
            bit_depth=24,
            format=self.cb_format.currentText(),
        )

    def append_log(self, s: str):
        self.te_log.append(s)
        self.te_log.moveCursor(QTextCursor.End)

    def on_start(self):
        files = []
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            if item.flags() != QtCore.Qt.ItemFlag.NoItemFlags:
                files.append(item.text())
        if not files:
            QtWidgets.QMessageBox.warning(self, self.tr("No Files"), self.tr("Please add at least one audio file for enhancement"))
            return

        outdir = self.le_outdir.text().strip() or os.path.abspath("enhanced_output")
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText(self.tr("Initializing enhanced processing..."))
        self.append_log(self.tr(f"Starting enhanced processing of {len(files)} files..."))
        self.append_log(self.tr(f"Enhancement strength: {self.dsb_decay.value()}"))
        self.append_log(self.tr(f"Harmonic intensity: {self.sb_m.value()}"))
        self.failed_files.clear()
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker = DSREWorker(files, outdir, self.params())
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_file_progress.connect(self.on_file_progress)
        self.worker.sig_step_progress.connect(self.on_step_progress)
        self.worker.sig_overall_progress.connect(self.on_overall_progress)
        self.worker.sig_file_done.connect(self.on_file_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.sig_processing_stats.connect(self.on_processing_stats)
        self.worker.sig_retry_available.connect(self.on_retry_available)
        self.worker.start()
        self.update_button_states()

    @QtCore.Slot(int, int, str)
    def on_file_progress(self, cur, total, fname):
        self.lbl_now.setText(f"Enhanced Processing... [{cur}/{total}]: {fname}")
        self.pb_file.setValue(0)

    @QtCore.Slot(int, str)
    def on_step_progress(self, pct, fname):
        self.pb_file.setValue(pct)

    @QtCore.Slot(int, int)
    def on_overall_progress(self, done, total):
        self.pb_all.setValue(int(done * 100 / max(1, total)))

    @QtCore.Slot(str, str)
    def on_file_done(self, in_path, out_path):
        self.append_log(self.tr(f"Enhancement completed: {os.path.basename(in_path)} -> {out_path}"))

    @QtCore.Slot(str, str)
    def on_error(self, fname, err):
        self.append_log(self.tr(f"[Error] {fname}: {err}"))

    @QtCore.Slot(str, str)
    def on_retry_available(self, fname, err):
        if fname not in self.failed_files:
            self.failed_files.append(fname)
        self.btn_retry.setEnabled(True)
        self.append_log(self.tr(f"[Retry Available] {fname}: {err}"))

    def on_retry_failed(self):
        if not self.failed_files:
            return
        self.append_log(self.tr(f"Retrying enhanced processing for {len(self.failed_files)} failed files..."))
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText(self.tr("Retrying enhanced processing..."))
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_retry.setEnabled(False)

        self.worker = DSREWorker(self.failed_files, self.le_outdir.text().strip() or os.path.abspath("enhanced_output"), self.params())
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_file_progress.connect(self.on_file_progress)
        self.worker.sig_step_progress.connect(self.on_step_progress)
        self.worker.sig_overall_progress.connect(self.on_overall_progress)
        self.worker.sig_file_done.connect(self.on_file_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_finished.connect(self.on_retry_finished)
        self.worker.sig_processing_stats.connect(self.on_processing_stats)
        self.worker.sig_retry_available.connect(self.on_retry_available)
        self.worker.start()
        self.update_button_states()

    def on_retry_finished(self):
        self.append_log(self.tr("Enhanced retry processing completed"))
        self.lbl_now.setText(self.tr("Control"))
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)
        self.worker = None
        self.update_button_states()

    @QtCore.Slot(dict)
    def on_processing_stats(self, stats):
        if stats['start_time']:
            elapsed = time.time() - stats['start_time']
            processed = stats['processed_files']
            total = stats['total_files']
            if processed > 0:
                avg = elapsed / processed
                remaining_files = total - processed
                eta_seconds = remaining_files * avg
                stats_text = f"Enhanced: {processed}/{total} files"
                if stats['total_size_mb'] > 0:
                    stats_text += f" | {stats['processed_size_mb']:.1f}/{stats['total_size_mb']:.1f} MB"
                self.lbl_stats.setText(stats_text)
                self.lbl_eta.setText(self.tr(f"Elapsed: {self.format_time(elapsed)} | ETA: {self.format_time(eta_seconds)}"))
            else:
                self.lbl_stats.setText(self.tr(f"Starting enhanced processing of {total} files..."))
                self.lbl_eta.setText("")
        else:
            self.lbl_stats.setText(self.tr("Ready for enhanced processing"))
            self.lbl_eta.setText("")

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

    def on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.append_log(self.tr("Cancelling enhanced processing..."))
            self.worker.abort()

    def on_finished(self):
        if self.failed_files:
            self.append_log(self.tr("Processing finished with some failed files. Use 'Retry Failed Files' if needed."))
        else:
            self.append_log(self.tr("All files have been enhanced successfully!"))
        self.append_log(self.tr("Enhanced audio files saved with '_enhanced' suffix"))
        self.lbl_now.setText(self.tr("Control"))
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)
        self.worker = None
        self.update_button_states()


def main():
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.lefanqv.dsre.enhanced.scipyfree")
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    translator = QtCore.QTranslator(app)
    try:
        qm_file = os.path.join(os.path.dirname(__file__), 'lang', 'ja_JP.qm')
        if translator.load(QtCore.QLocale.Japanese, qm_file):
            QtCore.QCoreApplication.installTranslator(translator)
    except Exception:
        pass

    icon_path = os.path.join(os.path.dirname(__file__), "logo.png")
    app.setWindowIcon(QIcon(icon_path))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
