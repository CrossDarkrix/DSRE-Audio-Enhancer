import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    RichLog,
    Static,
)

APP_NAME = "DSRE Textual v3.0"
CONFIG_FILE = "dsre_textual_config.json"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
    ".aac",
    ".wma",
    ".mka",
}


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


def _run_subprocess(
    cmd,
    check: bool = False,
    capture_stdout: bool = False,
    capture_stderr: bool = False,
):
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )

    result.stdout_text = _decode_subprocess_output(
        result.stdout if capture_stdout else b""
    )
    result.stderr_text = _decode_subprocess_output(
        result.stderr if capture_stderr else b""
    )

    if check and result.returncode != 0:
        err = subprocess.CalledProcessError(result.returncode, cmd)
        err.stdout = result.stdout
        err.stderr = result.stderr
        err.stdout_text = result.stdout_text
        err.stderr_text = result.stderr_text
        raise err

    return result


def add_ffmpeg_to_path():
    ffmpeg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg")
    if os.path.isdir(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir


add_ffmpeg_to_path()


def get_ffmpeg_executable() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    local = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ffmpeg",
        "ffmpeg.exe",
    )
    if os.path.exists(local):
        return local

    local_unix = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ffmpeg",
        "ffmpeg",
    )
    if os.path.exists(local_unix):
        return local_unix

    raise FileNotFoundError(
        "FFmpeg not found. Install FFmpeg or place it in the 'ffmpeg' folder next to this script."
    )


def get_ffprobe_executable() -> str:
    exe = shutil.which("ffprobe")
    if exe:
        return exe

    local = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ffmpeg",
        "ffprobe.exe",
    )
    if os.path.exists(local):
        return local

    local_unix = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ffmpeg",
        "ffprobe",
    )
    if os.path.exists(local_unix):
        return local_unix

    raise FileNotFoundError(
        "FFprobe not found. Install FFmpeg or place it in the 'ffmpeg' folder next to this script."
    )


def is_audio_file(path: str) -> bool:
    return (
        os.path.isfile(path)
        and os.path.splitext(path.lower())[1] in AUDIO_EXTENSIONS
    )


def collect_audio_files_from_directory(
    directory: str,
    recursive: bool = True,
) -> List[str]:
    found: List[str] = []

    if not os.path.isdir(directory):
        return found

    if recursive:
        for root, _, files in os.walk(directory):
            for name in files:
                path = os.path.join(root, name)
                if is_audio_file(path):
                    found.append(os.path.abspath(path))
    else:
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if is_audio_file(path):
                found.append(os.path.abspath(path))

    return sorted(found)


def ensure_ch_first(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)

    if y.ndim == 1:
        return y[np.newaxis, :]

    if y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            return y.T
        return y

    raise ValueError(f"Unsupported audio shape: {y.shape}")


def ensure_sf_shape(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)

    if y.ndim == 1:
        return y[:, None]

    if y.ndim == 2:
        if y.shape[0] <= y.shape[1]:
            return y.T
        return y

    raise ValueError(f"Unsupported audio shape: {y.shape}")


def sanitize_audio(
    x: Optional[np.ndarray],
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    if x is None:
        if fallback is not None:
            return fallback.copy().astype(np.float32)
        raise ValueError("Audio data is None")

    x = np.asarray(x)

    if x.size == 0:
        if fallback is not None:
            return fallback.copy().astype(np.float32)
        raise ValueError("Audio data is empty")

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

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
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        file_path,
    ]

    result = _run_subprocess(cmd, capture_stdout=True)
    data = json.loads(result.stdout_text)

    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if not audio_stream:
        raise ValueError(f"No audio stream found: {file_path}")

    return {
        "sample_rate": int(audio_stream.get("sample_rate", 0) or 0),
        "channels": int(audio_stream.get("channels", 0) or 0),
        "duration": float(
            audio_stream.get(
                "duration",
                data.get("format", {}).get("duration", 0),
            )
            or 0.0
        ),
        "codec_name": audio_stream.get("codec_name", ""),
        "bit_rate": int(
            audio_stream.get(
                "bit_rate",
                data.get("format", {}).get("bit_rate", 0),
            )
            or 0
        ),
    }


def load_audio_ffmpeg(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    ffmpeg = get_ffmpeg_executable()

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()

    cmd = [
        ffmpeg,
        "-i",
        file_path,
        "-vn",
        "-sn",
        "-dn",
        "-map",
        "0:a:0",
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_f32le",
        "-y",
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
        ffmpeg,
        "-i",
        in_path,
        "-an",
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-y",
        cover_tmp.name,
    ]

    try:
        result = _run_subprocess(cmd)
        if (
            result.returncode == 0
            and os.path.exists(cover_tmp.name)
            and os.path.getsize(cover_tmp.name) > 0
        ):
            return cover_tmp.name
    except Exception:
        pass

    try:
        os.remove(cover_tmp.name)
    except Exception:
        pass

    return None


def save_with_metadata(
    in_path: str,
    y_out: np.ndarray,
    sr: int,
    out_path: str,
    fmt: str = "ALAC",
    normalize: bool = True,
) -> str:
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

        codec_map = {
            "ALAC": "alac",
            "FLAC": "flac",
            "MP3": "libmp3lame",
        }

        ext_map = {
            "ALAC": "m4a",
            "FLAC": "flac",
            "MP3": "mp3",
        }

        sample_fmt_map = {
            "ALAC": "s32p",
            "FLAC": "s32",
            "MP3": "s16p",
        }

        out_path = os.path.splitext(out_path)[0] + "." + ext_map[fmt]
        cover_tmp = extract_cover_image(in_path)

        if fmt == "MP3":
            if cover_tmp:
                cmd = [
                    ffmpeg,
                    "-i",
                    tmp_wav.name,
                    "-i",
                    in_path,
                    "-i",
                    cover_tmp,
                    "-map",
                    "0:a",
                    "-map",
                    "2:v",
                    "-map_metadata",
                    "1",
                    "-id3v2_version",
                    "3",
                    "-c:a",
                    codec_map[fmt],
                    "-sample_fmt",
                    sample_fmt_map[fmt],
                    "-b:a",
                    "320k",
                    "-c:v",
                    "mjpeg",
                    "-y",
                    out_path,
                ]
            else:
                cmd = [
                    ffmpeg,
                    "-i",
                    tmp_wav.name,
                    "-i",
                    in_path,
                    "-map",
                    "0:a",
                    "-map_metadata",
                    "1",
                    "-c:a",
                    codec_map[fmt],
                    "-sample_fmt",
                    sample_fmt_map[fmt],
                    "-b:a",
                    "320k",
                    "-y",
                    out_path,
                ]
        else:
            if cover_tmp:
                cmd = [
                    ffmpeg,
                    "-i",
                    tmp_wav.name,
                    "-i",
                    in_path,
                    "-i",
                    cover_tmp,
                    "-map",
                    "0:a",
                    "-map",
                    "2:v",
                    "-disposition:v",
                    "attached_pic",
                    "-map_metadata",
                    "1",
                    "-c:a",
                    codec_map[fmt],
                    "-sample_fmt",
                    sample_fmt_map[fmt],
                    "-c:v",
                    "copy",
                    "-y",
                    out_path,
                ]
            else:
                cmd = [
                    ffmpeg,
                    "-i",
                    tmp_wav.name,
                    "-i",
                    in_path,
                    "-map",
                    "0:a",
                    "-map_metadata",
                    "1",
                    "-c:a",
                    codec_map[fmt],
                    "-sample_fmt",
                    sample_fmt_map[fmt],
                    "-y",
                    out_path,
                ]

        _run_subprocess(
            cmd,
            check=True,
            capture_stdout=True,
            capture_stderr=True,
        )

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            raise RuntimeError(f"Output file was not created correctly: {out_path}")

        return out_path

    except subprocess.CalledProcessError as e:
        stderr = getattr(e, "stderr_text", "") or _decode_subprocess_output(e.stderr)
        stdout = getattr(e, "stdout_text", "") or _decode_subprocess_output(e.stdout)

        raise RuntimeError(
            f"FFmpeg command failed: {' '.join(cmd)}\n"
            f"STDERR:\n{stderr}\n"
            f"STDOUT:\n{stdout}"
        )

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

    nb = len(b)
    na = len(a)

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

    front = 2 * x[0] - x[1 : pad + 1][::-1]
    back = 2 * x[-1] - x[-pad - 1 : -1][::-1]

    xp = np.concatenate([front, x, back])

    y = apply_iir_filter(b, a, xp)
    y = apply_iir_filter(b, a, y[::-1])[::-1]

    return y[pad : pad + len(x)].astype(np.float32)


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

    return (
        np.array([b0, b1, b2], dtype=np.float64),
        np.array([a0, a1, a2], dtype=np.float64),
    )


def bandpass_fft(
    x,
    sr,
    low_hz,
    high_hz,
    transition_ratio: float = 0.15,
):
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
        t = (freqs[rising] - low1) / max(1e-12, low2 - low1)
        mask[rising] = 0.5 - 0.5 * np.cos(np.pi * t)

    passband = (freqs >= low2) & (freqs <= high1)
    mask[passband] = 1.0

    falling = (freqs > high1) & (freqs <= high2)
    if np.any(falling):
        t = (freqs[falling] - high1) / max(1e-12, high2 - high1)
        mask[falling] = 0.5 + 0.5 * np.cos(np.pi * t)

    y = np.fft.irfft(X * mask, n=n)

    return y.astype(np.float32)


def generate_harmonics(
    signal_band,
    fundamental_freq,
    sr,
    num_harmonics: int = 5,
    harmonic_strength: float = 0.3,
):
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

            harmonic_oscillator = np.sin(
                phase_increment * np.arange(len(signal_band), dtype=np.float64)
            ).astype(np.float32)

            if not np.all(np.isfinite(harmonic_oscillator)):
                continue

            harmonic_content = signal_band * harmonic_oscillator * (
                harmonic_strength / h
            )

            if not np.all(np.isfinite(harmonic_content)):
                continue

            enhanced += harmonic_content

    return sanitize_audio(enhanced, fallback=signal_band)


def multiband_exciter(
    x,
    sr,
    harmonic_intensity: float = 0.6,
    progress_cb=None,
    abort_cb=None,
):
    x = ensure_ch_first(x).astype(np.float32, copy=False)

    enhanced = np.zeros_like(x, dtype=np.float32)
    nyquist = sr // 2
    base_strength_scale = float(np.clip(harmonic_intensity, 0.1, 1.5))

    band_definitions = [
        {
            "name": "Sub Bass",
            "low": 20,
            "high": 80,
            "gain": 1.10,
            "harmonics": 3,
            "strength": 0.08,
        },
        {
            "name": "Bass",
            "low": 80,
            "high": 250,
            "gain": 1.20,
            "harmonics": 4,
            "strength": 0.14,
        },
        {
            "name": "Low Mid",
            "low": 250,
            "high": 800,
            "gain": 1.25,
            "harmonics": 5,
            "strength": 0.18,
        },
        {
            "name": "Mid",
            "low": 800,
            "high": 2500,
            "gain": 1.35,
            "harmonics": 6,
            "strength": 0.22,
        },
        {
            "name": "High Mid",
            "low": 2500,
            "high": 8000,
            "gain": 1.45,
            "harmonics": 4,
            "strength": 0.25,
        },
        {
            "name": "Presence",
            "low": 8000,
            "high": 16000,
            "gain": 1.50,
            "harmonics": 3,
            "strength": 0.18,
        },
        {
            "name": "Air",
            "low": 16000,
            "high": min(20000, nyquist - 1000),
            "gain": 1.35,
            "harmonics": 2,
            "strength": 0.12,
        },
    ]

    bands = []
    for band in band_definitions:
        if (
            band["low"] < nyquist
            and band["high"] < nyquist
            and band["high"] > band["low"]
        ):
            bands.append(band)

    if not bands:
        return x.astype(np.float32)

    total_steps = max(1, len(bands) * x.shape[0])

    for ch in range(x.shape[0]):
        if abort_cb and abort_cb():
            break

        channel_enhanced = x[ch].copy()

        for i, band in enumerate(bands):
            if abort_cb and abort_cb():
                break

            if progress_cb:
                progress = int((i + ch * len(bands)) * 100 / total_steps)
                progress_cb(progress, f"Processing band {band['name']}")

            try:
                band_signal = bandpass_fft(
                    x[ch],
                    sr,
                    band["low"],
                    band["high"],
                )

                band_signal = sanitize_audio(
                    band_signal,
                    fallback=np.zeros_like(x[ch], dtype=np.float32),
                )

                if audio_peak(band_signal) < 1e-8:
                    continue

                center_freq = (band["low"] + band["high"]) / 2
                strength = band["strength"] * base_strength_scale

                harmonics_added = generate_harmonics(
                    band_signal,
                    center_freq,
                    sr,
                    band["harmonics"],
                    strength,
                )

                saturated = np.tanh(harmonics_added * 1.35).astype(np.float32) * 0.82
                band_enhanced = saturated * band["gain"]

                if not np.all(np.isfinite(band_enhanced)):
                    continue

                channel_enhanced = channel_enhanced + band_enhanced * 0.22

            except Exception:
                continue

        enhanced[ch] = sanitize_audio(channel_enhanced, fallback=x[ch])

    return enhanced


def psychoacoustic_enhancer(
    x,
    sr,
    strength: float = 1.0,
    progress_cb=None,
    abort_cb=None,
):
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

    total_steps = max(1, len(critical_bands) * x.shape[0])

    for ch in range(x.shape[0]):
        if abort_cb and abort_cb():
            break

        channel_enhanced = x[ch].copy()

        for i, band in enumerate(critical_bands):
            if abort_cb and abort_cb():
                break

            if progress_cb:
                progress = int((i + ch * len(critical_bands)) * 100 / total_steps)
                progress_cb(progress, f"Psychoacoustic enhancement at {band['freq']}Hz")

            if band["freq"] >= sr // 2:
                continue

            try:
                b, a = design_peaking_eq(
                    freq=band["freq"],
                    gain_db=band["boost"],
                    q=band["q"],
                    sr=sr,
                )

                filtered = filtfilt_np(b, a, x[ch])
                filtered = sanitize_audio(filtered, fallback=x[ch])

                blend_factor = 0.22
                channel_enhanced = (
                    channel_enhanced * (1.0 - blend_factor)
                    + filtered * blend_factor
                )

            except Exception:
                continue

        enhanced[ch] = sanitize_audio(channel_enhanced, fallback=x[ch])

    return enhanced


def stereo_width_enhancer(x, width_factor: float = 1.15):
    x = ensure_ch_first(x).astype(np.float32, copy=False)

    if x.shape[0] != 2:
        return x

    left, right = x[0], x[1]

    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    side_enhanced = side * float(np.clip(width_factor, 1.0, 1.8))

    return np.array(
        [
            mid + side_enhanced,
            mid - side_enhanced,
        ],
        dtype=np.float32,
    )


def dynamic_range_enhancer(
    x,
    ratio: float = 1.12,
    attack_ms: float = 5,
    release_ms: float = 50,
    sr: int = 44100,
):
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
        x,
        sr,
        harmonic_intensity=harmonic_intensity,
        progress_cb=lambda p, desc: progress_cb(10 + p // 4, desc)
        if progress_cb
        else None,
        abort_cb=abort_cb,
    )

    enhanced = sanitize_audio(enhanced, fallback=x)

    if abort_cb and abort_cb():
        return x

    use_psycho = False

    if use_psycho:
        if progress_cb:
            progress_cb(35, "Applying psychoacoustic enhancement")

        psycho_enhanced = psychoacoustic_enhancer(
            enhanced,
            sr,
            strength=enhancement_strength,
            progress_cb=lambda p, desc: progress_cb(35 + p // 4, desc)
            if progress_cb
            else None,
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

    stereo_enhanced = (
        stereo_width_enhancer(dynamic_enhanced, stereo_width)
        if x.shape[0] == 2
        else dynamic_enhanced
    )

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


class DSREProcessor:
    def __init__(
        self,
        files: List[str],
        output_dir: str,
        params: Dict[str, Any],
        log_cb,
        file_progress_cb,
        step_progress_cb,
        stats_cb,
        abort_cb,
    ):
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self.logs = log_cb
        self.file_progress = file_progress_cb
        self.step_progress = step_progress_cb
        self.stats = stats_cb
        self.abort_cb = abort_cb

        self.processing_stats = {
            "total_files": len(files),
            "processed_files": 0,
            "failed_files": 0,
            "total_size_mb": 0.0,
            "processed_size_mb": 0.0,
            "start_time": None,
        }

    def get_file_size_mb(self, file_path: str) -> float:
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            return 0.0

    def process_audio_chunked(
        self,
        y: np.ndarray,
        sr: int,
        chunk_seconds: float = 10.0,
        overlap_seconds: float = 0.05,
    ) -> np.ndarray:
        if y.ndim == 1:
            y = y[np.newaxis, :]

        total_samples = y.shape[1]
        chunk_size = max(2048, int(sr * chunk_seconds))
        overlap = max(256, int(sr * overlap_seconds))

        if total_samples <= chunk_size:
            return enhanced_audio_algorithm(
                y,
                sr,
                enhancement_strength=float(self.params["decay"]),
                harmonic_intensity=float(self.params["m"]) / 16.0,
                stereo_width=float(self.params["stereo_width"]),
                dynamic_enhancement=float(self.params["dynamic"]),
                progress_cb=None,
                abort_cb=self.abort_cb,
            )

        out = np.zeros_like(y, dtype=np.float32)
        weight = np.zeros((1, total_samples), dtype=np.float32)

        step = max(1, chunk_size - overlap)

        for start in range(0, total_samples, step):
            if self.abort_cb():
                break

            end = min(total_samples, start + chunk_size)
            chunk = y[:, start:end]

            if chunk.size == 0:
                continue

            processed_chunk = enhanced_audio_algorithm(
                chunk,
                sr,
                enhancement_strength=float(self.params["decay"]),
                harmonic_intensity=float(self.params["m"]) / 16.0,
                stereo_width=float(self.params["stereo_width"]),
                dynamic_enhancement=float(self.params["dynamic"]),
                progress_cb=None,
                abort_cb=self.abort_cb,
            )

            chunk_len = processed_chunk.shape[1]
            fade = np.ones(chunk_len, dtype=np.float32)

            if start > 0:
                fade_in = min(overlap, chunk_len)
                fade[:fade_in] = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)

            if end < total_samples:
                fade_out = min(overlap, chunk_len)
                fade[-fade_out:] = np.minimum(
                    fade[-fade_out:],
                    np.linspace(1.0, 0.0, fade_out, dtype=np.float32),
                )

            out[:, start:end] += processed_chunk * fade[np.newaxis, :]
            weight[:, start:end] += fade[np.newaxis, :]

        weight[weight == 0] = 1.0

        return (out / weight).astype(np.float32)

    def categorize_error(self, error: Exception) -> str:
        error_str = str(error).lower()

        if any(
            keyword in error_str
            for keyword in (
                "permission denied",
                "access denied",
                "disk full",
                "no space",
            )
        ):
            return "fatal"

        if any(
            keyword in error_str
            for keyword in (
                "file not found",
                "no such file",
                "network",
                "timeout",
                "connection",
            )
        ):
            return "io"

        if any(
            keyword in error_str
            for keyword in (
                "memory",
                "out of memory",
                "allocation",
            )
        ):
            return "memory"

        if any(
            keyword in error_str
            for keyword in (
                "format",
                "codec",
                "sample rate",
                "bitrate",
            )
        ):
            return "format"

        if any(
            keyword in error_str
            for keyword in (
                "ffmpeg",
                "encoder",
                "decoder",
                "ffprobe",
            )
        ):
            return "ffmpeg"

        return "retry"

    def run(self):
        total = len(self.files)
        done = 0

        self.processing_stats["start_time"] = time.time()
        self.processing_stats["total_size_mb"] = sum(
            self.get_file_size_mb(path) for path in self.files
        )

        self.file_progress(done, total, "")
        self.stats(dict(self.processing_stats))

        os.makedirs(self.output_dir, exist_ok=True)

        for idx, in_path in enumerate(self.files, start=1):
            if self.abort_cb():
                self.logs("[yellow]Processing aborted[/]")
                break

            fname = os.path.basename(in_path)
            file_size_mb = self.get_file_size_mb(in_path)

            self.file_progress(idx, total, fname)
            self.step_progress(0, fname)

            self.logs(
                f"[cyan]Processing[/] {fname} "
                f"({file_size_mb:.1f} MB, {idx}/{total})"
            )

            retry_count = 0
            max_retries = 3

            while retry_count <= max_retries:
                if self.abort_cb():
                    break

                try:
                    target_sr = int(self.params["target_sr"])

                    try:
                        info = ffprobe_audio_info(in_path)
                        self.logs(
                            "[dim]"
                            f"Source info: codec={info.get('codec_name', '')}, "
                            f"sr={info.get('sample_rate', 0)}Hz, "
                            f"ch={info.get('channels', 0)}, "
                            f"dur={info.get('duration', 0):.2f}s"
                            "[/]"
                        )
                    except Exception as probe_error:
                        self.logs(f"[yellow]FFprobe warning:[/] {probe_error}")

                    self.logs(f"Loading audio via FFmpeg: {fname}")

                    y, sr = load_audio_ffmpeg(in_path, target_sr=target_sr)

                    if y.ndim == 1:
                        y = y[np.newaxis, :]

                    if y is None or y.size == 0:
                        raise ValueError("Empty audio data loaded")

                    if audio_peak(y) < 1e-10:
                        raise RuntimeError(
                            "Input audio file appears to be silent or corrupted"
                        )

                    self.logs(f"Loaded successfully: shape={y.shape}, sr={sr}Hz")
                    self.logs(
                        f"Input range: {np.min(y):.4f} to {np.max(y):.4f}, "
                        f"RMS={audio_rms(y):.6f}"
                    )
                    self.logs(
                        "Enhancement parameters: "
                        f"strength={self.params['decay']}, "
                        f"harmonics={self.params['m']}, "
                        f"stereo_width={self.params['stereo_width']}, "
                        f"dynamic={self.params['dynamic']}"
                    )

                    def step_cb(cur, desc):
                        self.step_progress(int(cur), fname)
                        if desc:
                            self.logs(f"[dim]{desc}[/]")

                    if file_size_mb > float(self.params["chunk_threshold_mb"]):
                        self.logs(
                            f"Using chunked processing for large file: {fname}"
                        )
                        y_out = self.process_audio_chunked(y, sr)
                    else:
                        self.logs("Starting enhanced audio processing")
                        y_out = enhanced_audio_algorithm(
                            y,
                            sr,
                            enhancement_strength=float(self.params["decay"]),
                            harmonic_intensity=float(self.params["m"]) / 16.0,
                            stereo_width=float(self.params["stereo_width"]),
                            dynamic_enhancement=float(self.params["dynamic"]),
                            progress_cb=step_cb,
                            abort_cb=self.abort_cb,
                        )

                    if self.abort_cb():
                        break

                    self.logs(
                        f"Output range: {np.min(y_out):.4f} to {np.max(y_out):.4f}, "
                        f"RMS={audio_rms(y_out):.6f}"
                    )

                    if audio_peak(y_out) < 1e-10:
                        self.logs(
                            "[red]Output audio is silent. Using original audio instead.[/]"
                        )
                        y_out = y.copy()

                    if y.shape == y_out.shape:
                        diff = np.abs(y_out - y)
                        max_diff = float(np.max(diff))
                        mean_diff = float(np.mean(diff))
                        rms_original = audio_rms(y)
                        rms_enhanced = audio_rms(y_out)
                        enhancement_ratio = rms_enhanced / (rms_original + 1e-12)

                        self.logs("[bold]Enhancement Results:[/]")
                        self.logs(f"  Max difference: {max_diff:.6f}")
                        self.logs(f"  Mean difference: {mean_diff:.6f}")
                        self.logs(f"  RMS enhancement ratio: {enhancement_ratio:.3f}")

                        if max_diff < 0.001:
                            self.logs("[yellow]WARNING: Very small enhancement detected[/]")
                        else:
                            self.logs("[green]SUCCESS: Enhancement applied[/]")

                    base, _ = os.path.splitext(fname)

                    ext_map = {
                        "ALAC": "m4a",
                        "FLAC": "flac",
                        "MP3": "mp3",
                    }

                    out_ext = ext_map.get(self.params["format"], "m4a")

                    out_path = os.path.join(
                        self.output_dir,
                        f"{base}_enhanced.{out_ext}",
                    )

                    out_path = save_with_metadata(
                        in_path,
                        y_out,
                        sr,
                        out_path,
                        fmt=self.params["format"],
                    )

                    self.logs(f"[green]Saved:[/] {out_path}")

                    self.processing_stats["processed_files"] += 1
                    self.processing_stats["processed_size_mb"] += file_size_mb

                    break

                except Exception as e:
                    err = "".join(
                        traceback.format_exception_only(type(e), e)
                    ).strip()

                    retry_count += 1
                    error_type = self.categorize_error(e)

                    if retry_count <= max_retries and error_type != "fatal":
                        self.logs(
                            f"[yellow][Retry {retry_count}/{max_retries}][/]"
                            f" {fname}: {err}"
                        )
                        time.sleep(2 if error_type in ("io", "ffmpeg") else 1)
                    else:
                        self.logs(f"[red][Error][/] {fname}: {err}")
                        self.logs(traceback.format_exc())
                        self.processing_stats["failed_files"] += 1
                        break

            done += 1
            self.file_progress(done, total, fname)
            self.step_progress(100, fname)
            self.stats(dict(self.processing_stats))

        self.logs("[bold green]Processing finished[/]")


class DSRETextualApp(App):
    TITLE = APP_NAME
    SUB_TITLE = "SciPy-Free Audio Enhancement TUI"

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }
    .panel {
        height: 100%;
        border: round #555;
        padding: 1;
    }
    #left_panel {
        width: 42%;
        border: round #555;
        padding: 1;
    }

    #middle_panel {
        width: 30%;
        border: round #555;
        padding: 1;
    }

    #right_panel {
        width: 28%;
        border: round #555;
        padding: 1;
    }

    #log_panel {
        height: 15;
        border: round #555;
    }

    Input {
        margin-bottom: 1;
    }

    Button {
        margin-bottom: 1;
    }

    ProgressBar {
        margin-bottom: 1;
    }

    .section-title {
        text-style: bold;
        color: cyan;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("f5", "start_processing", "Start"),
        ("escape", "cancel_processing", "Cancel"),
        ("ctrl+l", "clear_files", "Clear"),
        ("ctrl+q", "quit", "Quit"),
    ]

    processing = reactive(False)

    def __init__(self):
        super().__init__()

        self.files: List[str] = []
        self.failed_files: List[str] = []
        self.processor_task: Optional[asyncio.Task] = None
        self.cancel_requested = False

        self.config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            CONFIG_FILE,
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main"):
            with VerticalScroll(id="left_panel", classes="panel"):
                yield Label("Input Audio Files", classes="section-title")
                self.input_directory = Input(
                    placeholder="Directory path for bulk add"
                )
                yield self.input_directory
                self.file_list = ListView()
                yield self.file_list
                yield Button("Scan Directory Recursively", id="scan_directory")
                yield Button("Clear Input List", id="clear")

            with VerticalScroll(id="middle_panel", classes="panel"):
                yield Label("Enhancement Parameters", classes="section-title")

                self.input_m = Input(
                    value="12",
                    placeholder="Harmonic Intensity 1-32",
                )
                self.input_decay = Input(
                    value="0.35",
                    placeholder="Enhancement Strength 0.1-1.0",
                )
                self.input_sr = Input(
                    value="96000",
                    placeholder="Target Sample Rate",
                )
                self.input_format = Input(
                    value="ALAC",
                    placeholder="ALAC / FLAC / MP3",
                )
                self.input_stereo_width = Input(
                    value="1.15",
                    placeholder="Stereo Width 1.0-1.8",
                )
                self.input_dynamic = Input(
                    value="1.12",
                    placeholder="Dynamic Enhancement 1.0-1.5",
                )
                self.input_chunk_threshold = Input(
                    value="150",
                    placeholder="Chunk threshold MB",
                )
                self.input_output_dir = Input(
                    value=os.path.join(os.path.expanduser('~'), "enhanced_output"),
                    placeholder="Output Directory",
                )

                yield Label("Harmonic Intensity")
                yield self.input_m

                yield Label("Enhancement Strength")
                yield self.input_decay

                yield Label("Target Sample Rate")
                yield self.input_sr

                yield Label("Output Format")
                yield self.input_format

                yield Label("Stereo Width")
                yield self.input_stereo_width

                yield Label("Dynamic Enhancement")
                yield self.input_dynamic

                yield Label("Chunk Threshold MB")
                yield self.input_chunk_threshold

                yield Label("Output Directory")
                yield self.input_output_dir

                yield Button("Save Config", id="save_config")
                yield Button("Load Config", id="load_config")

            with VerticalScroll(id="right_panel", classes="panel"):
                yield Label("Processing", classes="section-title")

                yield Button("Start Processing", id="start")
                yield Button("Cancel Processing", id="cancel")
                yield Button("Retry Failed Files", id="retry_failed")

                yield Label("Current File Progress")
                self.file_progress_bar = ProgressBar(total=100)
                yield self.file_progress_bar

                yield Label("Overall Progress")
                self.overall_progress_bar = ProgressBar(total=100)
                yield self.overall_progress_bar

                self.status_label = Static("Ready")
                yield self.status_label

                self.stats_label = Static("0 files ready")
                yield self.stats_label

                self.eta_label = Static("")
                yield self.eta_label

        self.logs = RichLog(id="log_panel", markup=True, highlight=True)
        yield self.logs

        yield Footer()

    async def on_mount(self):
        self.write_log("[bold cyan]DSRE Textual v3.0[/]")
        self.write_log("SciPy-Free Audio Processing Suite")
        self.write_log("FFmpeg-based loading / resampling")
        self.write_log("NumPy-only filtering / EQ")
        self.write_log("Directory bulk add enabled")
        self.write_log("=" * 60)

        self.load_config()
        self.update_status()

    def write_log(self, message: str):
        if hasattr(self, "log"):
            self.logs.write(message)

    def thread_log(self, message: str):
        self.call_from_thread(self.write_log, message)

    def update_status(self):
        if hasattr(self, "stats_label"):
            self.stats_label.update(f"{len(self.files)} files ready")

    def thread_update_file_progress(self, done: int, total: int, fname: str):
        self.call_from_thread(self.update_file_progress, done, total, fname)

    def update_file_progress(self, done: int, total: int, fname: str):
        pct = int(done * 100 / max(1, total))
        self.overall_progress_bar.update(progress=pct)

        if fname:
            self.status_label.update(f"Processing [{done}/{total}]: {fname}")
        else:
            self.status_label.update("Processing...")

    def thread_update_step_progress(self, pct: int, fname: str):
        self.call_from_thread(self.update_step_progress, pct, fname)

    def update_step_progress(self, pct: int, fname: str):
        self.file_progress_bar.update(progress=max(0, min(100, pct)))

    def thread_update_stats(self, stats: Dict[str, Any]):
        self.call_from_thread(self.update_stats, stats)

    def update_stats(self, stats: Dict[str, Any]):
        total = stats.get("total_files", 0)
        processed = stats.get("processed_files", 0)
        failed = stats.get("failed_files", 0)
        processed_size = stats.get("processed_size_mb", 0.0)
        total_size = stats.get("total_size_mb", 0.0)

        self.stats_label.update(
            f"Processed: {processed}/{total} | Failed: {failed} | "
            f"{processed_size:.1f}/{total_size:.1f} MB"
        )

        start_time = stats.get("start_time")
        if start_time and processed > 0:
            elapsed = time.time() - start_time
            avg = elapsed / processed
            remaining = max(0, total - processed)
            eta = remaining * avg
            self.eta_label.update(
                f"Elapsed: {self.format_time(elapsed)} | ETA: {self.format_time(eta)}"
            )
        else:
            self.eta_label.update("")

    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"

        if seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)

        return f"{hours}h {minutes}m"

    def abort_requested(self) -> bool:
        return self.cancel_requested

    async def add_file_to_list(self, path: str) -> bool:
        if not path:
            return False

        path = os.path.abspath(os.path.expanduser(path))

        if not is_audio_file(path):
            return False

        if path in self.files:
            return False

        self.files.append(path)

        await self.file_list.append(
            ListItem(Label(path))
        )

        self.update_status()
        return True

    async def add_directory_to_list(
        self,
        directory: str,
        recursive: bool = True,
    ) -> int:
        if not directory:
            return 0

        directory = os.path.abspath(os.path.expanduser(directory))

        audio_files = collect_audio_files_from_directory(
            directory,
            recursive=recursive,
        )

        added = 0

        for path in audio_files:
            ok = await self.add_file_to_list(path)
            if ok:
                added += 1

        self.update_status()
        return added

    async def on_button_pressed(self, event: Button.Pressed):
        button_id = event.button.id

        if button_id == "add_file":
            await self.handle_add_file()

        elif button_id == "scan_directory":
            await self.handle_scan_directory()

        elif button_id == "clear":
            await self.action_clear_files()

        elif button_id == "start":
            await self.start_processing()

        elif button_id == "cancel":
            self.action_cancel_processing()

        elif button_id == "retry_failed":
            await self.retry_failed_files()

        elif button_id == "save_config":
            self.save_config()
            self.write_log("[green]Config saved[/]")

        elif button_id == "load_config":
            self.load_config()
            self.write_log("[green]Config loaded[/]")

    async def handle_add_file(self):
        path = self.input_file.value.strip()

        if not path:
            self.write_log("[yellow]ファイルパスを入力してください[/]")
            return

        if await self.add_file_to_list(path):
            self.write_log(f"[green]Added file:[/] {os.path.abspath(path)}")
        else:
            self.write_log(
                f"[red]Not an audio file, missing file, or already added:[/] {path}"
            )

    async def handle_scan_directory(self):
        directory = self.input_directory.value.strip()

        if not directory:
            self.write_log("[yellow]ディレクトリパスを入力してください[/]")
            return

        directory = os.path.abspath(os.path.expanduser(directory))

        if not os.path.isdir(directory):
            self.write_log(f"[red]Directory not found:[/] {directory}")
            return

        self.write_log(f"[cyan]Scanning directory recursively:[/] {directory}")

        added = await self.add_directory_to_list(directory, recursive=True)

        self.write_log(
            f"[green]Directory scan completed:[/] {added} audio files added"
        )

    def read_params(self) -> Dict[str, Any]:
        m = int(self.input_m.value.strip() or "12")
        decay = float(self.input_decay.value.strip() or "0.35")
        target_sr = int(self.input_sr.value.strip() or "96000")
        fmt = (self.input_format.value.strip() or "ALAC").upper()
        stereo_width = float(self.input_stereo_width.value.strip() or "1.15")
        dynamic = float(self.input_dynamic.value.strip() or "1.12")
        chunk_threshold_mb = float(self.input_chunk_threshold.value.strip() or "150")

        m = int(np.clip(m, 1, 32))
        decay = float(np.clip(decay, 0.1, 1.0))
        target_sr = int(np.clip(target_sr, 44100, 192000))
        stereo_width = float(np.clip(stereo_width, 1.0, 1.8))
        dynamic = float(np.clip(dynamic, 1.0, 1.5))
        chunk_threshold_mb = max(1.0, chunk_threshold_mb)

        if fmt not in ("ALAC", "FLAC", "MP3"):
            raise ValueError("Output format must be ALAC, FLAC, or MP3")

        return {
            "m": m,
            "decay": decay,
            "target_sr": target_sr,
            "format": fmt,
            "stereo_width": stereo_width,
            "dynamic": dynamic,
            "chunk_threshold_mb": chunk_threshold_mb,
        }

    async def start_processing(self):
        if self.processing:
            self.write_log("[yellow]Already processing[/]")
            return

        if not self.files:
            self.write_log("[red]No files selected[/]")
            return

        try:
            params = self.read_params()
        except Exception as e:
            self.write_log(f"[red]Invalid parameters:[/] {e}")
            return

        output_dir = self.input_output_dir.value.strip() or os.path.join(os.path.expanduser('~'), "enhanced_output")
        output_dir = os.path.abspath(os.path.expanduser(output_dir))

        os.makedirs(output_dir, exist_ok=True)

        self.processing = True
        self.cancel_requested = False
        self.failed_files.clear()

        self.file_progress_bar.update(progress=0)
        self.overall_progress_bar.update(progress=0)

        self.status_label.update("Processing...")
        self.write_log("[bold cyan]Starting enhanced processing[/]")
        self.write_log(f"Files: {len(self.files)}")
        self.write_log(f"Output directory: {output_dir}")
        self.write_log(f"Format: {params['format']}")

        files = list(self.files)

        processor = DSREProcessor(
            files=files,
            output_dir=output_dir,
            params=params,
            log_cb=self.thread_log,
            file_progress_cb=self.thread_update_file_progress,
            step_progress_cb=self.thread_update_step_progress,
            stats_cb=self.thread_update_stats,
            abort_cb=self.abort_requested,
        )

        async def runner():
            try:
                await asyncio.to_thread(processor.run)
            finally:
                self.processing = False
                self.cancel_requested = False
                self.call_from_thread(self.on_processing_finished)

        self.processor_task = asyncio.create_task(runner())

    def on_processing_finished(self):
        self.status_label.update("Finished")
        self.file_progress_bar.update(progress=100)

        if self.cancel_requested:
            self.write_log("[yellow]Processing cancelled[/]")
        else:
            self.write_log("[bold green]All processing tasks finished[/]")

        self.update_status()

    async def retry_failed_files(self):
        if not self.failed_files:
            self.write_log("[yellow]No failed files to retry[/]")
            return

        original = self.files
        self.files = list(self.failed_files)
        await self.start_processing()
        self.files = original

    def action_cancel_processing(self):
        if self.processing:
            self.cancel_requested = True
            self.status_label.update("Cancel requested")
            self.write_log("[yellow]Cancel requested[/]")
        else:
            self.write_log("[yellow]No active processing[/]")

    async def action_start_processing(self):
        await self.start_processing()

    async def action_clear_files(self):
        if self.processing:
            self.write_log("[yellow]Cannot clear while processing[/]")
            return

        self.files.clear()
        self.failed_files.clear()

        await self.file_list.clear()

        self.file_progress_bar.update(progress=0)
        self.overall_progress_bar.update(progress=0)

        self.status_label.update("Ready")
        self.stats_label.update("0 files ready")
        self.eta_label.update("")

        self.write_log("[yellow]Input list cleared[/]")

    def save_config(self):
        try:
            config = {
                "m": self.input_m.value,
                "decay": self.input_decay.value,
                "target_sr": self.input_sr.value,
                "format": self.input_format.value,
                "stereo_width": self.input_stereo_width.value,
                "dynamic": self.input_dynamic.value,
                "chunk_threshold_mb": self.input_chunk_threshold.value,
                "output_dir": self.input_output_dir.value,
                "last_directory": self.input_directory.value,
                "last_file": self.input_file.value,
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.write_log(f"[red]Failed to save config:[/] {e}")

    def load_config(self):
        try:
            if not os.path.exists(self.config_path):
                return

            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.input_m.value = str(config.get("m", "12"))
            self.input_decay.value = str(config.get("decay", "0.35"))
            self.input_sr.value = str(config.get("target_sr", "96000"))
            self.input_format.value = str(config.get("format", "ALAC"))
            self.input_stereo_width.value = str(config.get("stereo_width", "1.15"))
            self.input_dynamic.value = str(config.get("dynamic", "1.12"))
            self.input_chunk_threshold.value = str(
                config.get("chunk_threshold_mb", "150")
            )
            self.input_output_dir.value = str(
                config.get(
                    "output_dir",
                    os.path.join(os.path.expanduser('~'), "enhanced_output"),
                )
            )
            self.input_directory.value = str(config.get("last_directory", ""))
            self.input_file.value = str(config.get("last_file", ""))

        except Exception as e:
            self.write_log(f"[red]Failed to load config:[/] {e}")


def main():
    try:
        if sys.platform.startswith("win"):
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "com.lefanqv.dsre.textual"
            )
    except Exception:
        pass

    app = DSRETextualApp()
    app.run()


if __name__ == "__main__":
    main()