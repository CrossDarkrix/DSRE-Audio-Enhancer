#!/usr/bin/env python3
"""E2E 用の実ファイル (タグ+カバー付き FLAC, 44.1kHz/15秒) を作る."""
import os
import subprocess
import numpy as np
import soundfile as sf

base = os.path.expanduser("~/workspace/DSRE-cpp/bench/e2e")
os.makedirs(base, exist_ok=True)
sr = 44100
dur = 15.0
n = int(sr * dur)
t = np.arange(n) / sr
rng = np.random.default_rng(7)
sig = np.zeros(n)
for f0, t0, t1 in [(196.0, 0, 5), (246.94, 5, 10), (293.66, 10, 15)]:
    m = (t >= t0) & (t < t1)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 0.5 * t)
    for h, amp in [(1, .5), (2, .28), (3, .18), (4, .1), (5, .06), (7, .03), (9, .015)]:
        sig[m] += amp * env[m] * np.sin(2 * np.pi * f0 * h * t[m] + h)
    sig[m] += 0.02 * np.sin(2 * np.pi * 14800 * t[m]) * env[m]
sig += 0.005 * rng.standard_normal(n)
x = np.stack([sig * 0.6, sig * 0.55], axis=1).astype(np.float32)
wav = f"{base}/song.wav"
sf.write(wav, x, sr, subtype="FLOAT")
# カバー画像 (400x400 の単色+グラデーション)
subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "gradients=s=400x400:c0=0x334466:c1=0xcc8844",
                "-frames:v", "1", f"{base}/cover.png"], check=True, capture_output=True)
# タグ+カバー付き FLAC
subprocess.run(["ffmpeg", "-y", "-i", wav, "-i", f"{base}/cover.png",
                "-map", "0:a", "-map", "1:v", "-disposition:v", "attached_pic",
                "-metadata", "title=Test Song", "-metadata", "artist=DSRE Test",
                "-metadata", "album=Cpp E2E", "-c:a", "flac", "-c:v", "png",
                f"{base}/song.flac"], check=True, capture_output=True)
print("created", f"{base}/song.flac")
