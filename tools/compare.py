#!/usr/bin/env python3
"""Python 正解出力と C++ 出力の数値比較."""
import json
import numpy as np
import os

base = os.path.expanduser("~/workspace/DSRE-cpp/bench")
golden = np.load(f"{base}/golden.npy")
cpp = np.fromfile(f"{base}/cpp_out.f32", dtype=np.float32).reshape(golden.shape)
inp = np.load(f"{base}/input.npy")

diff = (cpp.astype(np.float64) - golden.astype(np.float64))
max_abs = float(np.max(np.abs(diff)))
rms_diff = float(np.sqrt(np.mean(diff ** 2)))
rms_gold = float(np.sqrt(np.mean(golden.astype(np.float64) ** 2)))
snr_db = 20 * np.log10(rms_gold / max(rms_diff, 1e-20))
max_dbfs = 20 * np.log10(max(max_abs, 1e-20))

# チャンネル別・帯域別の差 (どの段で差が出ているかの当たりを付ける)
per_ch = []
for ch in range(golden.shape[0]):
    d = diff[ch]
    per_ch.append({"ch": ch, "max_abs": float(np.max(np.abs(d))),
                   "rms": float(np.sqrt(np.mean(d ** 2)))})

# スペクトル概形の比較 (1/3オクターブ相当の粗いバンド別エネルギー, ch0)
def band_energy(x, sr=96000):
    X = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    edges = [20, 80, 250, 800, 2500, 8000, 16000, 24000, 40000, 48000]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        out.append(float(np.sum(X[m])))
    return out

eg = band_energy(golden[0])
ec = band_energy(cpp[0])
band_db = [round(10 * np.log10(max(c, 1e-20) / max(g, 1e-20)), 3) for g, c in zip(eg, ec)]

res = {
    "max_abs_diff": max_abs,
    "max_diff_dbfs": round(float(max_dbfs), 2),
    "rms_diff": rms_diff,
    "signal_to_diff_db": round(float(snr_db), 2),
    "per_channel": per_ch,
    "band_energy_diff_db_ch0": band_db,
    "input_vs_golden_rms": [round(float(np.sqrt(np.mean(inp.astype(np.float64) ** 2))), 5),
                            round(rms_gold, 5)],
}
print(json.dumps(res, indent=2))
with open(f"{base}/compare.json", "w") as f:
    json.dump(res, f, indent=2)
