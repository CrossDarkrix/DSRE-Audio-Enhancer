#!/usr/bin/env python3
"""DSRE 現行Python版の基準計測 + 正解(ゴールデン)出力生成.

PySide6 をスタブして DSRE.py の DSP 関数だけを import する。
テスト信号: 96kHz / 2ch / 10秒 の合成信号 (GUI既定の処理サンプルレートに合わせる)
パラメータは GUI 既定: m=16 -> harmonic_intensity=1.0, decay=0.7 -> strength=0.7,
stereo_width=1.15, dynamic=1.12, use_psycho=True
"""
import importlib.util
import json
import os
import resource
import sys
import time
import types

import numpy as np

# ---- PySide6 スタブ (DSP関数の import だけが目的) ----
def _dummy_class(name):
    return type(name, (), {"__init__": lambda self, *a, **k: None})


def _signal(*a, **k):
    return None


pyside = types.ModuleType("PySide6")
qtcore = types.ModuleType("PySide6.QtCore")
qtgui = types.ModuleType("PySide6.QtGui")
qtwidgets = types.ModuleType("PySide6.QtWidgets")
qtcore.QThread = _dummy_class("QThread")
qtcore.Signal = _signal
qtcore.Slot = lambda *a, **k: (lambda f: f)
qtcore.Qt = types.SimpleNamespace()
qtcore.QTranslator = _dummy_class("QTranslator")
qtcore.QLocale = types.SimpleNamespace(Japanese=1)
qtcore.QCoreApplication = _dummy_class("QCoreApplication")
for n in ["QAction", "QDragEnterEvent", "QDropEvent", "QIcon", "QKeySequence", "QTextCursor"]:
    setattr(qtgui, n, _dummy_class(n))
qtwidgets.QListWidget = _dummy_class("QListWidget")
qtwidgets.QMainWindow = _dummy_class("QMainWindow")
sys.modules["PySide6"] = pyside
sys.modules["PySide6.QtCore"] = qtcore
sys.modules["PySide6.QtGui"] = qtgui
sys.modules["PySide6.QtWidgets"] = qtwidgets

spec = importlib.util.spec_from_file_location("dsre_py", os.path.expanduser("~/workspace/DSRE-check/DSRE.py"))
dsre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsre)

SR = 96000
import sys as _s
FRAMES = int(_s.argv[1]) if len(_s.argv) > 1 else 960000
DUR = FRAMES / SR
CH = 2
N = int(SR * DUR)

rng = np.random.default_rng(42)
t = np.arange(N, dtype=np.float64) / SR
# 音楽っぽい合成信号: 基音+倍音の和音進行 + 微ノイズ + 振幅包絡
sig = np.zeros(N, dtype=np.float64)
for f0, t0, t1 in [(220.0, 0.0, 3.5), (261.63, 3.5, 7.0), (329.63, 7.0, 10.0)]:
    m = (t >= t0) & (t < t1)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    for h, amp in [(1, 0.50), (2, 0.25), (3, 0.15), (4, 0.08), (6, 0.04), (8, 0.02)]:
        sig[m] += amp * env[m] * np.sin(2 * np.pi * f0 * h * t[m] + 0.3 * h)
    # 高域のチリチリ成分 (16kHz付近)
    sig[m] += 0.015 * np.sin(2 * np.pi * 15500 * t[m]) * env[m]
sig += 0.004 * rng.standard_normal(N)
x = np.empty((CH, N), dtype=np.float32)
x[0] = (sig * 0.55).astype(np.float32)
x[1] = (sig * 0.50 + 0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

os.makedirs(os.path.expanduser("~/workspace/DSRE-cpp/bench"), exist_ok=True)
np.save(os.path.expanduser("~/workspace/DSRE-cpp/bench/bad_input.npy"), x)
x.astype(np.float32).tofile(os.path.expanduser("~/workspace/DSRE-cpp/bench/bad_input.f32"))

results = {}


def timed(name, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    results[name] = round(dt, 3)
    print(f"{name}: {dt:.3f}s")
    return out


# 各段を個別に計測 (lowmem版は memmap 前提なので実memmapを使う)
tmp = os.path.expanduser("~/workspace/DSRE-cpp/bench/tmp")
os.makedirs(tmp, exist_ok=True)


def make_mm(arr):
    mm, path = dsre.create_audio_memmap(arr.shape, dtype=np.float32, dir=tmp)
    mm[:] = arr[:]
    mm.flush()
    return mm, path


def empty_mm(shape):
    return dsre.create_audio_memmap(shape, dtype=np.float32, dir=tmp)


mm_src, _ = make_mm(x)
mm_enh, _ = empty_mm(x.shape)
timed("multiband_exciter", lambda: dsre.multiband_exciter_lowmem(mm_src, mm_enh, SR, harmonic_intensity=1.0))
mm_psy, _ = empty_mm(x.shape)
timed("psychoacoustic", lambda: dsre.psychoacoustic_enhancer_lowmem(mm_enh, mm_psy, SR, strength=0.7))
mm_dyn, _ = empty_mm(x.shape)
timed("dynamic_range", lambda: dsre.dynamic_range_enhancer_lowmem(mm_psy, mm_dyn, ratio=1.12, sr=SR))
mm_out, _ = empty_mm(x.shape)
timed("stereo_width", lambda: dsre.stereo_width_enhancer_lowmem(mm_dyn, mm_out, width_factor=1.15))
mm_fin, _ = empty_mm(x.shape)
timed("final_blend", lambda: dsre.final_blend_lowmem(mm_src, mm_out, mm_fin, enhancement_strength=0.7))

golden = np.array(mm_fin[:], dtype=np.float32)
np.save(os.path.expanduser("~/workspace/DSRE-cpp/bench/bad_golden.npy"), golden)
golden.astype(np.float32).tofile(os.path.expanduser("~/workspace/DSRE-cpp/bench/bad_golden.f32"))

# パイプライン一括 (memmap 5面生成込み) も計測
mm_src2, _ = make_mm(x)
t0 = time.perf_counter()
final_mm, _paths = dsre.enhanced_audio_algorithm_memmap(
    mm_src2, SR, temp_dir=tmp, enhancement_strength=0.7, harmonic_intensity=1.0,
    stereo_width=1.15, dynamic_enhancement=1.12, use_psycho=True)
total = time.perf_counter() - t0
results["pipeline_total_incl_memmap"] = round(total, 3)
diff = np.max(np.abs(np.array(final_mm[:]) - golden))
results["pipeline_vs_stages_maxdiff"] = float(diff)
results["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
results["sr"] = SR
results["frames"] = N
results["channels"] = CH
print(json.dumps(results, indent=2))
with open(os.path.expanduser("~/workspace/DSRE-cpp/bench/bad_python.json"), "w") as f:
    json.dump(results, f, indent=2)
