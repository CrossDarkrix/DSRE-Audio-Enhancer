#!/usr/bin/env python3
"""Python 版ワーカーと全く同じ呼び方で E2E 出力を作る (正解生成)."""
import importlib.util
import os
import sys
import types

import numpy as np


def _dummy_class(name):
    return type(name, (), {"__init__": lambda self, *a, **k: None})


pyside = types.ModuleType("PySide6")
qtcore = types.ModuleType("PySide6.QtCore")
qtgui = types.ModuleType("PySide6.QtGui")
qtwidgets = types.ModuleType("PySide6.QtWidgets")
qtcore.QThread = _dummy_class("QThread")
qtcore.Signal = lambda *a, **k: None
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

# DSRE.py の _run_subprocess は shell=True + リスト渡しで、Windows では結合されて動くが
# POSIX では先頭要素しか実行されず ffmpeg が素で起動してしまう。
# 引数リスト自体が仕様なので、ハーネス側では shell=False の等価実装に差し替える。
import subprocess as _sp


def _run_fixed(cmd, check=False, capture_stdout=False, capture_stderr=False, input_bytes=None):
    result = _sp.run(cmd, shell=False, check=False, input=input_bytes,
                     stdout=_sp.PIPE if capture_stdout else _sp.DEVNULL,
                     stderr=_sp.PIPE if capture_stderr else _sp.DEVNULL)
    result.stdout_text = dsre._decode_subprocess_output(result.stdout if capture_stdout else b"")
    result.stderr_text = dsre._decode_subprocess_output(result.stderr if capture_stderr else b"")
    if check and result.returncode != 0:
        err = _sp.CalledProcessError(result.returncode, cmd)
        err.stdout, err.stderr = result.stdout, result.stderr
        err.stdout_text, err.stderr_text = result.stdout_text, result.stderr_text
        raise err
    return result


dsre._run_subprocess = _run_fixed

base = os.path.expanduser("~/workspace/DSRE-cpp/bench/e2e")
in_path = f"{base}/song.flac"
fmt = sys.argv[1] if len(sys.argv) > 1 else "FLAC"
out_dir = f"{base}/py_out"
os.makedirs(out_dir, exist_ok=True)
work = dsre.get_work_dir()

mm_y, sr, mm_path, wav_path = dsre.load_audio_to_memmap(in_path, target_sr=96000, temp_dir=work)
final_mm, buf_paths = dsre.enhanced_audio_algorithm_memmap(
    mm_y, sr, temp_dir=work, enhancement_strength=0.7, harmonic_intensity=16 / 16.0,
    stereo_width=1.15, dynamic_enhancement=1.12, use_psycho=True)
ext = {"ALAC": "m4a", "FLAC": "flac", "MP3": "mp3"}[fmt]
out_path = f"{out_dir}/song_enhanced.{ext}"
out_path = dsre.save_with_metadata_from_memmap(in_path, final_mm, sr, out_path, fmt=fmt,
                                               normalize=True, temp_dir=work)
print("python out:", out_path)
