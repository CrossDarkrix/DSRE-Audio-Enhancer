#pragma once
// DSRE DSP コア (C++17) — 現行 Python 版 (DSRE.py lowmem) の計算式・精度をそのまま移植。
// 入出力は float32 平面 (channels x frames)。
#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

namespace dsre {

struct Params {
    double harmonic_intensity = 1.0;   // GUI m=16 -> m/16
    double enhancement_strength = 0.7; // GUI decay
    float stereo_width = 1.15f;
    double dynamic_enhancement = 1.12;
    bool use_psycho = true;
};

struct StageTimes {
    double exciter = 0, psycho = 0, dynamic = 0, stereo = 0, blend = 0;
};

// 進捗と中断。progress は全体 0..100 と段名を通知する。
// abort が true になったら process は速やかに false を返す (出力は未完成)。
struct Callbacks {
    std::function<void(int percent, const char* stage)> progress;
    const std::atomic<bool>* abort = nullptr;
    bool aborted() const { return abort && abort->load(std::memory_order_relaxed); }
};

// x: channels x frames (planar)。成功で true、out に結果。
bool process(const std::vector<float>& x, int channels, int sr,
             const Params& p, std::vector<float>& out,
             const Callbacks& cb = {}, StageTimes* times = nullptr);

} // namespace dsre
