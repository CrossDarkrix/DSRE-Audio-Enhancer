#pragma once
// 1ファイル分の処理パイプライン (ffmpeg decode -> DSP -> encode+メタデータ)。
// Python 版 DSREWorker.run の処理手順・ffmpeg 引数をそのまま再現する。CLI と GUI で共有。
#include <atomic>
#include <functional>
#include <string>

namespace pipeline {

struct FileParams {
    int m = 16;                    // 1..32 (harmonic_intensity = m/16)
    double strength = 0.7;         // 0.1..1.0 (GUI の decay)
    int target_sr = 96000;
    std::string format = "ALAC";   // ALAC | FLAC | MP3
    std::string out_dir;           // 空なら入力と同じフォルダ
};

struct FileCallbacks {
    std::function<void(int percent, const std::string& stage)> progress; // 0..100 (ファイル内)
    std::function<void(const std::string& msg)> log;
    const std::atomic<bool>* abort = nullptr;
    bool aborted() const { return abort && abort->load(std::memory_order_relaxed); }
};

// 成功で true。out_path に実際の出力パス、error に理由。
bool process_file(const std::string& in_path, const FileParams& params,
                  const FileCallbacks& cb, std::string& out_path, std::string& error);

} // namespace pipeline
