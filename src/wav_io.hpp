#pragma once
// 最小 WAV (IEEE float32) 読み書き。ffmpeg の pcm_f32le 出力と
// Python 版が soundfile で書く FLOAT WAV の橋渡し専用。
#include <string>
#include <vector>

namespace wavio {

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples; // interleaved (frames x channels)
};

// 成功で true。error に理由が入る。
bool read_wav_f32(const std::string& path, WavData& out, std::string& error);
// planar (channels x frames) をインターリーブに直して書く。
bool write_wav_f32(const std::string& path, const std::vector<float>& planar,
                   int channels, int sample_rate, float scale, std::string& error);

} // namespace wavio
