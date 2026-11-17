#include "wav_io.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace wavio {
namespace {

uint16_t u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
void put_u16(std::ofstream& f, uint16_t v) {
    uint8_t b[2] = {(uint8_t)(v & 0xff), (uint8_t)(v >> 8)};
    f.write((const char*)b, 2);
}
void put_u32(std::ofstream& f, uint32_t v) {
    uint8_t b[4] = {(uint8_t)(v & 0xff), (uint8_t)((v >> 8) & 0xff),
                    (uint8_t)((v >> 16) & 0xff), (uint8_t)(v >> 24)};
    f.write((const char*)b, 4);
}
void put_tag(std::ofstream& f, const char* t) { f.write(t, 4); }

} // namespace

bool read_wav_f32(const std::string& path, WavData& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "cannot open: " + path; return false; }
    std::vector<uint8_t> all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (all.size() < 44 || std::memcmp(all.data(), "RIFF", 4) != 0 ||
        std::memcmp(all.data() + 8, "WAVE", 4) != 0) {
        error = "not a RIFF/WAVE file: " + path;
        return false;
    }
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t sr = 0;
    size_t data_off = 0, data_len = 0;
    size_t pos = 12;
    while (pos + 8 <= all.size()) {
        const uint8_t* hdr = all.data() + pos;
        uint32_t chunk_size = u32(hdr + 4);
        if (std::memcmp(hdr, "fmt ", 4) == 0 && chunk_size >= 16) {
            format = u16(hdr + 8);
            channels = u16(hdr + 10);
            sr = u32(hdr + 12);
            bits = u16(hdr + 22);
            if (format == 0xFFFE && chunk_size >= 40) // WAVE_FORMAT_EXTENSIBLE
                format = u16(hdr + 32);              // SubFormat の先頭2バイト
        } else if (std::memcmp(hdr, "data", 4) == 0) {
            data_off = pos + 8;
            data_len = std::min<size_t>(chunk_size, all.size() - data_off);
            if (format != 0) break; // fmt を読んだ後なら確定
        }
        pos += 8 + chunk_size + (chunk_size & 1);
    }
    if (format != 3 || bits != 32 || channels == 0 || sr == 0 || data_off == 0) {
        error = "unsupported WAV (need IEEE float32): " + path;
        return false;
    }
    const size_t frames = data_len / (sizeof(float) * channels);
    out.sample_rate = (int)sr;
    out.channels = (int)channels;
    out.samples.resize(frames * channels);
    std::memcpy(out.samples.data(), all.data() + data_off, frames * channels * sizeof(float));
    return true;
}

bool write_wav_f32(const std::string& path, const std::vector<float>& planar,
                   int channels, int sample_rate, float scale, std::string& error) {
    const size_t frames = channels > 0 ? planar.size() / channels : 0;
    const uint32_t data_bytes = (uint32_t)(frames * channels * sizeof(float));
    std::ofstream f(path, std::ios::binary);
    if (!f) { error = "cannot write: " + path; return false; }
    put_tag(f, "RIFF");
    put_u32(f, 36 + data_bytes);
    put_tag(f, "WAVE");
    put_tag(f, "fmt ");
    put_u32(f, 16);
    put_u16(f, 3); // IEEE float
    put_u16(f, (uint16_t)channels);
    put_u32(f, (uint32_t)sample_rate);
    put_u32(f, (uint32_t)(sample_rate * channels * sizeof(float)));
    put_u16(f, (uint16_t)(channels * sizeof(float)));
    put_u16(f, 32);
    put_tag(f, "data");
    put_u32(f, data_bytes);
    std::vector<float> inter(frames * channels);
    for (size_t fr = 0; fr < frames; ++fr)
        for (int ch = 0; ch < channels; ++ch)
            inter[fr * channels + ch] = planar[(size_t)ch * frames + fr] * scale;
    f.write((const char*)inter.data(), data_bytes);
    if (!f) { error = "write failed: " + path; return false; }
    return true;
}

} // namespace wavio
