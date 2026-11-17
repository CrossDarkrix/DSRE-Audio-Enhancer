#include "dsre_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "dsre_core.hpp"
#include "runner.hpp"
#include "wav_io.hpp"

namespace fs = std::filesystem;

namespace pipeline {
namespace {

void log_line(const FileCallbacks& cb, const std::string& s) {
    if (cb.log) cb.log(s);
}
void prog(const FileCallbacks& cb, int pct, const std::string& stage) {
    if (cb.progress) cb.progress(pct, stage);
}

long long pid_value() {
#ifdef _WIN32
    return (long long)GetCurrentProcessId();
#else
    return (long long)getpid();
#endif
}

std::string ffmpeg_path() {
    static std::string cached = runner::find_tool("ffmpeg");
    return cached;
}

fs::path make_temp_dir() {
    static int counter = 0;
    std::error_code ec;
    fs::path dir = fs::temp_directory_path(ec) /
                   ("dsre_cpp_" + std::to_string(pid_value()) + "_" + std::to_string(++counter));
    fs::create_directories(dir, ec);
    return dir;
}

float peak_of(const std::vector<float>& x) {
    float p = 0.0f;
    for (float v : x) p = std::max(p, std::fabs(v));
    return p;
}

} // namespace

bool process_file_impl(const std::string& in_path, const FileParams& params,
                       const FileCallbacks& cb, std::string& out_path, std::string& error) {
    const std::string ffmpeg = ffmpeg_path();
    if (ffmpeg.empty()) {
        error = "FFmpeg not found. Place ffmpeg in PATH or in the 'ffmpeg' folder next to the executable.";
        return false;
    }
    std::error_code ec;
    if (!fs::is_regular_file(in_path, ec)) {
        error = "Input file not found: " + in_path;
        return false;
    }
    std::string fmt = params.format;
    for (char& c : fmt) c = (char)std::toupper((unsigned char)c);
    if (fmt != "ALAC" && fmt != "FLAC" && fmt != "MP3") {
        error = "Unsupported format: " + params.format;
        return false;
    }

    fs::path tmp_dir = make_temp_dir();
    struct TempGuard {
        fs::path dir;
        ~TempGuard() { std::error_code e; fs::remove_all(dir, e); }
    } guard{tmp_dir};

    // ---- decode (Python decode_audio_to_temp_wav と同一引数) ----
    prog(cb, 5, "decode");
    fs::path tmp_wav = tmp_dir / "decoded.wav";
    {
        std::vector<std::string> cmd = {ffmpeg, "-y", "-i", in_path,
                                        "-vn", "-sn", "-dn", "-map", "0:a:0",
                                        "-ar", std::to_string(params.target_sr),
                                        "-c:a", "pcm_f32le", tmp_wav.string()};
        std::string err;
        if (runner::run(cmd, &err) != 0) {
            error = "FFmpeg decode failed:\n" + err;
            return false;
        }
    }
    if (cb.aborted()) { error = "aborted"; return false; }

    wavio::WavData wav;
    if (!wavio::read_wav_f32(tmp_wav.string(), wav, error)) return false;
    const int channels = wav.channels;
    const int sr = wav.sample_rate;
    const size_t frames = channels > 0 ? wav.samples.size() / channels : 0;
    // interleaved -> planar
    std::vector<float> planar(wav.samples.size());
    for (size_t fr = 0; fr < frames; ++fr)
        for (int ch = 0; ch < channels; ++ch)
            planar[(size_t)ch * frames + fr] = wav.samples[fr * channels + ch];
    wav.samples.clear();
    wav.samples.shrink_to_fit();

    if (peak_of(planar) < 1e-10f) {
        error = "Input audio file appears to be silent or corrupted";
        return false;
    }
    log_line(cb, "decoded: " + std::to_string(channels) + "ch " +
                 std::to_string(sr) + "Hz " + std::to_string(frames) + " frames");
    prog(cb, 15, "decode");

    // ---- DSP (進捗は Python ワーカーと同じく 15..85 に割り当てる) ----
    dsre::Params dp;
    dp.harmonic_intensity = params.m / 16.0;
    dp.enhancement_strength = params.strength;
    dp.stereo_width = 1.15f;
    dp.dynamic_enhancement = 1.12;
    dp.use_psycho = true;
    dsre::Callbacks dcb;
    dcb.abort = cb.abort;
    std::string last_stage;
    dcb.progress = [&cb, &last_stage](int pct, const char* stage) {
        if (stage && last_stage != stage) { // 段の遷移をログに残す (落下段の特定用)
            last_stage = stage;
            log_line(cb, std::string("stage: ") + stage);
        }
        prog(cb, 15 + (int)(pct * 0.70), stage);
    };
    log_line(cb, "DSP start");
    std::vector<float> result;
    if (!dsre::process(planar, channels, sr, dp, result, dcb)) {
        error = "aborted";
        return false;
    }
    log_line(cb, "DSP done");
    if (peak_of(result) < 1e-10f) {
        log_line(cb, "WARNING: Output audio is silent. Falling back to original.");
        result = planar;
    }

    // ---- save (Python save_with_metadata_from_memmap と同一手順) ----
    prog(cb, 88, "save");
    float scale = 1.0f;
    float peak = peak_of(result);
    if (peak > 1.0f) scale = 1.0f / peak; // normalize=True 相当
    fs::path out_wav = tmp_dir / "enhanced.wav";
    if (!wavio::write_wav_f32(out_wav.string(), result, channels, sr, scale, error))
        return false;
    result.clear();
    result.shrink_to_fit();

    // cover (失敗しても続行するのが Python の仕様)
    fs::path cover = tmp_dir / "cover.jpg";
    bool has_cover = false;
    {
        std::vector<std::string> cmd = {ffmpeg, "-y", "-i", in_path, "-an",
                                        "-map", "0:v:0", "-frames:v", "1", cover.string()};
        std::string err;
        int rc = runner::run(cmd, &err);
        std::error_code ec2;
        has_cover = rc == 0 && fs::is_regular_file(cover, ec2) && fs::file_size(cover, ec2) > 0;
    }

    const std::string codec = fmt == "ALAC" ? "alac" : fmt == "FLAC" ? "flac" : "libmp3lame";
    const std::string sfmt = fmt == "ALAC" ? "s32p" : fmt == "FLAC" ? "s32" : "s16p";
    const std::string ext = fmt == "ALAC" ? "m4a" : fmt == "FLAC" ? "flac" : "mp3";
    fs::path in_p(in_path);
    fs::path out_dir = params.out_dir.empty() ? in_p.parent_path() : fs::path(params.out_dir);
    fs::create_directories(out_dir, ec);
    fs::path out_p = out_dir / (in_p.stem().string() + "_enhanced." + ext);

    std::vector<std::string> cmd;
    if (has_cover) {
        cmd = {ffmpeg, "-y", "-i", out_wav.string(), "-i", in_path, "-i", cover.string(),
               "-map", "0:a", "-map", "2:v"};
        if (fmt == "MP3") {
            cmd.insert(cmd.end(), {"-map_metadata", "1", "-id3v2_version", "3",
                                   "-c:a", codec, "-sample_fmt", sfmt, "-b:a", "320k",
                                   "-c:v", "mjpeg"});
        } else {
            cmd.insert(cmd.end(), {"-disposition:v", "attached_pic", "-map_metadata", "1",
                                   "-c:a", codec, "-sample_fmt", sfmt, "-c:v", "copy"});
        }
    } else {
        cmd = {ffmpeg, "-y", "-i", out_wav.string(), "-i", in_path,
               "-map", "0:a", "-map_metadata", "1",
               "-c:a", codec, "-sample_fmt", sfmt};
        if (fmt == "MP3") cmd.insert(cmd.end(), {"-b:a", "320k"});
    }
    cmd.push_back(out_p.string());

    log_line(cb, "encode start: " + out_p.string());
    {
        std::string err;
        if (runner::run(cmd, &err) != 0) {
            error = "FFmpeg encode failed:\n" + err;
            return false;
        }
    }
    std::error_code ec3;
    if (!fs::is_regular_file(out_p, ec3) || fs::file_size(out_p, ec3) < 1000) {
        error = "Output file was not created correctly: " + out_p.string();
        return false;
    }
    prog(cb, 100, "done");
    out_path = out_p.string();
    return true;
}

// 例外 (bad_alloc 等) でプロセスごと落ちないよう、ここで受けてエラーに変換する。
// どの段で落ちたかは cb.log のステージ記録から追える。
bool process_file(const std::string& in_path, const FileParams& params,
                  const FileCallbacks& cb, std::string& out_path, std::string& error) {
    try {
        return process_file_impl(in_path, params, cb, out_path, error);
    } catch (const std::bad_alloc&) {
        error = "Out of memory during processing (file too long for available RAM?)";
    } catch (const std::exception& e) {
        error = std::string("Exception during processing: ") + e.what();
    } catch (...) {
        error = "Unknown exception during processing";
    }
    return false;
}

} // namespace pipeline
