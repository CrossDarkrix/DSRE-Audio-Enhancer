// dsre_cli — DSRE コンソール版。バッチ処理・進捗表示・Ctrl+C での安全な中断。
// 使い方: dsre_cli [options] <file-or-dir> [...]
//   --m N            倍音の強さ 1..32 (既定 16)
//   --strength F     強調の強さ 0.1..1.0 (既定 0.7)
//   --sr N           目標サンプルレート (既定 96000)
//   --format F       ALAC | FLAC | MP3 (既定 ALAC)
//   --outdir DIR     出力フォルダ (既定 enhanced_output)
//   --lang en|ja     表示言語 (既定は環境から判定)
//   --recursive      フォルダを再帰走査
#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "dsre_pipeline.hpp"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_abort{false};
void on_sigint(int) { g_abort.store(true); }

struct Texts {
    const char* usage;
    const char* processing;
    const char* done;
    const char* failed;
    const char* aborted;
    const char* summary;
    const char* no_files;
    const char* scanning;
};

const Texts EN = {
    "Usage: dsre_cli [options] <file-or-dir> [...]",
    "Processing",
    "Done",
    "Failed",
    "Aborted by user",
    "Summary: %d succeeded, %d failed, %d skipped\n",
    "No audio files found.",
    "Scanning",
};
const Texts JA = {
    "使い方: dsre_cli [options] <ファイルまたはフォルダ> [...]",
    "処理中",
    "完了",
    "失敗",
    "ユーザーにより中断されました",
    "結果: 成功 %d, 失敗 %d, スキップ %d\n",
    "音声ファイルが見つかりませんでした。",
    "走査中",
};

bool is_audio(const fs::path& p) {
    std::string ext = p.extension().string();
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext == ".wav" || ext == ".flac" || ext == ".mp3" || ext == ".m4a" ||
           ext == ".aac" || ext == ".ogg" || ext == ".opus" || ext == ".aif" ||
           ext == ".aiff" || ext == ".wma";
}

} // namespace

int main(int argc, char** argv) {
    // 言語: --lang > DSRE_LANG > ロケール環境 > en
    std::string lang;
    {
        const char* e = std::getenv("DSRE_LANG");
        if (e) lang = e;
        if (lang.empty()) {
            const char* l = std::getenv("LANG");
            if (l && std::string(l).find("ja") != std::string::npos) lang = "ja";
        }
    }

    pipeline::FileParams params;
    params.out_dir = "enhanced_output";
    bool recursive = false;
    std::vector<std::string> inputs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--m") params.m = std::stoi(next("--m"));
        else if (a == "--strength") params.strength = std::stod(next("--strength"));
        else if (a == "--sr") params.target_sr = std::stoi(next("--sr"));
        else if (a == "--format") params.format = next("--format");
        else if (a == "--outdir") params.out_dir = next("--outdir");
        else if (a == "--lang") lang = next("--lang");
        else if (a == "--recursive") recursive = true;
        else if (a == "--help" || a == "-h") {
            const Texts& t = (lang == "ja") ? JA : EN;
            std::printf("%s\n", t.usage);
            return 0;
        } else inputs.push_back(a);
    }
    const Texts& T = (lang == "ja") ? JA : EN;
    if (inputs.empty()) {
        std::printf("%s\n", T.usage);
        return 2;
    }

    std::signal(SIGINT, on_sigint);
#ifdef SIGTERM
    std::signal(SIGTERM, on_sigint);
#endif

    // 入力をファイル一覧に展開
    std::vector<fs::path> files;
    for (const std::string& in : inputs) {
        fs::path p(in);
        std::error_code ec;
        if (fs::is_regular_file(p, ec)) {
            files.push_back(p);
        } else if (fs::is_directory(p, ec)) {
            if (recursive) {
                for (auto iter = fs::recursive_directory_iterator(p, ec);
                     iter != fs::recursive_directory_iterator(); iter.increment(ec)) {
                    if (iter->is_regular_file(ec) && is_audio(iter->path())) files.push_back(iter->path());
                }
            } else {
                for (auto iter = fs::directory_iterator(p, ec);
                     iter != fs::directory_iterator(); iter.increment(ec)) {
                    if (iter->is_regular_file(ec) && is_audio(iter->path())) files.push_back(iter->path());
                }
            }
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::printf("%s\n", T.no_files);
        return 1;
    }

    int ok = 0, ng = 0, skipped = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        if (g_abort.load()) { skipped = (int)(files.size() - i); break; }
        const fs::path& f = files[i];
        std::printf("[%zu/%zu] %s: %s\n", i + 1, files.size(), T.processing, f.string().c_str());
        std::fflush(stdout);
        int last_pct = -1;
        pipeline::FileCallbacks cb;
        cb.abort = &g_abort;
        cb.progress = [&last_pct](int pct, const std::string& stage) {
            if (pct != last_pct && pct % 5 == 0) {
                last_pct = pct;
                std::printf("  %3d%% %s\n", pct, stage.c_str());
                std::fflush(stdout);
            }
        };
        cb.log = [](const std::string& m) { std::printf("  %s\n", m.c_str()); std::fflush(stdout); };
        std::string out_path, error;
        if (pipeline::process_file(f.string(), params, cb, out_path, error)) {
            std::printf("  %s: %s\n", T.done, out_path.c_str());
            ++ok;
        } else if (g_abort.load()) {
            std::printf("  %s\n", T.aborted);
            skipped = (int)(files.size() - i - 1);
            break;
        } else {
            std::printf("  %s: %s\n", T.failed, error.c_str());
            ++ng;
        }
    }
    std::printf(T.summary, ok, ng, skipped);
    return ng > 0 ? 1 : 0;
}
