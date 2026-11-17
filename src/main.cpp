// dsre_bench: 検証用 CLI ハーネス。float32 平面 (ch x frames) の raw を読み、処理して raw で書く。
// usage: dsre_bench <in.f32> <out.f32> <channels> <frames> <sr>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dsre_core.hpp"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: dsre_bench <in.f32> <out.f32> <channels> <frames> <sr>\n");
        return 2;
    }
    const int channels = std::atoi(argv[3]);
    const long frames = std::atol(argv[4]);
    const int sr = std::atoi(argv[5]);
    std::vector<float> x((size_t)channels * frames);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror("in"); return 1; }
    if (std::fread(x.data(), sizeof(float), x.size(), f) != x.size()) {
        std::fprintf(stderr, "short read\n");
        return 1;
    }
    std::fclose(f);

    dsre::Params p; // GUI 既定値
    dsre::StageTimes t;
    std::vector<float> y;
    if (!dsre::process(x, channels, sr, p, y, {}, &t)) return 1;
    std::printf("exciter: %.3fs\npsycho: %.3fs\ndynamic: %.3fs\nstereo: %.3fs\nblend: %.3fs\ntotal: %.3fs\n",
                t.exciter, t.psycho, t.dynamic, t.stereo, t.blend,
                t.exciter + t.psycho + t.dynamic + t.stereo + t.blend);

    f = std::fopen(argv[2], "wb");
    if (!f) { std::perror("out"); return 1; }
    std::fwrite(y.data(), sizeof(float), y.size(), f);
    std::fclose(f);
    return 0;
}
