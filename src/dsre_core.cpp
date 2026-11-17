// DSRE DSP コア実装 — Python 版 DSRE.py (lowmem) の各段を精度ごとに対応させた移植。
// 対応表は DESIGN_ja.md の「精度対応表」を参照。
// Phase 2: 進捗/中断コールバックとバッファ再利用。
// v0.1.5: メモリダイエット + 高速化 (計算式・処理順序・精度は不変)。
//   - チャンネル切り出しコピー (channel_of/put_channel) を廃止しポインタで処理
//   - dynamic は包絡を1パスで適用 (smoothed/out の2面を廃止)
//   - blend は work をそのまま out にして in-place 化 (コピー3面を廃止)
//   - filtfilt は double スクラッチ2面を使い回し、filtered の float コピーを廃止
//   - irfft のスペクトルコピーを廃止、bandpass の masked もスクラッチ再利用
//   - generate_harmonics の発振器をブロック漸化式化 (4096サンプル毎に
//     sin/cos を正確に再初期化するので誤差は double 丸め程度、float32
//     量子化より十分小さい)。非有限が出た倍音だけ従来の直接 sin で再計算する。
#include "dsre_core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <new>
#include <thread>

// FFT は pocketfft (numpy と同じエンジン、BSD)。大きな素因数を持つ長さでも
// Bluestein 法で高速に処理でき、KissFFT の素朴経路のような
// 「曲の長さ次第で事実上止まる」事態が起きない。
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace dsre {
namespace {

constexpr double kPi = 3.14159265358979323846;

using VecF = std::vector<float>;
using VecD = std::vector<double>;
using VecC = std::vector<std::complex<double>>;

float audio_peak(const float* x, size_t n) {
    float p = 0.0f;
    for (size_t i = 0; i < n; ++i) p = std::max(p, std::fabs(x[i]));
    return p;
}
float audio_peak(const VecF& x) { return audio_peak(x.data(), x.size()); }

// Python: sanitize_audio — NaN/Inf->0, peak>1000 のときだけ正規化。それ以外は素通し。
void sanitize(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) x[i] = 0.0f;
    float p = audio_peak(x, n);
    if (p > 1000.0f)
        for (size_t i = 0; i < n; ++i) x[i] /= p;
}
void sanitize(VecF& x) { sanitize(x.data(), x.size()); }

// ---- 並列実行 ----
// 並列化は「各要素の計算が逐次版と完全に同一」になる場所にだけ使う:
//   - radix-2 FFT の各段 (バタフライは独立。回転因子は逐次版の 4096 毎の
//     再初期化点で区切るので、因子の列が逐次版と完全一致する)
//   - four-step の行/列変換 (各行・各列は独立した 1 次元変換)
//   - チャンネル単位の段 (psycho/dynamic)
// そのためスレッド数によらず出力はビット単位で一致する (回帰テストで検証)。
// pocketfft 自体は単スレッドのまま (内部並列は丸めが変わり得るため使わない)。
// 入れ子の parallel_for は内側を逐次実行にして過剰なスレッド生成を防ぐ。
int g_threads = 1;
thread_local int tl_par_depth = 0;

int detect_threads() {
    if (const char* e = std::getenv("DSRE_THREADS")) {
        const int v = std::atoi(e);
        if (v >= 1) return std::min(v, 64);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 1;
    return (int)std::min(hw, 8u);
}

template <typename F>
void parallel_for(int count, const F& fn) {
    if (count <= 0) return;
    const int T = (tl_par_depth > 0) ? 1 : std::min(g_threads, count);
    if (T <= 1) {
        for (int i = 0; i < count; ++i) fn(i);
        return;
    }
    std::atomic<int> next{0};
    std::exception_ptr ep;
    std::mutex ep_mu;
    auto worker = [&] {
        tl_par_depth = 1;
        for (;;) {
            const int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) break;
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lk(ep_mu);
                if (!ep) ep = std::current_exception();
                next.store(count, std::memory_order_relaxed);
                break;
            }
        }
        tl_par_depth = 0;
    };
    std::vector<std::thread> ths;
    ths.reserve((size_t)T - 1);
    for (int t = 1; t < T; ++t) ths.emplace_back(worker);
    worker();
    for (auto& th : ths) th.join();
    if (ep) std::rethrow_exception(ep);
}

// ---- FFT (pocketfft, double) ----
// 長さに大きな素因数 (例: 22462660 = 2^2*5*11*102103) があると、素朴な
// バタフライは O(n*p) になって事実上終わらない。pocketfft は Bluestein で
// 処理するが、その内部確保が数GB級になる (実測ピーク 4.2GB) ため、
// 大きな素因数がある長さでは自前の経路を使う:
//   four-step 分解 (n = n1 x p) で素数サイズ p の DFT だけを Bluestein 化し、
//   畳み込み長も 2p 程度に抑える。作業バッファは n サイズ複素 2 面まで。
int largest_prime_factor(int n) {
    int best = 1;
    for (int d = 2; (long long)d * d <= n; ++d) {
        while (n % d == 0) { best = d; n /= d; }
    }
    if (n > 1) best = std::max(best, n);
    return best;
}
bool fft_friendly(int n) { return n >= 1 && largest_prime_factor(n) <= 64; }

int next_smooth(int m) { // 全ての素因数が 7 以下の最小の >= m
    for (int x = m;; ++x)
        if (largest_prime_factor(x) <= 7) return x;
}

void cfft_pocket(const VecC& in, VecC& out, int n, bool inverse) {
    out.assign((size_t)n, {0.0, 0.0});
    pocketfft::c2c<double>(
        pocketfft::shape_t{(size_t)n},
        pocketfft::stride_t{(ptrdiff_t)sizeof(std::complex<double>)},
        pocketfft::stride_t{(ptrdiff_t)sizeof(std::complex<double>)},
        pocketfft::shape_t{0}, !inverse, in.data(), out.data(),
        inverse ? 1.0 / n : 1.0);
}

// e^{-pi i j^2 / p} (j^2 は 2p で割った余りで位相を小さく保ち精度を出す)
std::complex<double> chirp_neg(long long j, int p) {
    double ph = kPi * (double)((j * j) % (2LL * p)) / (double)p;
    return {std::cos(ph), -std::sin(ph)};
}

// Bluestein の畳み込み用 FFT。pocketfft は 1 回の変換で配列の約 2 倍の
// ワークスペースを内部確保する (実測: 850MB の配列でピーク 2.4GB) ため、
// 大きな畳み込み長では自前の in-place radix-2 FFT (追加確保ゼロ) に切り替える。
void fft_radix2(VecC& a, bool inverse) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / (double)len * (inverse ? 1.0 : -1.0);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        const size_t half = len / 2;
        // 作業項目は約 4096 バタフライ単位にまとめる (細かすぎると項目の
        // 取り合いが支配的になり逆に遅くなる)。half が大きい段はブロック内を
        // 4096 単位の j 区間に割り、区間頭は逐次版の位相再初期化点と一致させる。
        // half が小さい段は複数ブロックを 1 項目にまとめ、各ブロック頭で
        // w = 1 から始める (逐次版のブロック処理と完全一致)。
        const size_t blocks = n / len;
        if (half >= 4096) {
            const size_t segs = half / 4096;
            parallel_for((int)(blocks * segs), [&](int item) {
                const size_t i = (size_t)(item / (int)segs) * len;
                const size_t j0 = (size_t)(item % (int)segs) * 4096;
                const size_t j1 = j0 + 4096;
                std::complex<double> w(std::cos(ang * (double)j0),
                                       std::sin(ang * (double)j0));
                for (size_t j = j0; j < j1; ++j) {
                    const std::complex<double> u = a[i + j];
                    const std::complex<double> v = a[i + j + half] * w;
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                    w *= wlen;
                }
            });
        } else {
            const size_t bpi = 4096 / half; // 1 項目あたりのブロック数
            parallel_for((int)((blocks + bpi - 1) / bpi), [&](int item) {
                const size_t b0 = (size_t)item * bpi;
                const size_t b1 = std::min(b0 + bpi, blocks);
                for (size_t blk = b0; blk < b1; ++blk) {
                    const size_t i = blk * len;
                    std::complex<double> w(1.0, 0.0);
                    for (size_t j = 0; j < half; ++j) {
                        const std::complex<double> u = a[i + j];
                        const std::complex<double> v = a[i + j + half] * w;
                        a[i + j] = u + v;
                        a[i + j + half] = u - v;
                        w *= wlen;
                    }
                }
            });
        }
    }
    if (inverse) {
        const double s = 1.0 / (double)n;
        for (auto& v : a) v *= s;
    }
}

void conv_fft(VecC& w, bool inverse) {
    const size_t L = w.size();
    if ((L & (L - 1)) == 0) { fft_radix2(w, inverse); return; }
    const pocketfft::stride_t st{(ptrdiff_t)sizeof(std::complex<double>)};
    pocketfft::c2c<double>(pocketfft::shape_t{L}, st, st,
                           pocketfft::shape_t{0}, !inverse,
                           w.data(), w.data(), inverse ? 1.0 / (double)L : 1.0);
}

int next_pow2(int m) {
    int x = 1;
    while (x < m) x <<= 1;
    return x;
}

// 素数サイズ p の前方 DFT を Bluestein 法で計算する (in/out は長さ p)。
// カーネルの FFT と chirp は p ごとに不変なのでキャッシュして使い回す。
struct BluesteinPlan {
    int L = 0;
    VecC bfft;
    VecC chirp; // e^{-pi i j^2/p}, 長さ p
};
const BluesteinPlan& bluestein_plan(int p) {
    static std::map<int, BluesteinPlan> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(p);
    if (it != cache.end()) return it->second;
    BluesteinPlan pl;
    // 畳み込み長: 小さいうちは 7-smooth (pocketfft)、大きい場合は 2 の冪
    // (自前 radix-2 で追加確保ゼロ) にしてピークメモリを抑える
    pl.L = (2 * p - 1 > (1 << 21)) ? next_pow2(2 * p - 1) : next_smooth(2 * p - 1);
    pl.chirp.resize((size_t)p);
    for (int j = 0; j < p; ++j) pl.chirp[j] = chirp_neg(j, p);
    // カーネル列を pl.bfft に直接置き、その場で FFT してキャッシュする
    // (別バッファに作ってから変換すると L サイズ複素が 1 面余分に要る)
    pl.bfft.assign((size_t)pl.L, {0.0, 0.0});
    for (int j = 0; j < p; ++j) pl.bfft[j] = std::conj(pl.chirp[j]); // e^{+pi i j^2/p}
    for (int j = 1; j < p; ++j) pl.bfft[pl.L - j] = pl.bfft[j];
    conv_fft(pl.bfft, false);
    return cache.emplace(p, std::move(pl)).first->second;
}
// 素数サイズ p の前方 DFT を a の中で直接計算する (コピーを作らない)
void bluestein_inplace(VecC& a, int p) {
    const BluesteinPlan& pl = bluestein_plan(p);
    const int L = pl.L;
    // 作業バッファは 1 面だけ (in-place FFT)。素数長が数千万級でも
    // ピークを L サイズ複素 1 面 + キャッシュ済みカーネルに抑える。
    VecC w((size_t)L, {0.0, 0.0});
    for (int j = 0; j < p; ++j) w[j] = a[j] * pl.chirp[j];
    conv_fft(w, false);
    for (int i = 0; i < L; ++i) w[i] *= pl.bfft[i];
    conv_fft(w, true);
    for (int k = 0; k < p; ++k) a[k] = pl.chirp[k] * w[k];
}

void cfft_fwd(VecC& a, int n); // 前方宣言 (相互再帰)

// four-step: n = n1 * n2 (n2 = 最大の素因数)。in-place で a を前方 DFT する。
void cfft_fourstep(VecC& a, int n, int n1, int n2) {
    // buf[j2][j1] = a[j1*n2 + j2] (転置して各 DFT を連続メモリにする)
    VecC buf((size_t)n), res((size_t)n);
    for (int j1 = 0; j1 < n1; ++j1)
        for (int j2 = 0; j2 < n2; ++j2)
            buf[(size_t)j2 * n1 + j1] = a[(size_t)j1 * n2 + j2];
    // 手順1: j1 方向の DFT (各行は独立した長さ n1 の変換なので並列化できる)
    parallel_for(n2, [&](int j2) {
        VecC row((size_t)n1);
        std::copy_n(buf.begin() + (size_t)j2 * n1, (size_t)n1, row.begin());
        cfft_fwd(row, n1);
        std::copy_n(row.begin(), (size_t)n1, res.begin() + (size_t)j2 * n1);
    });
    // 手順2: 回転因子 e^{-2pi i j2 k1 / n}
    for (int j2 = 0; j2 < n2; ++j2)
        for (int k1 = 0; k1 < n1; ++k1) {
            double ph = -2.0 * kPi * (double)j2 * (double)k1 / (double)n;
            res[(size_t)j2 * n1 + k1] *= std::complex<double>(std::cos(ph), std::sin(ph));
        }
    // 手順3: j2 方向の DFT (サイズ n2、素数なら Bluestein 経路)。列ごとに並列化
    parallel_for(n1, [&](int k1) {
        VecC col((size_t)n2);
        for (int j2 = 0; j2 < n2; ++j2) col[j2] = res[(size_t)j2 * n1 + k1];
        cfft_fwd(col, n2);
        for (int k2 = 0; k2 < n2; ++k2) buf[(size_t)k2 * n1 + k1] = col[k2];
    });
    a.swap(buf);
}

void cfft_fwd(VecC& a, int n) {
    if (n <= 1) return;
    if (fft_friendly(n)) {
        VecC out;
        cfft_pocket(a, out, n, false);
        a.swap(out);
        return;
    }
    const int p = largest_prime_factor(n);
    const int n1 = n / p;
    if (n1 == 1) { // n は素数
        bluestein_inplace(a, n);
        return;
    }
    cfft_fourstep(a, n, n1, p);
}

// in-place 複素 DFT。inverse は 1/n 正規化込み (numpy の ifft と同じ)
void cfft(VecC& a, int n, bool inverse) {
    if (!inverse) { cfft_fwd(a, n); return; }
    for (auto& v : a) v = std::conj(v);
    cfft_fwd(a, n);
    const double s = 1.0 / n;
    for (auto& v : a) v = std::conj(v) * s;
}

// 大きな素因数がある長さ用の rfft: 偶数長は実数パッキング + cfft、
// 奇数長は複素埋め込み + cfft で計算する (結果は pocketfft 経路と一致)。
VecC rfft_bad(const float* x, int n) {
    if (n % 2 == 0) {
        const int N = n / 2;
        VecC F(N);
        for (int j = 0; j < N; ++j) F[j] = {(double)x[2 * j], (double)x[2 * j + 1]};
        cfft(F, N, false);
        VecC out(N + 1);
        for (int k = 0; k < N; ++k) {
            std::complex<double> Fk = F[k];
            std::complex<double> Fmk = std::conj(k == 0 ? F[0] : F[N - k]);
            std::complex<double> E = (Fk + Fmk) * 0.5;
            std::complex<double> d = (Fk - Fmk) * 0.5; // O は d / i
            std::complex<double> O(d.imag(), -d.real());
            double ph = -2.0 * kPi * (double)k / (double)n;
            std::complex<double> tw(std::cos(ph), std::sin(ph));
            out[k] = E + tw * O;
        }
        out[N] = {F[0].real() - F[0].imag(), 0.0};
        return out;
    }
    VecC F(n);
    for (int j = 0; j < n; ++j) F[j] = {(double)x[j], 0.0};
    cfft(F, n, false);
    F.resize(n / 2 + 1);
    return F;
}

VecF irfft_bad(const VecC& spec, int n) {
    VecC full(n);
    const int half = n / 2;
    for (int k = 0; k <= half; ++k) full[k] = spec[k];
    for (int k = half + 1; k < n; ++k) full[k] = std::conj(spec[n - k]);
    cfft(full, n, true);
    VecF out(n);
    for (int i = 0; i < n; ++i) out[i] = (float)full[i].real();
    return out;
}

VecC rfft(const float* x, int n) {
    if (!fft_friendly(n)) return rfft_bad(x, n);
    VecC out(n / 2 + 1);
    VecD in(n);
    for (int i = 0; i < n; ++i) in[i] = (double)x[i];
    pocketfft::r2c<double>(
        pocketfft::shape_t{(size_t)n},
        pocketfft::stride_t{(ptrdiff_t)sizeof(double)},
        pocketfft::stride_t{(ptrdiff_t)sizeof(std::complex<double>)},
        0, true, in.data(), out.data(), 1.0);
    return out;
}

VecF irfft(const VecC& spec, int n) {
    if (!fft_friendly(n)) return irfft_bad(spec, n);
    VecD outd(n);
    pocketfft::c2r<double>(
        pocketfft::shape_t{(size_t)n},
        pocketfft::stride_t{(ptrdiff_t)sizeof(std::complex<double>)},
        pocketfft::stride_t{(ptrdiff_t)sizeof(double)},
        0, false, spec.data(), outd.data(), 1.0);
    VecF out(n);
    for (int i = 0; i < n; ++i) out[i] = (float)(outd[i] / n); // numpy irfft の 1/n スケール
    return out;
}

double band_mask(double freq, double low_hz, double high_hz, double nyquist) {
    const double bw = std::max(20.0, (high_hz - low_hz) * 0.15);
    const double low1 = std::max(0.0, low_hz - bw);
    const double low2 = low_hz;
    const double high1 = high_hz;
    const double high2 = std::min(nyquist, high_hz + bw);
    if (freq >= low1 && freq < low2) {
        double t = (freq - low1) / std::max(1e-12, low2 - low1);
        return 0.5 - 0.5 * std::cos(kPi * t);
    }
    if (freq >= low2 && freq <= high1) return 1.0;
    if (freq > high1 && freq <= high2) {
        double t = (freq - high1) / std::max(1e-12, high2 - high1);
        return 0.5 + 0.5 * std::cos(kPi * t);
    }
    return 0.0;
}

VecF bandpass_from_spec(const VecC& spec, VecC& masked_scratch, int n, int sr,
                        double low_hz, double high_hz) {
    const double nyquist = sr / 2.0;
    low_hz = std::max(1.0, low_hz);
    high_hz = std::min(high_hz, nyquist - 1.0);
    if (low_hz >= high_hz) return VecF(n, 0.0f);
    masked_scratch = spec;
    for (size_t k = 0; k < masked_scratch.size(); ++k) {
        double freq = (double)k * sr / n;
        masked_scratch[k] *= band_mask(freq, low_hz, high_hz, nyquist);
    }
    return irfft(masked_scratch, n);
}

// Python generate_harmonics: 発振器は double で sin、乗算・加算は float32。
// band は所有権ごと受け取り in-place で使う。content はスクラッチ再利用。
VecF generate_harmonics(VecF band, double fundamental_freq, int sr,
                        int num_harmonics, double harmonic_strength, VecF& content) {
    sanitize(band);
    if (band.empty()) return band;
    VecF enhanced = band;
    const int n = (int)band.size();
    content.assign(n, 0.0f);
    constexpr int kBlock = 4096;
    for (int h = 2; h < num_harmonics + 2; ++h) {
        double harmonic_freq = fundamental_freq * h;
        if (!(harmonic_freq < sr / 2.0)) continue;
        const double phase_increment = 2.0 * kPi * harmonic_freq / sr;
        if (!std::isfinite(phase_increment)) continue;
        const float scale = (float)(harmonic_strength / h);
        bool ok = true;
        // ブロック漸化式: ブロック頭で位相を正確に置き、内部は回転で進める
        const double step_s = std::sin(phase_increment);
        const double step_c = std::cos(phase_increment);
        for (int base = 0; base < n && ok; base += kBlock) {
            const int len = std::min(kBlock, n - base);
            double s = std::sin(phase_increment * (double)base);
            double c = std::cos(phase_increment * (double)base);
            for (int j = 0; j < len; ++j) {
                const int i = base + j;
                float osc = (float)s;
                if (!std::isfinite(osc)) { ok = false; break; }
                content[i] = band[i] * osc * scale;
                if (!std::isfinite(content[i])) { ok = false; break; }
                const double ns = s * step_c + c * step_s;
                c = c * step_c - s * step_s;
                s = ns;
            }
        }
        if (!ok) {
            // 従来どおり直接 sin で作り直し、それでも駄目なら倍音ごと捨てる
            for (int i = 0; i < n; ++i) {
                float osc = (float)std::sin(phase_increment * (double)i);
                if (!std::isfinite(osc)) { ok = false; break; }
                content[i] = band[i] * osc * scale;
                if (!std::isfinite(content[i])) { ok = false; break; }
            }
            if (!ok) continue;
        }
        for (int i = 0; i < n; ++i) enhanced[i] += content[i];
    }
    VecF out = enhanced;
    sanitize(out);
    return out;
}

struct BandDef { const char* name; double low, high, gain; int harmonics; double strength; };

bool multiband_exciter_ch(const float* x_ch, int n, int sr, double harmonic_intensity,
                          const Callbacks& cb, int ch_index, int channels, float* out_ch) {
    const int nyquist_i = sr / 2;
    const double base = std::clamp(harmonic_intensity, 0.1, 1.5);
    const BandDef defs[] = {
        {"Sub Bass", 20, 80, 1.10, 3, 0.08},
        {"Bass", 80, 250, 1.20, 4, 0.14},
        {"Low Mid", 250, 800, 1.25, 5, 0.18},
        {"Mid", 800, 2500, 1.35, 6, 0.22},
        {"High Mid", 2500, 8000, 1.45, 4, 0.25},
        {"Presence", 8000, 16000, 1.50, 3, 0.18},
        {"Air", 16000, std::min(20000.0, (double)nyquist_i - 1000), 1.35, 2, 0.12},
    };
    VecC spec = rfft(x_ch, n); // FFT はチャンネルで1回だけ
    VecC masked_scratch;
    VecF content_scratch;
    std::memcpy(out_ch, x_ch, sizeof(float) * (size_t)n);
    int band_i = 0;
    for (const BandDef& b : defs) {
        if (cb.aborted()) return false;
        if (cb.progress) {
            double frac = (ch_index + (double)band_i / 7.0) / channels;
            cb.progress((int)(frac * 45), "exciter");
        }
        ++band_i;
        if (!(b.low < nyquist_i && b.high < nyquist_i && b.high > b.low)) continue;
        VecF band_signal = bandpass_from_spec(spec, masked_scratch, n, sr, b.low, b.high);
        sanitize(band_signal);
        if (audio_peak(band_signal) < 1e-8f) continue;
        const double center = (b.low + b.high) / 2.0;
        const double strength = b.strength * base;
        VecF harmonics = generate_harmonics(std::move(band_signal), center, sr,
                                            b.harmonics, strength, content_scratch);
        bool finite = true;
        for (int i = 0; i < n; ++i) {
            float s = std::tanh(harmonics[i] * 1.35f) * 0.82f;
            float be = s * (float)b.gain;
            if (!std::isfinite(be)) { finite = false; break; }
            harmonics[i] = be; // band_enhanced を harmonics バッファに流用
        }
        if (!finite) continue;
        for (int i = 0; i < n; ++i) out_ch[i] += harmonics[i] * 0.22f;
    }
    sanitize(out_ch, (size_t)n);
    return true;
}

// ---- psychoacoustic ----
void design_peaking_eq(double freq, double gain_db, double q, int sr,
                       double b[3], double a[3]) {
    const double A = std::pow(10.0, gain_db / 40.0);
    const double w0 = 2.0 * kPi * freq / sr;
    const double alpha = std::sin(w0) / (2.0 * q);
    const double cw = std::cos(w0);
    b[0] = 1 + alpha * A;  b[1] = -2 * cw;  b[2] = 1 - alpha * A;
    a[0] = 1 + alpha / A;  a[1] = -2 * cw;  a[2] = 1 - alpha / A;
}

// x (double, 長さ m) を in-place で IIR 濾波する (従来 apply_iir と同一の漸化式)
void apply_iir_inplace(const double b_in[3], const double a_in[3], double* x, int m) {
    double b[3], a[3];
    for (int i = 0; i < 3; ++i) { b[i] = b_in[i] / a_in[0]; a[i] = a_in[i] / a_in[0]; }
    double y0 = 0.0, y1 = 0.0; // y[i-2], y[i-1]
    double x0 = 0.0, x1 = 0.0; // x[i-2], x[i-1] (読み終わった元の値)
    for (int i = 0; i < m; ++i) {
        const double xi = x[i];
        double acc = b[0] * xi;
        if (i - 1 >= 0) acc += b[1] * x1;
        if (i - 2 >= 0) acc += b[2] * x0;
        if (i - 1 >= 0) acc -= a[1] * y1;
        if (i - 2 >= 0) acc -= a[2] * y0;
        x[i] = acc;
        x0 = x1; x1 = xi;
        y0 = y1; y1 = acc;
    }
}

// filtfilt: 結果は work (double, 長さ n+2*pad) に入り、先頭 pad 分がパディング。
// 戻り値は pad。scratch は呼び出し側で使い回す作業領域 (上書きされる)。
int filtfilt_to(const double b[3], const double a[3], const float* x_in, int n,
                VecD& work) {
    if (n < 9) {
        work.assign(x_in, x_in + n);
        apply_iir_inplace(b, a, work.data(), n);
        return 0;
    }
    const int pad = std::min(n - 1, 9);
    work.assign((size_t)n + 2 * (size_t)pad, 0.0);
    for (int i = 0; i < pad; ++i) work[i] = 2.0 * x_in[0] - x_in[pad - i];
    for (int i = 0; i < n; ++i) work[pad + i] = (double)x_in[i];
    for (int i = 0; i < pad; ++i) work[pad + n + i] = 2.0 * x_in[n - 1] - x_in[n - 2 - i];
    const int m = n + 2 * pad;
    apply_iir_inplace(b, a, work.data(), m);
    std::reverse(work.begin(), work.end());
    apply_iir_inplace(b, a, work.data(), m);
    std::reverse(work.begin(), work.end());
    return pad;
}

// src (元のチャンネル) を各バンドで濾波して dst へ blend する。dst は src の
// コピーから始める。従来版の filtered (float) + sanitize + blend と等価。
bool psychoacoustic_inplace(const float* src, float* dst, int n, int sr,
                            double strength, const Callbacks& cb, VecD& scratch) {
    const double scale = std::clamp(strength, 0.3, 1.5);
    struct CB { double freq, boost, q; };
    const CB bands[] = {
        {1000, 1.0 * scale, 1.5}, {2500, 1.8 * scale, 2.0}, {4000, 2.0 * scale, 1.8},
        {6000, 1.5 * scale, 1.2}, {10000, 1.0 * scale, 0.8},
    };
    if (dst != src) std::memcpy(dst, src, sizeof(float) * (size_t)n);
    for (const CB& band : bands) {
        if (cb.aborted()) return false;
        if (band.freq >= sr / 2) continue;
        double b[3], a[3];
        design_peaking_eq(band.freq, band.boost, band.q, sr, b, a);
        const int pad = filtfilt_to(b, a, src, n, scratch);
        const double* y = scratch.data() + pad;
        // sanitize(filtered) 相当: 非有限->0、peak>1000 のときだけ割る
        float peak = 0.0f;
        for (int i = 0; i < n; ++i) {
            float f = (float)y[i];
            if (!std::isfinite(f)) f = 0.0f;
            peak = std::max(peak, std::fabs(f));
        }
        const float blend = 0.22f;
        for (int i = 0; i < n; ++i) {
            float f = (float)y[i];
            if (!std::isfinite(f)) f = 0.0f;
            if (peak > 1000.0f) f /= peak; // sanitize の割り算をそのまま再現
            dst[i] = dst[i] * (1.0f - blend) + f * blend;
        }
    }
    sanitize(dst, (size_t)n);
    return true;
}

// ---- dynamic range (float32 漸化包絡、1パス in-place) ----
// 従来版は smoothed を全長に作ってから gain を掛けていたが、包絡は過去だけに
// 依存するので同じ順序でその場で適用できる (結果は同一)。
void dynamic_range_inplace(float* x, int n, double ratio_in, int sr) {
    const float ratio = (float)std::clamp(ratio_in, 1.0, 1.5);
    const int attack_samples = std::max(1, (int)(5.0 * sr / 1000));
    const int release_samples = std::max(1, (int)(50.0 * sr / 1000));
    if (n == 0) return;
    const float threshold = 0.08f;
    float current = std::fabs(x[0]);
    for (int i = 0; i < n; ++i) {
        float e = std::fabs(x[i]);
        if (e > current) current += (e - current) / (float)attack_samples;
        else current -= (current - e) / (float)release_samples;
        float gain = 1.0f;
        if (current > threshold)
            gain = std::pow(current / threshold, ratio - 1.0f);
        gain = std::clamp(gain, 1.0f, 1.8f);
        x[i] = x[i] * gain;
    }
    sanitize(x, (size_t)n);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void report(const Callbacks& cb, int pct, const char* stage) {
    if (cb.progress) cb.progress(pct, stage);
}

} // namespace

bool process(const std::vector<float>& x, int channels, int sr,
             const Params& p, std::vector<float>& out,
             const Callbacks& cb, StageTimes* times) {
    const int frames = channels > 0 ? (int)(x.size() / channels) : 0;
    StageTimes t;
    double t0;
    g_threads = detect_threads();

    std::vector<float> work(x.size());
    t0 = now_s();
    for (int ch = 0; ch < channels; ++ch) {
        const float* src = x.data() + (size_t)ch * frames;
        float* dst = work.data() + (size_t)ch * frames;
        if (!multiband_exciter_ch(src, frames, sr, p.harmonic_intensity, cb, ch, channels, dst))
            return false;
    }
    t.exciter = now_s() - t0;
    report(cb, 45, "exciter");

    t0 = now_s();
    if (p.use_psycho) {
        // psycho は「元の」チャンネルを毎バンド濾波するため src と dst を分ける。
        // チャンネルは互いに独立なので並列化する (チャンネル内の計算は逐次版と同一)。
        std::atomic<bool> ok{true};
        parallel_for(channels, [&](int ch) {
            if (cb.aborted()) { ok.store(false); return; }
            VecF chan((size_t)frames);
            VecD scratch;
            float* wch = work.data() + (size_t)ch * frames;
            std::memcpy(chan.data(), wch, sizeof(float) * (size_t)frames);
            if (!psychoacoustic_inplace(chan.data(), wch, frames, sr,
                                        p.enhancement_strength, cb, scratch))
                ok.store(false);
        });
        if (!ok.load()) return false;
    }
    t.psycho = now_s() - t0;
    report(cb, 70, "psychoacoustic");

    t0 = now_s();
    if (cb.aborted()) return false;
    parallel_for(channels, [&](int ch) {
        dynamic_range_inplace(work.data() + (size_t)ch * frames, frames,
                              p.dynamic_enhancement, sr);
    });
    t.dynamic = now_s() - t0;
    report(cb, 82, "dynamic");

    t0 = now_s();
    if (channels == 2) {
        const float w = std::clamp(p.stereo_width, 1.0f, 1.8f);
        float* l = work.data();
        float* r = work.data() + frames;
        for (int i = 0; i < frames; ++i) {
            float mid = (l[i] + r[i]) / 2.0f;
            float side = (l[i] - r[i]) / 2.0f * w;
            l[i] = mid + side;
            r[i] = mid - side;
        }
    }
    t.stereo = now_s() - t0;
    report(cb, 90, "stereo");

    t0 = now_s();
    out = std::move(work); // blend は out (=処理済み) へ原音を混ぜる in-place で行う
    const float blend = (float)std::clamp(p.enhancement_strength, 0.1, 0.8);
    for (int ch = 0; ch < channels; ++ch) {
        if (cb.aborted()) return false;
        const float* xc = x.data() + (size_t)ch * frames;
        float* oc = out.data() + (size_t)ch * frames;
        if (audio_peak(oc, (size_t)frames) < 1e-10f) {
            std::memcpy(oc, xc, sizeof(float) * (size_t)frames);
        } else {
            for (int i = 0; i < frames; ++i)
                oc[i] = xc[i] * (1.0f - blend) + oc[i] * blend;
        }
        sanitize(oc, (size_t)frames);
    }
    float peak = audio_peak(out);
    if (peak > 0.95f) {
        float s = 0.95f / peak;
        for (float& v : out) v *= s;
    }
    t.blend = now_s() - t0;
    report(cb, 100, "blend");

    if (times) *times = t;
    return true;
}

} // namespace dsre
