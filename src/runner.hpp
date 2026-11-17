#pragma once
// 外部プロセス実行 (ffmpeg 用)。argv 配列で渡し、シェルを経由しない。
#include <string>
#include <vector>

namespace runner {

// 終了コードを返す (起動失敗は -1)。stderr は err_out に回収する。
int run(const std::vector<std::string>& args, std::string* err_out = nullptr);

// 実行ファイルの場所を探す: PATH -> exe横の ffmpeg フォルダ -> exe横。
std::string find_tool(const std::string& name);
std::string exe_dir();

} // namespace runner
