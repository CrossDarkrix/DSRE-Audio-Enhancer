#pragma once
// GUI の日英文字列テーブル。全ての表示文字列はここに集約し、
// 言語切替は retranslate() で即時反映する (.qm 不要で全環境同一に動くため)。
#include <QString>

struct UiStrings {
    QString window_title;
    QString add_files, add_folder, remove_selected, clear_list;
    QString output_folder, select_output;
    QString harmonic, harmonic_tip;
    QString strength, strength_tip;
    QString target_sr, target_sr_tip;
    QString format;
    QString start, cancel, retry;
    QString control, log_title, files_title;
    QString ready, processing, done_file, failed_file, finished_all, finished_some;
    QString aborted, dropped_hint;
    QString menu_file, menu_recent, menu_language, menu_theme_dark, menu_about;
    QString about_text;
    QString stats_fmt; // "%1 files / %2 MB / %3s"
};

inline UiStrings strings_en() {
    UiStrings s;
    s.window_title = "DSRE Audio Enhancer";
    s.add_files = "Add Input Files";
    s.add_folder = "Add Folder";
    s.remove_selected = "Remove Selected";
    s.clear_list = "Clear Input List";
    s.output_folder = "Output folder";
    s.select_output = "Select Output Directory";
    s.harmonic = "Harmonic Intensity (1-32):";
    s.harmonic_tip = "Harmonic intensity (1-32): Higher values add more harmonic richness";
    s.strength = "Enhancement Strength (0.1-1.0):";
    s.strength_tip = "Enhancement strength (0.1-1.0): Controls overall enhancement intensity";
    s.target_sr = "Target Sample Rate (Hz):";
    s.target_sr_tip = "Target sample rate: Uses FFmpeg resampling during loading";
    s.format = "Output Format:";
    s.start = "Start Enhanced Processing";
    s.cancel = "Cancel Processing";
    s.retry = "Retry Failed Files";
    s.control = "Control";
    s.log_title = "Log";
    s.files_title = "Input Audio Files";
    s.ready = "Ready";
    s.processing = "Processing %1 (%2/%3)";
    s.done_file = "Saved: %1";
    s.failed_file = "Error: %1";
    s.finished_all = "All files have been enhanced successfully!";
    s.finished_some = "Processing finished with some failed files. Use 'Retry Failed Files' if needed.";
    s.aborted = "Processing aborted";
    s.dropped_hint = "Drag and drop audio files here...";
    s.menu_file = "File";
    s.menu_recent = "Recent Files";
    s.menu_language = "Language";
    s.menu_theme_dark = "Dark Mode";
    s.menu_about = "About";
    s.about_text = "DSRE Audio Enhancer (C++ edition)\nDeep Sound Resolution Enhancer — fast, light, and faithful to the original sound.";
    s.stats_fmt = "%1 files / %2 MB / %3 s";
    return s;
}

inline UiStrings strings_ja() {
    UiStrings s = strings_en();
    s.add_files = "入力ファイルを追加";
    s.add_folder = "フォルダを追加";
    s.remove_selected = "選択項目を削除";
    s.clear_list = "リストをクリア";
    s.output_folder = "出力フォルダ";
    s.select_output = "出力フォルダを選択";
    s.harmonic = "倍音の強さ (1-32):";
    s.harmonic_tip = "倍音の強さ (1-32): 大きいほど倍音が豊かになります";
    s.strength = "強調の強さ (0.1-1.0):";
    s.strength_tip = "強調の強さ (0.1-1.0): 全体の強調量を調整します";
    s.target_sr = "目標サンプルレート (Hz):";
    s.target_sr_tip = "目標サンプルレート: 読み込み時に FFmpeg でリサンプリングします";
    s.format = "出力フォーマット:";
    s.start = "強調処理を開始";
    s.cancel = "処理を中止";
    s.retry = "失敗したファイルを再試行";
    s.control = "コントロール";
    s.log_title = "ログ";
    s.files_title = "入力音声ファイル";
    s.ready = "準備完了";
    s.processing = "処理中 %1 (%2/%3)";
    s.done_file = "保存しました: %1";
    s.failed_file = "エラー: %1";
    s.finished_all = "全ファイルの強調が完了しました！";
    s.finished_some = "一部のファイルが失敗しました。「失敗したファイルを再試行」が使えます。";
    s.aborted = "処理を中止しました";
    s.dropped_hint = "ここに音声ファイルをドラッグ＆ドロップ...";
    s.menu_file = "ファイル";
    s.menu_recent = "最近使ったファイル";
    s.menu_language = "言語 / Language";
    s.menu_theme_dark = "ダークモード";
    s.about_text = "DSRE Audio Enhancer (C++版)\nDeep Sound Resolution Enhancer — 高速・軽量で、本来の音に忠実に。";
    s.stats_fmt = "%1 ファイル / %2 MB / %3 秒";
    return s;
}
