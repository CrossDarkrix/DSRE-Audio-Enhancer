#pragma once
// 今風のスタイリッシュテーマ (QSS のみ、追加依存なし)。
#include <QString>

inline QString dark_qss() {
    return QStringLiteral(R"QSS(
QMainWindow, QWidget { background: #0e1320; color: #e8edf7; font-size: 13px; }
QFrame#panel { background: #161d2e; border: 1px solid #232c44; border-radius: 12px; }
QLabel#panelTitle { color: #8fb8ff; font-size: 12px; font-weight: 600; letter-spacing: 1px; }
QListWidget { background: #101828; border: 1px solid #232c44; border-radius: 8px; padding: 4px; }
QListWidget::item { padding: 5px 6px; border-radius: 6px; }
QListWidget::item:selected { background: #274268; color: #ffffff; }
QPushButton { background: #1d2740; border: 1px solid #2d3a5c; border-radius: 8px; padding: 7px 12px; }
QPushButton:hover { background: #263354; }
QPushButton:pressed { background: #2f4066; }
QPushButton:disabled { color: #5b6a8a; background: #141b2c; }
QPushButton#primary { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f7fe0, stop:1 #37b6a2); color: white; font-weight: 600; border: none; }
QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3c8bf2, stop:1 #43c7b2); }
QPushButton#primary:disabled { background: #233350; color: #7d8dab; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit { background: #101828; border: 1px solid #2d3a5c; border-radius: 7px; padding: 5px 8px; }
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus { border-color: #2f7fe0; }
QProgressBar { background: #101828; border: 1px solid #232c44; border-radius: 7px; height: 14px; text-align: center; color: #aebbd6; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f7fe0, stop:1 #37b6a2); border-radius: 6px; }
QTextEdit { background: #0b101c; border: 1px solid #232c44; border-radius: 8px; font-family: monospace; font-size: 12px; }
QMenuBar { background: #0e1320; }
QMenuBar::item:selected, QMenu::item:selected { background: #274268; }
QStatusBar { color: #8b9bbd; }
QSplitter::handle { background: transparent; width: 6px; }
)QSS");
}

inline QString light_qss() {
    return QStringLiteral(R"QSS(
QMainWindow, QWidget { background: #f2f5fa; color: #1b2436; font-size: 13px; }
QFrame#panel { background: #ffffff; border: 1px solid #d8e0ee; border-radius: 12px; }
QLabel#panelTitle { color: #2f6fd0; font-size: 12px; font-weight: 600; letter-spacing: 1px; }
QListWidget { background: #fbfcfe; border: 1px solid #d8e0ee; border-radius: 8px; padding: 4px; }
QListWidget::item { padding: 5px 6px; border-radius: 6px; }
QListWidget::item:selected { background: #d7e7ff; }
QPushButton { background: #e8eefb; border: 1px solid #c3d2ec; border-radius: 8px; padding: 7px 12px; }
QPushButton:hover { background: #d9e6fb; }
QPushButton:disabled { color: #93a3bd; }
QPushButton#primary { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f7fe0, stop:1 #37b6a2); color: white; font-weight: 600; border: none; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit { background: #ffffff; border: 1px solid #c3d2ec; border-radius: 7px; padding: 5px 8px; }
QProgressBar { background: #ffffff; border: 1px solid #d8e0ee; border-radius: 7px; height: 14px; text-align: center; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f7fe0, stop:1 #37b6a2); border-radius: 6px; }
QTextEdit { background: #fbfcfe; border: 1px solid #d8e0ee; border-radius: 8px; font-family: monospace; font-size: 12px; }
)QSS");
}
