// DSRE Audio Enhancer — Qt6 GUI (C++版)
// Python 版 GUI と同じ設定ファイル (dsre_enhanced_config.json)・同じパラメータ・同じ処理手順。
// 処理はワーカースレッドで実行し、進捗シグナルで表示を更新、中止は atomic フラグで即時に伝える。
#include <QApplication>
#include <QIcon>
#include <QComboBox>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTextEdit>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>

#include <atomic>
#include <memory>

#include "dsre_pipeline.hpp"
#include "strings.hpp"
#include "style.hpp"

namespace {

bool is_audio_file(const QString& path) {
    QString ext = QFileInfo(path).suffix().toLower();
    return QStringList{"wav", "flac", "mp3", "m4a", "aac", "ogg", "opus", "aif", "aiff", "wma"}
        .contains(ext);
}

} // namespace

class DropListWidget : public QListWidget {
    Q_OBJECT
public:
    explicit DropListWidget(QWidget* parent = nullptr) : QListWidget(parent) {
        setAcceptDrops(true);
        setSelectionMode(QAbstractItemView::ExtendedSelection);
        setDragDropMode(QAbstractItemView::DropOnly);
    }
signals:
    void files_dropped(const QStringList& paths);
protected:
    void dragEnterEvent(QDragEnterEvent* e) override {
        if (e->mimeData()->hasUrls()) e->acceptProposedAction();
    }
    void dragMoveEvent(QDragMoveEvent* e) override {
        if (e->mimeData()->hasUrls()) e->acceptProposedAction();
    }
    void dropEvent(QDropEvent* e) override {
        QStringList paths;
        for (const QUrl& u : e->mimeData()->urls())
            if (u.isLocalFile()) paths << u.toLocalFile();
        if (!paths.isEmpty()) emit files_dropped(paths);
        e->acceptProposedAction();
    }
};

class Worker : public QThread {
    Q_OBJECT
public:
    QStringList files;
    pipeline::FileParams params;
    std::shared_ptr<std::atomic<bool>> abort_flag = std::make_shared<std::atomic<bool>>(false);

    void request_abort() { abort_flag->store(true); }

signals:
    void log_line(const QString& msg);
    void file_progress(int pct, const QString& stage);
    void overall_progress(int done, int total);
    void file_done(const QString& in_path, const QString& out_path);
    void file_failed(const QString& in_path, const QString& error);
    void all_finished(bool aborted);

protected:
    void run() override {
        int done = 0;
        const int total = (int)files.size();
        emit overall_progress(0, total);
        bool was_aborted = false;
        for (const QString& f : files) {
            if (abort_flag->load()) { was_aborted = true; break; }
            pipeline::FileCallbacks cb;
            cb.abort = abort_flag.get();
            cb.progress = [this](int pct, const std::string& stage) {
                emit file_progress(pct, QString::fromStdString(stage));
            };
            cb.log = [this](const std::string& m) { emit log_line(QString::fromStdString(m)); };
            std::string out_path, error;
            bool ok = pipeline::process_file(f.toStdString(), params, cb, out_path, error);
            if (ok) {
                emit file_done(f, QString::fromStdString(out_path));
            } else if (abort_flag->load()) {
                was_aborted = true;
                break;
            } else {
                emit file_failed(f, QString::fromStdString(error));
            }
            ++done;
            emit overall_progress(done, total);
        }
        emit all_finished(was_aborted);
    }
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow() {
        load_config_values();
        S = (language == "ja") ? strings_ja() : strings_en();
        build_ui();
        apply_theme();
        retranslate();
        restore_config_to_widgets();
        resize(1100, 720);
    }

    ~MainWindow() override {
        if (worker) {
            worker->request_abort();
            worker->wait(3000);
        }
        save_config();
    }

private:
    // ---- config (Python 版と同一ファイル名・同一キー) ----
    // 置き場所は %USERPROFILE%\DSREAudioEnhancer\ (Program Files 等に
    // インストールしても管理者権限なしで読み書きできるように)。
    // 旧場所 (exe 横) に設定が残っていれば初回に自動で移行する。
    static QString user_data_dir() {
        const QString dir = QDir::homePath() + "/DSREAudioEnhancer";
        QDir().mkpath(dir);
        return dir;
    }
    QString config_path() const {
        return user_data_dir() + "/dsre_enhanced_config.json";
    }
    void load_config_values() {
        const QString path = config_path();
        if (!QFile::exists(path)) {
            const QString old =
                QCoreApplication::applicationDirPath() + "/dsre_enhanced_config.json";
            if (QFile::exists(old)) QFile::copy(old, path); // 旧設定の移行
        }
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly)) return;
        QJsonObject o = QJsonDocument::fromJson(f.readAll()).object();
        cfg_m = o.value("m").toInt(16);
        cfg_decay = o.value("decay").toDouble(0.7);
        cfg_sr = o.value("target_sr").toInt(96000);
        cfg_outdir = o.value("output_dir").toString(QDir::currentPath() + "/enhanced_output");
        cfg_format = o.value("format").toString("ALAC");
        cfg_dark = o.value("dark_mode").toBool(true);
        language = o.value("language").toString(
            QLocale::system().name().startsWith("ja") ? "ja" : "en");
        for (const QJsonValue& v : o.value("recent_files").toArray())
            recent_files << v.toString();
    }
    void save_config() {
        QJsonObject o;
        o["m"] = sb_m->value();
        o["decay"] = dsb_strength->value();
        o["target_sr"] = sb_sr->value();
        o["output_dir"] = le_outdir->text();
        o["format"] = cb_format->currentText();
        o["dark_mode"] = cfg_dark;
        o["language"] = language;
        QJsonArray arr;
        for (const QString& r : recent_files) arr.append(r);
        o["recent_files"] = arr;
        QFile f(config_path());
        if (f.open(QIODevice::WriteOnly))
            f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
    }
    void restore_config_to_widgets() {
        sb_m->setValue(cfg_m);
        dsb_strength->setValue(cfg_decay);
        sb_sr->setValue(cfg_sr);
        le_outdir->setText(cfg_outdir);
        int idx = cb_format->findText(cfg_format);
        if (idx >= 0) cb_format->setCurrentIndex(idx);
        rebuild_recent_menu();
    }

    void build_ui() {
        auto* central = new QWidget(this);
        setCentralWidget(central);
        auto* root = new QVBoxLayout(central);
        root->setContentsMargins(12, 12, 12, 12);
        root->setSpacing(10);

        // メニュー
        menu_file = menuBar()->addMenu(QString());
        menu_recent = menu_file->addMenu(QString());
        menu_language = menuBar()->addMenu(QString());
        act_lang_en = menu_language->addAction("English");
        act_lang_ja = menu_language->addAction(QString::fromUtf8("日本語"));
        connect(act_lang_en, &QAction::triggered, this, [this] { set_language("en"); });
        connect(act_lang_ja, &QAction::triggered, this, [this] { set_language("ja"); });
        act_theme = menuBar()->addAction(QString());
        act_theme->setCheckable(true);
        act_theme->setChecked(cfg_dark);
        connect(act_theme, &QAction::triggered, this, [this](bool on) {
            cfg_dark = on;
            apply_theme();
            save_config();
        });
        act_about = menuBar()->addAction(QString());
        connect(act_about, &QAction::triggered, this, [this] {
            QMessageBox::about(this, S.window_title, S.about_text);
        });

        auto* splitter = new QSplitter(Qt::Horizontal, central);
        root->addWidget(splitter, 1);

        // 左: ファイル一覧
        auto* left = new QFrame(splitter);
        left->setObjectName("panel");
        auto* lv = new QVBoxLayout(left);
        lbl_files = new QLabel(left);
        lbl_files->setObjectName("panelTitle");
        lv->addWidget(lbl_files);
        list = new DropListWidget(left);
        lv->addWidget(list, 1);
        auto* file_btns = new QHBoxLayout();
        btn_add = new QPushButton(left);
        btn_folder = new QPushButton(left);
        btn_remove = new QPushButton(left);
        btn_clear = new QPushButton(left);
        file_btns->addWidget(btn_add);
        file_btns->addWidget(btn_folder);
        file_btns->addWidget(btn_remove);
        file_btns->addWidget(btn_clear);
        lv->addLayout(file_btns);
        splitter->addWidget(left);

        // 右: コントロール + ログ
        auto* right = new QWidget(splitter);
        auto* rv = new QVBoxLayout(right);
        auto* panel = new QFrame(right);
        panel->setObjectName("panel");
        auto* pv = new QVBoxLayout(panel);
        lbl_control = new QLabel(panel);
        lbl_control->setObjectName("panelTitle");
        pv->addWidget(lbl_control);

        auto* row_out = new QHBoxLayout();
        le_outdir = new QLineEdit(panel);
        btn_outdir = new QPushButton(panel);
        row_out->addWidget(le_outdir, 1);
        row_out->addWidget(btn_outdir);
        pv->addLayout(row_out);

        sb_m = new QSpinBox(panel);
        sb_m->setRange(1, 32);
        dsb_strength = new QDoubleSpinBox(panel);
        dsb_strength->setRange(0.1, 1.0);
        dsb_strength->setSingleStep(0.05);
        sb_sr = new QSpinBox(panel);
        sb_sr->setRange(8000, 384000);
        sb_sr->setSingleStep(100);
        cb_format = new QComboBox(panel);
        cb_format->addItems({"ALAC", "FLAC", "MP3"});
        lbl_m = new QLabel(panel);
        lbl_strength = new QLabel(panel);
        lbl_sr = new QLabel(panel);
        lbl_format = new QLabel(panel);
        auto add_row = [&](QLabel* l, QWidget* w) {
            auto* r = new QHBoxLayout();
            r->addWidget(l, 1);
            r->addWidget(w, 1);
            pv->addLayout(r);
        };
        add_row(lbl_m, sb_m);
        add_row(lbl_strength, dsb_strength);
        add_row(lbl_sr, sb_sr);
        add_row(lbl_format, cb_format);

        pb_file = new QProgressBar(panel);
        pb_file->setRange(0, 100);
        pb_all = new QProgressBar(panel);
        pb_all->setRange(0, 100);
        pv->addWidget(pb_file);
        pv->addWidget(pb_all);
        lbl_stats = new QLabel(panel);
        pv->addWidget(lbl_stats);

        auto* run_btns = new QHBoxLayout();
        btn_start = new QPushButton(panel);
        btn_start->setObjectName("primary");
        btn_cancel = new QPushButton(panel);
        btn_cancel->setEnabled(false);
        btn_retry = new QPushButton(panel);
        btn_retry->setEnabled(false);
        run_btns->addWidget(btn_start, 2);
        run_btns->addWidget(btn_cancel, 1);
        run_btns->addWidget(btn_retry, 1);
        pv->addLayout(run_btns);
        rv->addWidget(panel);

        lbl_log = new QLabel(right);
        lbl_log->setObjectName("panelTitle");
        rv->addWidget(lbl_log);
        te_log = new QTextEdit(right);
        te_log->setReadOnly(true);
        rv->addWidget(te_log, 1);
        splitter->addWidget(right);
        splitter->setStretchFactor(0, 3);
        splitter->setStretchFactor(1, 2);

        statusBar()->showMessage(QString());

        connect(btn_add, &QPushButton::clicked, this, &MainWindow::on_add_files);
        connect(btn_folder, &QPushButton::clicked, this, &MainWindow::on_add_folder);
        connect(btn_remove, &QPushButton::clicked, this, &MainWindow::on_remove_selected);
        connect(btn_clear, &QPushButton::clicked, this, &MainWindow::on_clear);
        connect(btn_outdir, &QPushButton::clicked, this, &MainWindow::on_choose_outdir);
        connect(btn_start, &QPushButton::clicked, this, &MainWindow::on_start);
        connect(btn_cancel, &QPushButton::clicked, this, &MainWindow::on_cancel);
        connect(btn_retry, &QPushButton::clicked, this, &MainWindow::on_retry);
        connect(list, &DropListWidget::files_dropped, this, &MainWindow::add_paths);
        for (auto* w : {sb_m}) connect(w, &QSpinBox::valueChanged, this, [this] { save_config(); });
        connect(dsb_strength, &QDoubleSpinBox::valueChanged, this, [this] { save_config(); });
        connect(sb_sr, &QSpinBox::valueChanged, this, [this] { save_config(); });
        connect(cb_format, &QComboBox::currentTextChanged, this, [this] { save_config(); });
    }

    void apply_theme() { qApp->setStyleSheet(cfg_dark ? dark_qss() : light_qss()); }

    void retranslate() {
        setWindowTitle(S.window_title);
        menu_file->setTitle(S.menu_file);
        menu_recent->setTitle(S.menu_recent);
        menu_language->setTitle(S.menu_language);
        act_theme->setText(S.menu_theme_dark);
        act_about->setText(S.menu_about);
        lbl_files->setText(S.files_title.toUpper());
        lbl_control->setText(S.control.toUpper());
        lbl_log->setText(S.log_title.toUpper());
        btn_add->setText(S.add_files);
        btn_folder->setText(S.add_folder);
        btn_remove->setText(S.remove_selected);
        btn_clear->setText(S.clear_list);
        btn_outdir->setText(S.select_output);
        le_outdir->setPlaceholderText(S.output_folder);
        lbl_m->setText(S.harmonic);
        sb_m->setToolTip(S.harmonic_tip);
        lbl_strength->setText(S.strength);
        dsb_strength->setToolTip(S.strength_tip);
        lbl_sr->setText(S.target_sr);
        sb_sr->setToolTip(S.target_sr_tip);
        lbl_format->setText(S.format);
        btn_start->setText(S.start);
        btn_cancel->setText(S.cancel);
        btn_retry->setText(S.retry);
        lbl_stats->setText(S.ready);
        statusBar()->showMessage(S.ready);
        if (list->count() == 0) show_hint_item();
    }

    void set_language(const QString& lang) {
        language = lang;
        S = (lang == "ja") ? strings_ja() : strings_en();
        retranslate();
        save_config();
    }

    void show_hint_item() {
        // 空リストのときのヒント表示 (先頭に置き、ファイル追加時に消す)
        if (list->count() == 0) {
            auto* it = new QListWidgetItem(S.dropped_hint, list);
            it->setFlags(Qt::NoItemFlags);
            hint_item = it;
        }
    }
    void clear_hint() {
        if (hint_item) {
            delete hint_item;
            hint_item = nullptr;
        }
    }

    void rebuild_recent_menu() {
        menu_recent->clear();
        for (const QString& r : recent_files) {
            QAction* a = menu_recent->addAction(QFileInfo(r).fileName());
            a->setToolTip(r);
            connect(a, &QAction::triggered, this, [this, r] { add_paths({r}); });
        }
    }
    void add_recent(const QString& path) {
        recent_files.removeAll(path);
        recent_files.prepend(path);
        while (recent_files.size() > 10) recent_files.removeLast();
        rebuild_recent_menu();
    }

    void add_paths(const QStringList& paths) {
        clear_hint();
        for (const QString& p : paths) {
            QFileInfo fi(p);
            if (fi.isDir()) {
                QDirIterator it(p, QDir::Files, QDirIterator::Subdirectories);
                while (it.hasNext()) {
                    QString f = it.next();
                    if (is_audio_file(f)) add_one(f);
                }
            } else if (fi.isFile() && is_audio_file(p)) {
                add_one(p);
            }
        }
        update_stats_label();
    }
    void add_one(const QString& path) {
        for (int i = 0; i < list->count(); ++i)
            if (list->item(i)->data(Qt::UserRole).toString() == path) return;
        auto* it = new QListWidgetItem(QFileInfo(path).fileName(), list);
        it->setData(Qt::UserRole, path);
        it->setToolTip(path);
    }

    QStringList current_files() const {
        QStringList out;
        for (int i = 0; i < list->count(); ++i) {
            QString p = list->item(i)->data(Qt::UserRole).toString();
            if (!p.isEmpty()) out << p;
        }
        return out;
    }

    void update_stats_label() {
        QStringList files = current_files();
        double mb = 0;
        for (const QString& f : files) mb += QFileInfo(f).size() / 1048576.0;
        lbl_stats->setText(S.stats_fmt.arg(files.size()).arg(mb, 0, 'f', 1).arg(0));
    }

    void log(const QString& msg) {
        te_log->append(msg);
        // クラッシュ調査用: 設定と同じユーザーフォルダの dsre_log.txt にも
        // 逐次書き出す (flush する)
        static QFile lf(user_data_dir() + "/dsre_log.txt");
        if (!lf.isOpen()) lf.open(QIODevice::Append | QIODevice::Text);
        if (lf.isOpen()) {
            lf.write((msg + "\n").toUtf8());
            lf.flush();
        }
    }

    // ---- actions ----
    void on_add_files() {
        QStringList files = QFileDialog::getOpenFileNames(
            this, S.add_files, QString(),
            "Audio (*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.opus *.aif *.aiff *.wma)");
        add_paths(files);
    }
    void on_add_folder() {
        QString dir = QFileDialog::getExistingDirectory(this, S.add_folder);
        if (!dir.isEmpty()) add_paths({dir});
    }
    void on_remove_selected() {
        auto sel = list->selectedItems();
        for (auto* it : sel) delete it;
        if (list->count() == 0) show_hint_item();
        update_stats_label();
    }
    void on_clear() {
        list->clear();
        hint_item = nullptr;
        show_hint_item();
        failed_files.clear();
        btn_retry->setEnabled(false);
        update_stats_label();
    }
    void on_choose_outdir() {
        QString dir = QFileDialog::getExistingDirectory(this, S.select_output, le_outdir->text());
        if (!dir.isEmpty()) {
            le_outdir->setText(dir);
            save_config();
        }
    }

    void set_running(bool running) {
        btn_start->setEnabled(!running);
        btn_cancel->setEnabled(running);
        btn_add->setEnabled(!running);
        btn_folder->setEnabled(!running);
        btn_remove->setEnabled(!running);
        btn_clear->setEnabled(!running);
    }

    void start_with(const QStringList& files) {
        if (files.isEmpty() || worker) return;
        failed_files.clear();
        btn_retry->setEnabled(false);
        worker = new Worker();
        worker->files = files;
        worker->params.m = sb_m->value();
        worker->params.strength = dsb_strength->value();
        worker->params.target_sr = sb_sr->value();
        worker->params.format = cb_format->currentText().toStdString();
        worker->params.out_dir = le_outdir->text().toStdString();
        connect(worker, &Worker::log_line, this, &MainWindow::log);
        connect(worker, &Worker::file_progress, this, [this](int pct, const QString& stage) {
            pb_file->setValue(pct);
            statusBar()->showMessage(stage);
        });
        connect(worker, &Worker::overall_progress, this, [this](int done, int total) {
            pb_all->setValue(total > 0 ? done * 100 / total : 0);
        });
        connect(worker, &Worker::file_done, this, [this](const QString& in, const QString& out) {
            log(S.done_file.arg(out));
            add_recent(in);
            save_config();
        });
        connect(worker, &Worker::file_failed, this, [this](const QString& in, const QString& err) {
            log(S.failed_file.arg(QFileInfo(in).fileName()) + " " + err.left(200));
            failed_files << in;
        });
        connect(worker, &Worker::all_finished, this, [this](bool aborted) {
            set_running(false);
            btn_retry->setEnabled(!failed_files.isEmpty());
            log(aborted ? S.aborted : (failed_files.isEmpty() ? S.finished_all : S.finished_some));
            statusBar()->showMessage(aborted ? S.aborted : S.ready);
            worker->deleteLater();
            worker = nullptr;
            update_stats_label();
        });
        set_running(true);
        pb_file->setValue(0);
        pb_all->setValue(0);
        worker->setStackSize(256 * 1024 * 1024); // Windows の小さい既定スタックへの保険
        worker->start();
    }

    void on_start() { start_with(current_files()); }
    void on_retry() { start_with(failed_files); }
    void on_cancel() {
        if (worker) worker->request_abort();
    }

    // ---- members ----
    UiStrings S;
    QString language = "en";
    int cfg_m = 16;
    double cfg_decay = 0.7;
    int cfg_sr = 96000;
    QString cfg_outdir;
    QString cfg_format = "ALAC";
    bool cfg_dark = true;
    QStringList recent_files;
    QStringList failed_files;
    Worker* worker = nullptr;
    QListWidgetItem* hint_item = nullptr;

    QMenu* menu_file = nullptr;
    QMenu* menu_recent = nullptr;
    QMenu* menu_language = nullptr;
    QAction* act_lang_en = nullptr;
    QAction* act_lang_ja = nullptr;
    QAction* act_theme = nullptr;
    QAction* act_about = nullptr;
    DropListWidget* list = nullptr;
    QLabel* lbl_files = nullptr;
    QLabel* lbl_control = nullptr;
    QLabel* lbl_log = nullptr;
    QLabel* lbl_stats = nullptr;
    QLabel* lbl_m = nullptr;
    QLabel* lbl_strength = nullptr;
    QLabel* lbl_sr = nullptr;
    QLabel* lbl_format = nullptr;
    QPushButton* btn_add = nullptr;
    QPushButton* btn_folder = nullptr;
    QPushButton* btn_remove = nullptr;
    QPushButton* btn_clear = nullptr;
    QPushButton* btn_outdir = nullptr;
    QPushButton* btn_start = nullptr;
    QPushButton* btn_cancel = nullptr;
    QPushButton* btn_retry = nullptr;
    QLineEdit* le_outdir = nullptr;
    QSpinBox* sb_m = nullptr;
    QSpinBox* sb_sr = nullptr;
    QDoubleSpinBox* dsb_strength = nullptr;
    QComboBox* cb_format = nullptr;
    QProgressBar* pb_file = nullptr;
    QProgressBar* pb_all = nullptr;
    QTextEdit* te_log = nullptr;
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/NewIcon.png")); // 平佐作の新アイコン (resources.qrc)
    MainWindow w;
    w.show();
    return app.exec();
}

#include "main.moc"
