#include "runner.hpp"

#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace runner {

#ifdef _WIN32
namespace {
std::wstring widen(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
std::string narrow(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
std::wstring quote_arg(const std::wstring& a) {
    if (a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring q = L"\"";
    for (wchar_t c : a) {
        if (c == L'"') q += L'\\';
        q += c;
    }
    q += L'"';
    return q;
}
} // namespace

int run(const std::vector<std::string>& args, std::string* err_out) {
    if (args.empty()) return -1;
    std::wstring cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i) cmd += L' ';
        cmd += quote_arg(widen(args[i]));
    }
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE err_rd = nullptr, err_wr = nullptr;
    HANDLE nul_in = nullptr;
    if (err_out) {
        CreatePipe(&err_rd, &err_wr, &sa, 0);
        // 読み側は子へ継承させない (継承されると EOF が来ず読み取りが止まる)
        SetHandleInformation(err_rd, HANDLE_FLAG_INHERIT, 0);
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    if (err_out) {
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdError = err_wr;
        si.hStdOutput = err_wr;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        if (si.hStdInput == nullptr || si.hStdInput == INVALID_HANDLE_VALUE) {
            // GUI サブシステムでは標準入力が無いので NUL を渡す
            nul_in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, 0, nullptr);
            si.hStdInput = nul_in;
        }
    }
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(L'\0');
    BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr,
                             err_out ? TRUE : FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    if (err_wr) CloseHandle(err_wr);
    if (nul_in) CloseHandle(nul_in);
    if (!ok) {
        if (err_rd) CloseHandle(err_rd);
        return -1;
    }
    std::string collected;
    if (err_rd) {
        char tmp[4096];
        DWORD got = 0;
        while (ReadFile(err_rd, tmp, sizeof(tmp), &got, nullptr) && got > 0)
            collected.append(tmp, got);
        CloseHandle(err_rd);
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (err_out) *err_out = collected;
    return (int)code;
}

std::string exe_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0) return ".";
    return narrow(fs::path(std::wstring(buf, n)).parent_path().wstring());
}

#else // POSIX

int run(const std::vector<std::string>& args, std::string* err_out) {
    if (args.empty()) return -1;
    int pipefd[2] = {-1, -1};
    if (err_out && pipe(pipefd) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        if (err_out) {
            dup2(pipefd[1], STDERR_FILENO);
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[0]);
            close(pipefd[1]);
        }
        std::vector<char*> argv;
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    std::string collected;
    if (err_out) {
        close(pipefd[1]);
        char tmp[4096];
        ssize_t got;
        while ((got = read(pipefd[0], tmp, sizeof(tmp))) > 0) collected.append(tmp, (size_t)got);
        close(pipefd[0]);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (err_out) *err_out = collected;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

std::string exe_dir() {
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (ec) return ".";
    return p.parent_path().string();
}
#endif

std::string find_tool(const std::string& name) {
    std::string exe_name = name;
#ifdef _WIN32
    if (fs::path(exe_name).extension().empty()) exe_name += ".exe";
#endif
    // PATH
    const char* path_env = std::getenv("PATH");
    if (path_env) {
        std::string cur;
        std::string all = path_env;
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        for (size_t i = 0; i <= all.size(); ++i) {
            if (i == all.size() || all[i] == sep) {
                if (!cur.empty()) {
                    fs::path cand = fs::path(cur) / exe_name;
                    std::error_code ec;
                    if (fs::is_regular_file(cand, ec)) return cand.string();
                }
                cur.clear();
            } else {
                cur += all[i];
            }
        }
    }
    // exe横の ffmpeg フォルダ / exe横 (Python 版と同じ流儀)
    fs::path base = exe_dir();
    for (const fs::path& cand : {base / "ffmpeg" / exe_name, base / exe_name}) {
        std::error_code ec;
        if (fs::is_regular_file(cand, ec)) return cand.string();
    }
    return "";
}

} // namespace runner
