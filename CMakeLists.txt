cmake_minimum_required(VERSION 3.16)
project(dsre_cpp C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# FFT は pocketfft (ヘッダオンリー、numpy と同じエンジン) を double 精度で使う
add_library(dsre_core STATIC src/dsre_core.cpp)
target_include_directories(dsre_core PUBLIC src third_party)
find_package(Threads REQUIRED) # コアの並列化 (std::thread)
target_link_libraries(dsre_core PUBLIC Threads::Threads)
if(UNIX)
  target_link_libraries(dsre_core PUBLIC m)
endif()

add_library(dsre_io STATIC src/wav_io.cpp src/runner.cpp src/dsre_pipeline.cpp)
target_include_directories(dsre_io PUBLIC src)
target_link_libraries(dsre_io PUBLIC dsre_core)

add_executable(dsre_bench src/main.cpp)
target_link_libraries(dsre_bench PRIVATE dsre_core)

add_executable(dsre_cli src/dsre_cli.cpp)
target_link_libraries(dsre_cli PRIVATE dsre_io)
if(WIN32)
  target_sources(dsre_cli PRIVATE assets/app.rc) # GUI と同じ exe アイコン
  set_source_files_properties(assets/app.rc
    PROPERTIES OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assets/app.ico")
endif()
if(WIN32)
  set_target_properties(dsre_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

option(DSRE_BUILD_GUI "Build the Qt6 GUI" ON)
if(DSRE_BUILD_GUI)
  find_package(Qt6 COMPONENTS Widgets QUIET)
  if(Qt6_FOUND)
    add_subdirectory(gui)
  else()
    message(STATUS "Qt6 Widgets not found - GUI skipped")
  endif()
endif()

# ---- Windows: ランタイム同梱 (素の Windows でもそのまま実行できるように) ----
# LiteDesktopStudio C++版 v0.3 で実績のある方式:
#  1) GUI は windeployqt で Qt DLL + プラグインを exe の隣へ
#     (見つからなければ Qt DLL と主要プラグインを手動コピーにフォールバック)
#  2) DeployRuntime.cmake が objdump で PE import を再帰走査し、MinGW
#     ランタイムと Qt DLL 自身の依存 DLL の閉包をコピー (システム DLL は
#     コンパイラの bin に無いので自動的に除外される)
#  3) プロジェクト直下に ffmpeg/ フォルダ (ffmpeg.exe / ffprobe.exe) が
#     あればそれも隣へコピーし、dist zip にも含める
# `dist` ターゲットで build/dist/DSREAudioEnhancer-win64.zip を作る。
if(WIN32)
  get_filename_component(_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)

  add_custom_command(TARGET dsre_cli POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
            "-DEXE=$<TARGET_FILE:dsre_cli>"
            "-DBIN_DIR=${_mingw_bin}"
            "-DDEST_DIR=$<TARGET_FILE_DIR:dsre_cli>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeployRuntime.cmake"
    COMMENT "Copying MinGW runtime DLL closure (CLI)")

  # GUI 側の同梱コマンドは gui/CMakeLists.txt 側に置く。
  # add_custom_command(TARGET ...) はターゲットを作ったディレクトリでしか
  # 呼べないため、トップレベルから dsre_gui を触ると configure エラーになる。
  set(_dist_deps dsre_cli)
  if(TARGET dsre_gui)
    list(APPEND _dist_deps dsre_gui)
  endif()

  add_custom_target(dist
    COMMAND "${CMAKE_COMMAND}"
            "-DSRC_DIR=${CMAKE_BINARY_DIR}/bin"
            "-DOUT_DIR=${CMAKE_BINARY_DIR}/dist"
            "-DPROJECT_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageDist.cmake"
    DEPENDS ${_dist_deps}
    COMMENT "Packaging DSREAudioEnhancer-win64.zip")
endif()
