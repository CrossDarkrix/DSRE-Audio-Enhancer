set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# WIN32: Windows では GUI サブシステムにし、コンソール窓を出さない
# (他プラットフォームでは無視される)。LiteDesktopStudio C++版と同じ指定。
# resources.qrc で NewIcon.png を埋め込み、ウィンドウアイコンに使う。
# Windows では assets/app.rc 経由で exe アイコンにも同じ絵を設定する。
add_executable(dsre_gui WIN32 main.cpp resources.qrc)
if(WIN32)
  target_sources(dsre_gui PRIVATE ${CMAKE_SOURCE_DIR}/assets/app.rc)
  set_source_files_properties(${CMAKE_SOURCE_DIR}/assets/app.rc
    PROPERTIES OBJECT_DEPENDS "${CMAKE_SOURCE_DIR}/assets/app.ico")
endif()
target_include_directories(dsre_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../src)
target_link_libraries(dsre_gui PRIVATE Qt6::Widgets dsre_io)
set_target_properties(dsre_gui PROPERTIES OUTPUT_NAME "DSREAudioEnhancer")
if(WIN32)
  # CLI と同じ bin フォルダに出力し、ランタイム同梱を一括で行う
  set_target_properties(dsre_gui PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  # ---- ランタイム同梱 (add_custom_command はこのディレクトリ内で行う) ----
  get_filename_component(_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  find_program(WINDEPLOYQT_EXE
    NAMES windeployqt6 windeployqt-qt6 windeployqt
    HINTS "${_mingw_bin}" "${CMAKE_PREFIX_PATH}/bin")
  if(WINDEPLOYQT_EXE)
    add_custom_command(TARGET dsre_gui POST_BUILD
      COMMAND "${WINDEPLOYQT_EXE}" --release --no-translations
              --no-opengl-sw --no-system-d3d-compiler
              "$<TARGET_FILE:dsre_gui>"
      COMMENT "Deploying Qt runtime with windeployqt")
  else()
    message(WARNING "windeployqt not found - falling back to manual Qt deployment")
    get_target_property(_qmake_exe Qt6::qmake IMPORTED_LOCATION)
    execute_process(COMMAND "${_qmake_exe}" -query QT_INSTALL_PLUGINS
      OUTPUT_VARIABLE _qt_plugins OUTPUT_STRIP_TRAILING_WHITESPACE)
    foreach(_qt_lib Core Gui Widgets)
      add_custom_command(TARGET dsre_gui POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:Qt6::${_qt_lib}>"
                "$<TARGET_FILE_DIR:dsre_gui>")
    endforeach()
    foreach(_plugdir platforms tls styles imageformats iconengines)
      if(EXISTS "${_qt_plugins}/${_plugdir}")
        add_custom_command(TARGET dsre_gui POST_BUILD
          COMMAND "${CMAKE_COMMAND}" -E copy_directory
                  "${_qt_plugins}/${_plugdir}"
                  "$<TARGET_FILE_DIR:dsre_gui>/${_plugdir}")
      endif()
    endforeach()
  endif()
  add_custom_command(TARGET dsre_gui POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
            "-DEXE=$<TARGET_FILE:dsre_gui>"
            "-DBIN_DIR=${_mingw_bin}"
            "-DDEST_DIR=$<TARGET_FILE_DIR:dsre_gui>"
            -P "${CMAKE_SOURCE_DIR}/cmake/DeployRuntime.cmake"
    COMMENT "Copying MinGW runtime DLL closure (GUI)")
  if(EXISTS "${CMAKE_SOURCE_DIR}/ffmpeg/ffmpeg.exe")
    add_custom_command(TARGET dsre_gui POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
              "${CMAKE_SOURCE_DIR}/ffmpeg"
              "$<TARGET_FILE_DIR:dsre_gui>/ffmpeg"
      COMMENT "Copying bundled ffmpeg")
  endif()
endif()
