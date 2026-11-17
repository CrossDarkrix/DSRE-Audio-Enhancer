# PackageDist.cmake - builds OUT_DIR/DSREAudioEnhancer-win64.zip from the
# deployed build output in SRC_DIR (exe + runtime DLLs + Qt plugin subfolders
# + ffmpeg subfolder when it was deployed).
# Required -D variables: SRC_DIR, OUT_DIR. Optional: PROJECT_DIR (for README).

cmake_minimum_required(VERSION 3.21)

set(_pkg "${OUT_DIR}/DSREAudioEnhancer")
file(REMOVE_RECURSE "${_pkg}")
file(MAKE_DIRECTORY "${_pkg}")

file(GLOB _files "${SRC_DIR}/*.exe" "${SRC_DIR}/*.dll")
foreach(_f IN LISTS _files)
  file(COPY "${_f}" DESTINATION "${_pkg}")
endforeach()
foreach(_sub IN ITEMS platforms tls styles imageformats iconengines ffmpeg)
  if(EXISTS "${SRC_DIR}/${_sub}")
    file(COPY "${SRC_DIR}/${_sub}" DESTINATION "${_pkg}")
  endif()
endforeach()
if(PROJECT_DIR AND EXISTS "${PROJECT_DIR}/README.md")
  file(COPY "${PROJECT_DIR}/README.md" DESTINATION "${_pkg}")
endif()

file(REMOVE "${OUT_DIR}/DSREAudioEnhancer-win64.zip")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar cf
          "${OUT_DIR}/DSREAudioEnhancer-win64.zip" --format=zip
          "DSREAudioEnhancer"
  WORKING_DIRECTORY "${OUT_DIR}"
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "PackageDist: failed to create zip (rc=${_rc})")
endif()
message(STATUS "PackageDist: wrote ${OUT_DIR}/DSREAudioEnhancer-win64.zip")
