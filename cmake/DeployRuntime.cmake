# DeployRuntime.cmake - run at POST_BUILD on Windows.
# Copies the transitive closure of runtime DLLs needed by EXE (and by the
# DLLs already deployed next to it, e.g. Qt's own dependencies in MSYS2
# builds: freetype / harfbuzz / ICU / brotli ...) from BIN_DIR into DEST_DIR.
#
# Method: scan PE imports with `objdump -p` (binutils ships with the MinGW
# toolchain). A DLL is copied only when it exists in BIN_DIR; Windows system
# DLLs (KERNEL32.dll etc.) are not there, so they are skipped automatically.
#
# Required -D variables: EXE, BIN_DIR, DEST_DIR.

cmake_minimum_required(VERSION 3.21) # sets CMP0057 NEW (IN_LIST) in script mode

find_program(OBJDUMP_EXE NAMES objdump HINTS "${BIN_DIR}")
if(NOT OBJDUMP_EXE)
  message(WARNING "DeployRuntime: objdump not found in ${BIN_DIR}; "
                  "MinGW runtime DLLs were NOT copied. "
                  "Copy libgcc_s_seh-1.dll, libstdc++-6.dll and "
                  "libwinpthread-1.dll next to the exe by hand.")
else()
  file(GLOB_RECURSE _queue "${DEST_DIR}/*.exe" "${DEST_DIR}/*.dll")
  set(_seen "")
  list(LENGTH _queue _todo)
  while(_todo GREATER 0)
    list(POP_FRONT _queue _file)
    if(NOT _file IN_LIST _seen)
      list(APPEND _seen "${_file}")
      execute_process(COMMAND "${OBJDUMP_EXE}" -p "${_file}"
        OUTPUT_VARIABLE _out ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REGEX MATCHALL "DLL Name: [^\n]+" _lines "${_out}")
      foreach(_line IN LISTS _lines)
        string(REGEX REPLACE ".*DLL Name:[ ]*" "" _dll "${_line}")
        string(REPLACE "\r" "" _dll "${_dll}")
        string(STRIP "${_dll}" _dll)
        if(_dll AND EXISTS "${BIN_DIR}/${_dll}" AND NOT EXISTS "${DEST_DIR}/${_dll}")
          file(COPY "${BIN_DIR}/${_dll}" DESTINATION "${DEST_DIR}")
          list(APPEND _queue "${DEST_DIR}/${_dll}")
          message(STATUS "DeployRuntime: copied ${_dll}")
        endif()
      endforeach()
    endif()
    list(LENGTH _queue _todo)
  endwhile()
endif()
