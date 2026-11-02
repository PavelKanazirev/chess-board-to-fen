cmake_minimum_required(VERSION 3.20)
project(board2fen_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime: either point ONNXRUNTIME_ROOT at an extracted package (e.g. one for an ARM board),
# or leave it empty and the official prebuilt package for this computer is downloaded once.
set(ORT_VERSION "1.31.1" CACHE STRING "ONNX Runtime release to download")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Extracted ONNX Runtime package (contains include/ and lib/)")

if(NOT ONNXRUNTIME_ROOT)
  if(WIN32)
    set(ort_package "onnxruntime-win-x64-${ORT_VERSION}")
    set(ort_extension "zip")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ort_package "onnxruntime-linux-aarch64-${ORT_VERSION}")
    set(ort_extension "tgz")
  else()
    set(ort_package "onnxruntime-linux-x64-${ORT_VERSION}")
    set(ort_extension "tgz")
  endif()

  set(ONNXRUNTIME_ROOT "${CMAKE_BINARY_DIR}/${ort_package}")
  if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
    set(archive "${CMAKE_BINARY_DIR}/${ort_package}.${ort_extension}")
    message(STATUS "Downloading ${ort_package}.${ort_extension}")
    file(DOWNLOAD
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ort_package}.${ort_extension}"
      "${archive}" SHOW_PROGRESS STATUS download_status)
    list(GET download_status 0 download_code)
    if(NOT download_code EQUAL 0)
      message(FATAL_ERROR "ONNX Runtime download failed: ${download_status}")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${CMAKE_BINARY_DIR}")
  endif()
endif()

find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(board2fen_cpp main.cpp)
target_include_directories(board2fen_cpp PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(board2fen_cpp PRIVATE "${ONNXRUNTIME_LIB}")
if(MSVC)
  target_compile_options(board2fen_cpp PRIVATE /W4 /permissive-)
else()
  target_compile_options(board2fen_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
  # Windows has an older onnxruntime.dll in System32; copy ours next to the program so it wins.
  add_custom_command(TARGET board2fen_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:board2fen_cpp>")
else()
  # Linux: remember where libonnxruntime.so is, so the program finds it without LD_LIBRARY_PATH.
  set_target_properties(board2fen_cpp PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()
