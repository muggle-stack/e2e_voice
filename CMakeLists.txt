cmake_minimum_required(VERSION 3.16)
project(asr_cpp VERSION 1.0.0)

# Use GCC-14 for better RISC-V support if available
find_program(GCC14 gcc-14)
find_program(GXX14 g++-14)

if(GCC14 AND GXX14)
    set(CMAKE_C_COMPILER gcc-14)
    set(CMAKE_CXX_COMPILER g++-14)
    message(STATUS "Using GCC-14 for better RISC-V support")
else()
    message(STATUS "GCC-14 not found, using default compiler")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug/Release/RelWithDebInfo/MinSizeRel)" FORCE)
endif()

# Ensure Release mode has proper optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
endif()

# RISC-V specific flags for memory alignment
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv|RISCV")
    message(STATUS "Detected RISC-V architecture, adding alignment flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mstrict-align -fno-aggressive-loop-optimizations")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mstrict-align -fno-aggressive-loop-optimizations")
    # Disable SIMD optimizations that may not be available on RISC-V
    add_definitions(-DFFTW_NO_SIMD)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# Find PortAudio
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Find libsndfile
pkg_check_modules(SNDFILE REQUIRED sndfile)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS 
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
    /opt/homebrew/include/onnxruntime
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS 
    /usr/local/lib
    /usr/lib
    /opt/homebrew/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found. Please install ONNX Runtime C++ library.")
endif()

# Find cURL for downloading models
find_package(CURL REQUIRED)

# Find FFTW3
if(APPLE)
    # macOS通过Homebrew安装的FFTW
    find_library(FFTW3_LIBRARIES NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include)
    if(NOT FFTW3_LIBRARIES OR NOT FFTW3_INCLUDE_DIRS)
        message(FATAL_ERROR "FFTW3 not found. Please install with: brew install fftw")
    endif()
else()
    # Linux使用pkg-config
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
endif()

# Find libarchive for tar.gz extraction (optional)
find_package(LibArchive QUIET)
if(NOT LibArchive_FOUND)
    message(WARNING "libarchive not found. Using system tar command for extraction.")
endif()

# Use official cppjieba
set(JIEBA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
set(JIEBA_DEPS_DIR ${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps)
set(JIEBA_DICT_DIR ${CMAKE_SOURCE_DIR}/third_party/cppjieba/dict)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(${PORTAUDIO_INCLUDE_DIRS})
include_directories(${SNDFILE_INCLUDE_DIRS})
include_directories(${FFTW3_INCLUDE_DIRS})
if(LibArchive_FOUND)
    include_directories(${LibArchive_INCLUDE_DIRS})
endif()
if(JIEBA_INCLUDE_DIR)
    include_directories(${JIEBA_INCLUDE_DIR})
    include_directories(${JIEBA_DEPS_DIR}/limonp/include)
endif()

# Common source files (shared between both executables)
set(COMMON_SOURCES
    src/audio_recorder.cpp
    src/vad_detector.cpp
    src/asr_model.cpp
    src/audio_processor.cpp
    src/tokenizer.cpp
    src/model_downloader.cpp
)

# ASE source files
set(ASE_SOURCES
    src/asr_model.cpp
    src/audio_processor.cpp
    src/tokenizer.cpp
    src/model_downloader.cpp
)


# TTS source files
set(TTS_SOURCES
    src/tts/tts_model.cpp
    src/tts/tts_model_downloader.cpp
    src/tts_demo.cpp
)

# VAD+ASR executable
set(VAD_ASR_SOURCES ${COMMON_SOURCES} src/main_asr.cpp)
add_executable(vad_asr ${VAD_ASR_SOURCES})

# ASR executable (file processing)
set(ASR_SOURCES ${ASE_SOURCES} src/main_ase.cpp)
add_executable(asr ${ASR_SOURCES})

# ASR-LLM executable
set(ASR_LLM_SOURCES ${COMMON_SOURCES} src/main_llm.cpp)
add_executable(asr_llm ${ASR_LLM_SOURCES})

set(ASR_FC_SOURCES ${COMMON_SOURCES} src/main_fc.cpp)
add_executable(asr_fc ${ASR_FC_SOURCES})

# ASR-LLM-TTS executable
set(ASR_LLM_TTS_SOURCES ${COMMON_SOURCES} ${TTS_SOURCES} src/text_buffer.cpp src/ordered_audio_queue.cpp src/main_asr_llm_tts.cpp)
add_executable(asr_llm_tts ${ASR_LLM_TTS_SOURCES})

# TTS executable (standalone)
set(TTS_STANDALONE_SOURCES ${TTS_SOURCES} src/main_tts.cpp)
add_executable(tts ${TTS_STANDALONE_SOURCES})

# Common libraries
if(APPLE)
    set(COMMON_LIBRARIES
        ${ONNXRUNTIME_LIB}
        ${PORTAUDIO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${CURL_LIBRARIES}
        ${FFTW3_LIBRARIES}
        pthread
    )
else()
    set(COMMON_LIBRARIES
        ${ONNXRUNTIME_LIB}
        ${PORTAUDIO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${CURL_LIBRARIES}
        ${FFTW3_LIBRARIES}
        pthread
    )
endif()

# Link libraries for ASR executable
target_link_libraries(asr ${COMMON_LIBRARIES})

# Link libraries for VAD+ASR executable
target_link_libraries(vad_asr ${COMMON_LIBRARIES})

# Link libraries for ASR-LLM executable
target_link_libraries(asr_llm ${COMMON_LIBRARIES})

# Link libraries for ASR_fc executable
target_link_libraries(asr_fc ${COMMON_LIBRARIES})

target_link_libraries(tts ${COMMON_LIBRARIES})

# Link libraries for ASR-LLM-TTS executable
target_link_libraries(asr_llm_tts ${COMMON_LIBRARIES})

# Add library search paths for all executables
target_link_directories(asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(vad_asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(vad_asr PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_llm PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_llm PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_fc PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_fc PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(tts PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(tts PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_llm_tts PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_llm_tts PRIVATE ${SNDFILE_LIBRARY_DIRS})

# Compiler flags for all executables
target_compile_options(asr PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(vad_asr PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(vad_asr PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_llm PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_llm PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_fc PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_fc PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(tts PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(tts PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_llm_tts PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_llm_tts PRIVATE ${SNDFILE_CFLAGS_OTHER})

# Set output directory for all executables
set_target_properties(asr vad_asr asr_llm asr_fc asr_llm_tts tts PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy models directory if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
endif()
