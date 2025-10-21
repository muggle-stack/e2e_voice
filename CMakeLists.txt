cmake_minimum_required(VERSION 3.16)
project(asr_cpp VERSION 3.0.0)

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

# Find OpenSSL for HMAC-SHA1 signature (required for cloud ASR)
find_package(OpenSSL REQUIRED)
if(OPENSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
    message(STATUS "OpenSSL include dir: ${OPENSSL_INCLUDE_DIR}")
    message(STATUS "OpenSSL libraries: ${OPENSSL_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenSSL not found. Please install with: brew install openssl")
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
include_directories(${OPENSSL_INCLUDE_DIR})
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

# Realtime ASR API source (one-sentence recognition)
set(ASR_REALTIME_API_SOURCES
    src/asr_realtime_api.cpp
)

# Speaker Recognition sources
set(SPEAKER_SOURCES
    src/speaker_embedder.cpp
    src/speaker_manager.cpp
    src/mel_filterbank.cpp
    src/onnx_utils.cpp
    src/wave_reader.cpp
    src/speaker_model_downloader.cpp
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

# ASR-Correct executable (ASR with error correction)
set(ASR_CORRECT_SOURCES ${COMMON_SOURCES} src/main_asr_correct.cpp)
add_executable(asr_correct ${ASR_CORRECT_SOURCES})

set(ASR_FC_SOURCES ${COMMON_SOURCES} src/main_fc.cpp)
add_executable(asr_fc ${ASR_FC_SOURCES})

# Options for cloud services (before defining executables)
option(USE_CLOUD_ASR "Use cloud ASR service instead of local model" OFF)
option(USE_CLOUD_LLM "Use cloud LLM API instead of local Ollama" OFF)

# ASR-LLM-TTS executable (with speaker recognition)
set(ASR_LLM_TTS_SOURCES ${COMMON_SOURCES} ${SPEAKER_SOURCES} ${TTS_SOURCES} src/text_buffer.cpp src/ordered_audio_queue.cpp src/main_asr_llm_tts.cpp)
if(USE_CLOUD_ASR)
    list(APPEND ASR_LLM_TTS_SOURCES src/asr_realtime_api.cpp)
endif()
if(USE_CLOUD_LLM)
    list(APPEND ASR_LLM_TTS_SOURCES src/api_comm.cpp)
endif()
add_executable(asr_llm_tts ${ASR_LLM_TTS_SOURCES})
target_compile_definitions(asr_llm_tts PRIVATE WITH_SPEAKER_RECOGNITION)
if(USE_CLOUD_ASR)
    target_compile_definitions(asr_llm_tts PRIVATE USE_CLOUD_ASR)
endif()
if(USE_CLOUD_LLM)
    target_compile_definitions(asr_llm_tts PRIVATE USE_CLOUD_LLM)
endif()

# ASR-LLM-TTS API executable
set(ASR_LLM_TTS_API_SOURCES ${COMMON_SOURCES} ${TTS_SOURCES} src/text_buffer.cpp src/ordered_audio_queue.cpp src/api_comm.cpp src/main_asr_llm_tts_api.cpp)
add_executable(asr_llm_tts_api ${ASR_LLM_TTS_API_SOURCES})

# Register Speaker executable (with speaker recognition)
set(REGISTER_SPEAKER_SOURCES ${SPEAKER_SOURCES} src/register_speaker.cpp)
add_executable(register_speaker ${REGISTER_SPEAKER_SOURCES})
target_compile_definitions(register_speaker PRIVATE WITH_SPEAKER_RECOGNITION)

# List Speakers executable (with speaker recognition)
set(LIST_SPEAKERS_SOURCES ${SPEAKER_SOURCES} src/list_speakers.cpp)
add_executable(list_speakers ${LIST_SPEAKERS_SOURCES})
target_compile_definitions(list_speakers PRIVATE WITH_SPEAKER_RECOGNITION)

# TTS executable (standalone)
set(TTS_STANDALONE_SOURCES ${TTS_SOURCES} src/main_tts.cpp)
add_executable(tts ${TTS_STANDALONE_SOURCES})

# Streaming ASR executable
set(STREAMING_ASR_SOURCES ${COMMON_SOURCES} src/streaming_audio_recorder.cpp src/asr_thread_pool.cpp src/main_streaming_asr.cpp)
add_executable(streaming_asr ${STREAMING_ASR_SOURCES})

# ASR Realtime Test executable (one-sentence recognition with AudioRecorder)
set(ASR_REALTIME_TEST_SOURCES
    ${ASR_REALTIME_API_SOURCES}
    src/audio_recorder.cpp
    src/vad_detector.cpp
    src/model_downloader.cpp
    src/main_asr_realtime_test.cpp)
add_executable(asr_realtime_test ${ASR_REALTIME_TEST_SOURCES})

# Common libraries
if(APPLE)
    set(COMMON_LIBRARIES
        ${ONNXRUNTIME_LIB}
        ${PORTAUDIO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${CURL_LIBRARIES}
        ${FFTW3_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        pthread
    )
    # Simplified libraries for API test programs (no audio I/O needed)
    # (Currently unused, kept for potential future simple test programs)
    # Libraries for realtime test (needs audio recording + VAD)
    set(REALTIME_TEST_LIBRARIES
        /opt/homebrew/lib/libportaudio.dylib
        /opt/homebrew/lib/libsndfile.dylib
        ${ONNXRUNTIME_LIB}
        ${CURL_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        pthread
    )
else()
    set(COMMON_LIBRARIES
        ${ONNXRUNTIME_LIB}
        ${PORTAUDIO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${CURL_LIBRARIES}
        ${FFTW3_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        pthread
    )
    # Libraries for realtime test (needs audio recording + VAD)
    set(REALTIME_TEST_LIBRARIES
        ${PORTAUDIO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${ONNXRUNTIME_LIB}
        ${CURL_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        pthread
    )
endif()

# Link libraries for ASR executable
target_link_libraries(asr ${COMMON_LIBRARIES})

# Link libraries for VAD+ASR executable
target_link_libraries(vad_asr ${COMMON_LIBRARIES})

# Link libraries for ASR-LLM executable
target_link_libraries(asr_llm ${COMMON_LIBRARIES})

# Link libraries for ASR-Correct executable
target_link_libraries(asr_correct ${COMMON_LIBRARIES})

# Link libraries for ASR_fc executable
target_link_libraries(asr_fc ${COMMON_LIBRARIES})

target_link_libraries(tts ${COMMON_LIBRARIES})

# Link libraries for ASR-LLM-TTS executable
target_link_libraries(asr_llm_tts ${COMMON_LIBRARIES})

# Link libraries for ASR-LLM-TTS API executable
target_link_libraries(asr_llm_tts_api ${COMMON_LIBRARIES})

# Link libraries for Register Speaker executable
target_link_libraries(register_speaker ${COMMON_LIBRARIES})

# Link libraries for List Speakers executable
target_link_libraries(list_speakers ${COMMON_LIBRARIES})

# Link libraries for Streaming ASR executable
target_link_libraries(streaming_asr ${COMMON_LIBRARIES})

# Link libraries for ASR Realtime Test executable (needs AudioRecorder+VAD)
target_link_libraries(asr_realtime_test ${REALTIME_TEST_LIBRARIES})

# Add library search paths for all executables
target_link_directories(asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(vad_asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(vad_asr PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_llm PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_llm PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_correct PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_correct PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_fc PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_fc PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(tts PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(tts PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_llm_tts PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_llm_tts PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(asr_llm_tts_api PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(asr_llm_tts_api PRIVATE ${SNDFILE_LIBRARY_DIRS})

target_link_directories(register_speaker PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(register_speaker PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(list_speakers PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(list_speakers PRIVATE ${SNDFILE_LIBRARY_DIRS})
target_link_directories(streaming_asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_directories(streaming_asr PRIVATE ${SNDFILE_LIBRARY_DIRS})

# Compiler flags for all executables
target_compile_options(asr PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(vad_asr PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(vad_asr PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_llm PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_llm PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_correct PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_correct PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_fc PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_fc PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(tts PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(tts PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_llm_tts PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_llm_tts PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_llm_tts_api PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(asr_llm_tts_api PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(register_speaker PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(register_speaker PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(list_speakers PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(list_speakers PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(streaming_asr PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(streaming_asr PRIVATE ${SNDFILE_CFLAGS_OTHER})
target_compile_options(asr_realtime_test PRIVATE ${SNDFILE_CFLAGS_OTHER})

# Set output directory for all executables
set_target_properties(asr vad_asr asr_llm asr_correct asr_fc asr_llm_tts asr_llm_tts_api tts streaming_asr register_speaker list_speakers asr_realtime_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy models directory if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
endif()
