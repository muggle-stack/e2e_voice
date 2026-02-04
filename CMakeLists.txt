cmake_minimum_required(VERSION 3.16)

# Use GCC-14 for better RISC-V support if available (must be before project())
find_program(GCC14 gcc-14)
find_program(GXX14 g++-14)

if(GCC14 AND GXX14)
    set(CMAKE_C_COMPILER ${GCC14} CACHE PATH "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER ${GXX14} CACHE PATH "CXX compiler" FORCE)
endif()

project(asr_cpp VERSION 3.0.0)

if(GCC14 AND GXX14)
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
    find_library(FFTW3F_LIB_PATH NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include)
    if(NOT FFTW3F_LIB_PATH OR NOT FFTW3_INCLUDE_DIRS)
        message(FATAL_ERROR "FFTW3 not found. Please install with: brew install fftw")
    endif()
    # Use full path (CACHE INTERNAL to avoid being overwritten by subprojects)
    set(MAIN_FFTW3_LIBRARIES ${FFTW3F_LIB_PATH} CACHE INTERNAL "FFTW3F library path")
    set(FFTW3_LIBRARY_DIRS /opt/homebrew/lib)
else()
    # Linux使用pkg-config
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
    set(MAIN_FFTW3_LIBRARIES ${FFTW3_LIBRARIES} CACHE INTERNAL "FFTW3F library path")
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

# Add mcp library (Model Context Protocol)
option(USE_MCP "Enable MCP (Model Context Protocol) support" OFF)
if(USE_MCP)
    set(MCP_BUILD_EXAMPLES OFF CACHE BOOL "Disable MCP examples" FORCE)
    add_subdirectory(mcp)
    message(STATUS "MCP support enabled")
endif()

# Use official cppjieba
set(JIEBA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
set(JIEBA_DEPS_DIR ${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps)
set(JIEBA_DICT_DIR ${CMAKE_SOURCE_DIR}/third_party/cppjieba/dict)

# Use cpp-pinyin for Chinese to Pinyin conversion (zh-en TTS)
set(CPP_PINYIN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cpp-pinyin/include)
set(CPP_PINYIN_SRC_DIR ${CMAKE_SOURCE_DIR}/third_party/cpp-pinyin/src)
set(CPP_PINYIN_DICT_DIR ${CMAKE_SOURCE_DIR}/third_party/cpp-pinyin/res/dict)

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
if(CPP_PINYIN_INCLUDE_DIR)
    include_directories(${CPP_PINYIN_INCLUDE_DIR})
    include_directories(${CPP_PINYIN_SRC_DIR})
    include_directories(${CPP_PINYIN_SRC_DIR}/toneUtil)
    add_definitions(-DCPP_PINYIN_STATIC)
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

# cpp-pinyin source files (for zh-en TTS)
set(CPP_PINYIN_SOURCES
    ${CPP_PINYIN_SRC_DIR}/ChineseG2p.cpp
    ${CPP_PINYIN_SRC_DIR}/DictUtil.cpp
    ${CPP_PINYIN_SRC_DIR}/G2pglobal.cpp
    ${CPP_PINYIN_SRC_DIR}/Pinyin.cpp
    ${CPP_PINYIN_SRC_DIR}/PinyinRes.cpp
    ${CPP_PINYIN_SRC_DIR}/ToneConverter.cpp
    ${CPP_PINYIN_SRC_DIR}/U16Str.cpp
    ${CPP_PINYIN_SRC_DIR}/toneUtil/ManTone.cpp
    ${CPP_PINYIN_SRC_DIR}/toneUtil/ManToneUtil.cpp
)

# VAD+ASR executable
set(VAD_ASR_SOURCES ${COMMON_SOURCES} example/main_asr.cpp)
add_executable(vad_asr ${VAD_ASR_SOURCES})

# ASR executable (file processing)
set(ASR_SOURCES ${ASE_SOURCES} example/main_ase.cpp)
add_executable(asr ${ASR_SOURCES})

# ASR-LLM executable
set(ASR_LLM_SOURCES ${COMMON_SOURCES} example/main_llm.cpp)
add_executable(asr_llm ${ASR_LLM_SOURCES})

# ASR-Correct executable (ASR with error correction)
set(ASR_CORRECT_SOURCES ${COMMON_SOURCES} example/main_asr_correct.cpp)
add_executable(asr_correct ${ASR_CORRECT_SOURCES})

set(ASR_FC_SOURCES ${COMMON_SOURCES} example/main_fc.cpp)
add_executable(asr_fc ${ASR_FC_SOURCES})

# Options for cloud services (before defining executables)
option(USE_CLOUD_ASR "Use cloud ASR service instead of local model" OFF)
option(USE_CLOUD_LLM "Use cloud LLM API instead of local Ollama" OFF)
option(USE_AEC "Enable WebRTC AEC (Acoustic Echo Cancellation) for barge-in support" OFF)

# WebRTC Audio Processing for AEC
if(USE_AEC)
    set(WEBRTC_DIR ${CMAKE_SOURCE_DIR}/webrtc-audio-processing)
    set(WEBRTC_BUILD_DIR ${WEBRTC_DIR}/build)

    # Check if WebRTC is built
    if(NOT EXISTS ${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.dylib AND
       NOT EXISTS ${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.so)
        message(FATAL_ERROR "WebRTC Audio Processing not built. Please run: cd webrtc-audio-processing && meson build && ninja -C build")
    endif()

    # Find abseil (required by WebRTC)
    if(APPLE)
        # macOS: find abseil via Homebrew
        execute_process(
            COMMAND brew --prefix abseil
            OUTPUT_VARIABLE ABSEIL_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(ABSEIL_PREFIX)
            set(ABSEIL_INCLUDE_DIR ${ABSEIL_PREFIX}/include)
        else()
            # Fallback to common Homebrew path
            set(ABSEIL_INCLUDE_DIR /opt/homebrew/include)
        endif()
    else()
        # Linux: check multiple locations
        find_path(ABSEIL_INCLUDE_DIR absl/base/config.h
            PATHS
            /usr/include
            /usr/local/include
            ${WEBRTC_BUILD_DIR}/subprojects/abseil-cpp-20240722.0
            ${WEBRTC_BUILD_DIR}/subprojects/abseil-cpp
        )
        if(NOT ABSEIL_INCLUDE_DIR)
            message(FATAL_ERROR "Abseil not found. Install with: sudo apt install libabsl-dev")
        endif()
    endif()

    # WebRTC include directories
    set(WEBRTC_INCLUDE_DIRS
        ${WEBRTC_DIR}
        ${WEBRTC_DIR}/webrtc
        ${WEBRTC_BUILD_DIR}
        ${WEBRTC_BUILD_DIR}/webrtc
        ${ABSEIL_INCLUDE_DIR}
    )

    # WebRTC library
    if(APPLE)
        set(WEBRTC_LIB ${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.dylib)
    else()
        set(WEBRTC_LIB ${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.so)
    endif()

    message(STATUS "WebRTC AEC support enabled")
    message(STATUS "WebRTC include dirs: ${WEBRTC_INCLUDE_DIRS}")
    message(STATUS "Abseil include dir: ${ABSEIL_INCLUDE_DIR}")
    message(STATUS "WebRTC library: ${WEBRTC_LIB}")
endif()

# ASR-LLM-TTS executable (with speaker recognition)
set(ASR_LLM_TTS_SOURCES ${COMMON_SOURCES} ${SPEAKER_SOURCES} ${TTS_SOURCES} ${CPP_PINYIN_SOURCES} src/text_buffer.cpp src/ordered_audio_queue.cpp example/main_asr_llm_tts.cpp)
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
set(ASR_LLM_TTS_API_SOURCES ${COMMON_SOURCES} ${TTS_SOURCES} ${CPP_PINYIN_SOURCES} src/text_buffer.cpp src/ordered_audio_queue.cpp src/api_comm.cpp example/main_asr_llm_tts_api.cpp)
add_executable(asr_llm_tts_api ${ASR_LLM_TTS_API_SOURCES})

# Register Speaker executable (with speaker recognition)
set(REGISTER_SPEAKER_SOURCES ${SPEAKER_SOURCES} src/register_speaker.cpp)
add_executable(register_speaker ${REGISTER_SPEAKER_SOURCES})
target_compile_definitions(register_speaker PRIVATE WITH_SPEAKER_RECOGNITION)

# List Speakers executable (with speaker recognition)
set(LIST_SPEAKERS_SOURCES ${SPEAKER_SOURCES} src/list_speakers.cpp)
add_executable(list_speakers ${LIST_SPEAKERS_SOURCES})
target_compile_definitions(list_speakers PRIVATE WITH_SPEAKER_RECOGNITION)

# TTS demo executable (standalone, renamed to avoid conflict with tts library)
set(TTS_STANDALONE_SOURCES ${TTS_SOURCES} ${CPP_PINYIN_SOURCES} example/main_tts.cpp)
add_executable(tts_demo ${TTS_STANDALONE_SOURCES})

# Streaming ASR executable
set(STREAMING_ASR_SOURCES ${COMMON_SOURCES} src/streaming_audio_recorder.cpp src/asr_thread_pool.cpp example/main_streaming_asr.cpp)
add_executable(streaming_asr ${STREAMING_ASR_SOURCES})

# ASR Realtime Test executable (one-sentence recognition with AudioRecorder)
set(ASR_REALTIME_TEST_SOURCES
    ${ASR_REALTIME_API_SOURCES}
    src/audio_recorder.cpp
    src/vad_detector.cpp
    src/model_downloader.cpp
    example/main_asr_realtime_test.cpp)
add_executable(asr_realtime_test ${ASR_REALTIME_TEST_SOURCES})

# Common libraries
if(APPLE)
    set(COMMON_LIBRARIES
        ${ONNXRUNTIME_LIB}
        ${PORTAUDIO_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${CURL_LIBRARIES}
        /opt/homebrew/lib/libfftw3f.dylib
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
        ${MAIN_FFTW3_LIBRARIES}
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

target_link_libraries(tts_demo ${COMMON_LIBRARIES})

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
target_link_directories(asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(vad_asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(asr_llm PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(asr_correct PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(asr_fc PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(tts_demo PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(asr_llm_tts PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(asr_llm_tts_api PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(register_speaker PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(list_speakers PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
target_link_directories(streaming_asr PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})

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
target_compile_options(tts_demo PRIVATE ${PORTAUDIO_CFLAGS_OTHER})
target_compile_options(tts_demo PRIVATE ${SNDFILE_CFLAGS_OTHER})
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
set_target_properties(asr vad_asr asr_llm asr_correct asr_fc asr_llm_tts asr_llm_tts_api tts_demo streaming_asr register_speaker list_speakers asr_realtime_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy models directory if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
endif()

# MCP examples (calculator server and client)
if(USE_MCP)
    # MCP Calculator Server
    add_executable(mcp_calculator_server example/mcp_calculator_server.cpp)
    target_include_directories(mcp_calculator_server PRIVATE ${MCP_INCLUDE_DIR} ${MCP_COMMON_DIR})
    target_link_libraries(mcp_calculator_server mcp pthread)
    set_target_properties(mcp_calculator_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # MCP Calculator Client (test client)
    add_executable(mcp_calculator_client example/mcp_calculator_client.cpp)
    target_include_directories(mcp_calculator_client PRIVATE ${MCP_INCLUDE_DIR} ${MCP_COMMON_DIR})
    target_link_libraries(mcp_calculator_client mcp pthread)
    set_target_properties(mcp_calculator_client PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # ASR-LLM-TTS with MCP support
    set(ASR_LLM_TTS_MCP_SOURCES ${COMMON_SOURCES} ${SPEAKER_SOURCES} ${TTS_SOURCES} ${CPP_PINYIN_SOURCES}
        src/text_buffer.cpp src/ordered_audio_queue.cpp example/main_asr_llm_tts_mcp.cpp)
    if(USE_CLOUD_ASR)
        list(APPEND ASR_LLM_TTS_MCP_SOURCES src/asr_realtime_api.cpp)
    endif()
    if(USE_CLOUD_LLM)
        list(APPEND ASR_LLM_TTS_MCP_SOURCES src/api_comm.cpp)
    endif()
    add_executable(asr_llm_tts_mcp ${ASR_LLM_TTS_MCP_SOURCES})
    target_compile_definitions(asr_llm_tts_mcp PRIVATE WITH_SPEAKER_RECOGNITION USE_MCP)
    if(USE_CLOUD_ASR)
        target_compile_definitions(asr_llm_tts_mcp PRIVATE USE_CLOUD_ASR)
    endif()
    if(USE_CLOUD_LLM)
        target_compile_definitions(asr_llm_tts_mcp PRIVATE USE_CLOUD_LLM)
    endif()
    target_include_directories(asr_llm_tts_mcp PRIVATE ${MCP_INCLUDE_DIR} ${MCP_COMMON_DIR})
    target_link_libraries(asr_llm_tts_mcp ${COMMON_LIBRARIES} mcp)
    target_link_directories(asr_llm_tts_mcp PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
    target_compile_options(asr_llm_tts_mcp PRIVATE ${PORTAUDIO_CFLAGS_OTHER} ${SNDFILE_CFLAGS_OTHER})
    set_target_properties(asr_llm_tts_mcp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# AEC-enabled ASR-LLM-TTS executable (with WebRTC echo cancellation for barge-in)
if(USE_AEC)
    # AEC source files
    set(AEC_SOURCES
        src/aec_audio_processor.cpp
    )

    set(ASR_LLM_TTS_AEC_SOURCES
        ${COMMON_SOURCES}
        ${SPEAKER_SOURCES}
        ${TTS_SOURCES}
        ${CPP_PINYIN_SOURCES}
        ${AEC_SOURCES}
        src/text_buffer.cpp
        src/ordered_audio_queue.cpp
        example/main_asr_llm_tts_aec.cpp
    )

    if(USE_CLOUD_ASR)
        list(APPEND ASR_LLM_TTS_AEC_SOURCES src/asr_realtime_api.cpp)
    endif()
    if(USE_CLOUD_LLM)
        list(APPEND ASR_LLM_TTS_AEC_SOURCES src/api_comm.cpp)
    endif()

    add_executable(asr_llm_tts_aec ${ASR_LLM_TTS_AEC_SOURCES})

    # Add WebRTC includes
    target_include_directories(asr_llm_tts_aec PRIVATE ${WEBRTC_INCLUDE_DIRS})

    # Compile definitions
    target_compile_definitions(asr_llm_tts_aec PRIVATE WITH_SPEAKER_RECOGNITION USE_AEC)
    if(USE_CLOUD_ASR)
        target_compile_definitions(asr_llm_tts_aec PRIVATE USE_CLOUD_ASR)
    endif()
    if(USE_CLOUD_LLM)
        target_compile_definitions(asr_llm_tts_aec PRIVATE USE_CLOUD_LLM)
    endif()

    # Link libraries (including WebRTC)
    target_link_libraries(asr_llm_tts_aec ${COMMON_LIBRARIES} ${WEBRTC_LIB})

    target_link_directories(asr_llm_tts_aec PRIVATE ${PORTAUDIO_LIBRARY_DIRS} ${SNDFILE_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS})
    target_compile_options(asr_llm_tts_aec PRIVATE ${PORTAUDIO_CFLAGS_OTHER} ${SNDFILE_CFLAGS_OTHER})
    set_target_properties(asr_llm_tts_aec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # Set RPATH for WebRTC library
    if(APPLE)
        set_target_properties(asr_llm_tts_aec PROPERTIES
            INSTALL_RPATH "@executable_path/../lib;${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(asr_llm_tts_aec PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib:${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    message(STATUS "Building asr_llm_tts_aec with WebRTC AEC support")
endif()

# ============================================================================
# voice_chat - 简洁语音对话系统 (使用模块化 stt/tts/audio/vad 库)
# ============================================================================

# 添加模块化子库
add_subdirectory(audio)
add_subdirectory(stt)
add_subdirectory(tts)
add_subdirectory(vad)

# Voice Chat 源文件 (使用新的 VAD SDK)
set(VOICE_CHAT_SOURCES
    example/main_voice_chat.cpp
)

add_executable(voice_chat ${VOICE_CHAT_SOURCES})

target_include_directories(voice_chat PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/audio/space_audio/include
    ${CMAKE_SOURCE_DIR}/audio/resampler/include
    ${CMAKE_SOURCE_DIR}/stt/inc
    ${CMAKE_SOURCE_DIR}/tts/inc
    ${CMAKE_SOURCE_DIR}/vad/inc
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(voice_chat
    space_audio
    audio_resampler
    stt
    tts
    vad
    ${ONNXRUNTIME_LIB}
    ${CURL_LIBRARIES}
    pthread
)

target_link_directories(voice_chat PRIVATE
    ${PORTAUDIO_LIBRARY_DIRS}
    ${SNDFILE_LIBRARY_DIRS}
    ${FFTW3_LIBRARY_DIRS}
)

set_target_properties(voice_chat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Building voice_chat with modular stt/tts/audio/vad libraries")

# ============================================================================
# voice_chat_aec - 带回声消除的语音对话系统 (全双工 + WebRTC AEC)
# ============================================================================
if(USE_AEC)
    set(VOICE_CHAT_AEC_SOURCES
        example/main_voice_chat_aec.cpp
        src/aec_duplex_processor.cpp
        src/text_buffer.cpp
    )

    add_executable(voice_chat_aec ${VOICE_CHAT_AEC_SOURCES})

    target_include_directories(voice_chat_aec PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/audio/space_audio/include
        ${CMAKE_SOURCE_DIR}/audio/stream/include
        ${CMAKE_SOURCE_DIR}/audio/resampler/include
        ${CMAKE_SOURCE_DIR}/stt/inc
        ${CMAKE_SOURCE_DIR}/tts/inc
        ${CMAKE_SOURCE_DIR}/vad/inc
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${WEBRTC_INCLUDE_DIRS}
    )

    target_compile_definitions(voice_chat_aec PRIVATE USE_AEC)

    target_link_libraries(voice_chat_aec
        space_audio
        audio_resampler
        stt
        tts
        vad
        ${ONNXRUNTIME_LIB}
        ${CURL_LIBRARIES}
        ${WEBRTC_LIB}
        pthread
    )

    target_link_directories(voice_chat_aec PRIVATE
        ${PORTAUDIO_LIBRARY_DIRS}
        ${SNDFILE_LIBRARY_DIRS}
        ${FFTW3_LIBRARY_DIRS}
    )

    set_target_properties(voice_chat_aec PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Set RPATH for WebRTC library
    if(APPLE)
        set_target_properties(voice_chat_aec PROPERTIES
            INSTALL_RPATH "@executable_path/../lib;${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(voice_chat_aec PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib:${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    message(STATUS "Building voice_chat_aec with WebRTC AEC + full-duplex audio")

    # 可选 MCP 支持
    if(USE_MCP)
        target_compile_definitions(voice_chat_aec PRIVATE USE_MCP)
        target_include_directories(voice_chat_aec PRIVATE
            ${CMAKE_SOURCE_DIR}/mcp/inc
        )
        target_link_libraries(voice_chat_aec mcp_static)
        message(STATUS "  + MCP support enabled for voice_chat_aec")
    endif()
endif()
