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
    find_library(FFTW3F_LIB_PATH NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include)
    if(NOT FFTW3F_LIB_PATH OR NOT FFTW3_INCLUDE_DIRS)
        message(FATAL_ERROR "FFTW3 not found. Please install with: brew install fftw")
    endif()
    set(FFTW3_LIBRARY_DIRS /opt/homebrew/lib)
else()
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
endif()

# Options
option(USE_MCP "Enable MCP (Model Context Protocol) support" OFF)
option(USE_AEC "Enable WebRTC AEC (Acoustic Echo Cancellation) for barge-in support" ON)

# Add mcp library (Model Context Protocol)
if(USE_MCP)
    set(MCP_BUILD_EXAMPLES OFF CACHE BOOL "Disable MCP examples" FORCE)
    add_subdirectory(modules/mcp)
    message(STATUS "MCP support enabled")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(${FFTW3_INCLUDE_DIRS})

# WebRTC Audio Processing for AEC
if(USE_AEC)
    set(WEBRTC_DIR ${CMAKE_SOURCE_DIR}/modules/webrtc-audio-processing)
    set(WEBRTC_BUILD_DIR ${WEBRTC_DIR}/build)

    # Check if WebRTC is built
    if(NOT EXISTS ${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.dylib AND
       NOT EXISTS ${WEBRTC_BUILD_DIR}/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.so)
        message(FATAL_ERROR "WebRTC Audio Processing not built. Please run: cd modules/webrtc-audio-processing && meson build && ninja -C build")
    endif()

    # Find abseil (required by WebRTC)
    if(APPLE)
        execute_process(
            COMMAND brew --prefix abseil
            OUTPUT_VARIABLE ABSEIL_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(ABSEIL_PREFIX)
            set(ABSEIL_INCLUDE_DIR ${ABSEIL_PREFIX}/include)
        else()
            set(ABSEIL_INCLUDE_DIR /opt/homebrew/include)
        endif()
    else()
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

# Copy models directory if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
endif()

# ============================================================================
# Modular libraries
# ============================================================================
add_subdirectory(modules/audio)
add_subdirectory(modules/stt)
add_subdirectory(modules/tts)
add_subdirectory(modules/vad)
add_subdirectory(modules/llm)

# ============================================================================
# voice_chat_aec - Full-duplex voice chat with WebRTC AEC + barge-in
# ============================================================================
if(USE_AEC)
    set(VOICE_CHAT_AEC_SOURCES
        examples/main_voice_chat_aec.cpp
        src/aec_duplex_processor.cpp
        src/text_buffer.cpp
    )

    add_executable(voice_chat_aec ${VOICE_CHAT_AEC_SOURCES})

    target_include_directories(voice_chat_aec PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/modules/audio/inc
        ${CMAKE_SOURCE_DIR}/modules/stt/inc
        ${CMAKE_SOURCE_DIR}/modules/tts/inc
        ${CMAKE_SOURCE_DIR}/modules/vad/inc
        ${CMAKE_SOURCE_DIR}/modules/llm/inc
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
        llm
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

    # Optional MCP support
    if(USE_MCP)
        target_compile_definitions(voice_chat_aec PRIVATE USE_MCP)
        # Retrieve FetchContent'd nlohmann/json path (populated by modules/mcp)
        include(FetchContent)
        FetchContent_GetProperties(json)
        # Use BEFORE to ensure FetchContent'd json (v3.11.3) takes priority
        # over system-installed nlohmann-json, preventing ABI mismatch
        target_include_directories(voice_chat_aec BEFORE PRIVATE
            ${json_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/modules/mcp/inc
        )
        target_link_libraries(voice_chat_aec mcp_static)
        message(STATUS "  + MCP support enabled for voice_chat_aec")
    endif()
endif()
