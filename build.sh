#!/bin/bash

# ASR C++ Build Script
set -e

echo "Building SpaceMit AI Speech Recognition Applications"
echo "=================================================="

# Function to check for a command's existence
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: Required command '$1' not found."
        echo "Please install it before proceeding."
        exit 1
    fi
}

# Function to check for a pkg-config package's existence
check_pkg_config_package() {
    if ! pkg-config --exists "$1"; then
        echo "Error: Required development package '$1' not found."
        echo "Please install it before proceeding."
        echo "  - For macOS (Homebrew): brew install $2"
        echo "  - For Ubuntu/Debian: sudo apt install $3"
        exit 1
    fi
}

echo "Checking essential build dependencies..."

# --- Check for Compilers (GCC/G++) ---
# Try to find specific gcc-14/g++-14 first, then fall back to generic gcc/g++
if command -v g++-14 &> /dev/null; then
    echo "G++ 14 Compiler (g++-14) found."
else
    # Fallback to generic g++
    check_command "g++"
    echo "Generic G++ Compiler (g++) found. (Consider installing g++-14 if available for specific optimizations)"
fi

if command -v gcc-14 &> /dev/null; then
    echo "GCC 14 Compiler (gcc-14) found."
else
    # Fallback to generic gcc
    check_command "gcc"
    echo "Generic GCC Compiler (gcc) found. (Consider installing gcc-14 if available for specific optimizations)"
fi

# --- Check for CMake ---
check_command "cmake"
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo "CMake $CMAKE_VERSION found."
# Optional: enforce minimum CMake version
# if (( $(echo "$CMAKE_VERSION < 3.16" | bc -l) )); then
#     echo "Error: CMake version must be 3.16 or higher. Found $CMAKE_VERSION."
#     exit 1
# fi


# --- Check for pkg-config ---
# pkg-config is usually needed for C++ development on Linux/macOS to find libs
check_command "pkg-config"


# --- Check for PortAudio development files ---
# macOS: portaudio (Homebrew installs dev files)
# Ubuntu: libportaudio-dev
check_pkg_config_package "portaudio-2.0" "portaudio" "libportaudio-dev"
echo "PortAudio development files found."


# --- Check for libsndfile development files ---
# macOS: libsndfile (Homebrew installs dev files)
# Ubuntu: libsndfile1-dev
check_pkg_config_package "sndfile" "libsndfile" "libsndfile1-dev"
echo "libsndfile development files found."


# --- Check for ONNX Runtime headers (specific file check as pkg-config might not cover it universally) ---
ONNXRUNTIME_FOUND=false
# Paths for macOS (Homebrew) and Linux common install locations
for path in /usr/local/include/onnxruntime /usr/include/onnxruntime /opt/homebrew/include/onnxruntime /usr/local/include /usr/include; do
    if [ -f "$path/onnxruntime_c_api.h" ]; then
        ONNXRUNTIME_FOUND=true
        echo "ONNX Runtime headers found at $path"
        break
    fi
done

if [ "$ONNXRUNTIME_FOUND" = false ]; then
    echo "Error: ONNX Runtime C++ library headers not found."
    echo "Please install ONNX Runtime C++ library."
    echo "  - For macOS (Homebrew): brew install onnxruntime"
    echo "  - For Ubuntu/Debian: Download pre-built packages from https://github.com/microsoft/onnxruntime/releases"
    echo "    and extract to a system path (e.g., /usr/local) or set ONNXRUNTIME_HOME environment variable."
    exit 1
fi


# --- Check for cURL development files ---
# macOS: curl (Homebrew installs dev files)
# Ubuntu: libcurl4-openssl-dev
check_pkg_config_package "libcurl" "curl" "libcurl4-openssl-dev"
echo "cURL development files found."

# --- Check for FFTW3 development files ---
# macOS: fftw (Homebrew installs dev files)
# Ubuntu: libfftw3-dev
check_pkg_config_package "fftw3f" "fftw" "libfftw3-dev"
echo "FFTW3 development files found."

echo "All essential build dependencies checked."
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo ""
echo "Configuring build..."
cmake ..

# Build
echo ""
echo "Building..."
# Use nproc for Linux, sysctl for macOS for parallel build jobs
MAKE_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Using $MAKE_JOBS parallel jobs."
make -j$MAKE_JOBS

echo ""
echo "Build completed successfully!"
echo ""
echo "Built applications:"
echo "   vad_asr     - VAD + ASR real-time speech recognition (microphone input)"
echo "   asr         - Audio file processing engine (batch processing)"
echo "   asr_llm     - ASR + Large Language Model integration"
echo "   asr_fc      - ASR + Function Calling"
echo "   tts         - Text-to-Speech synthesis tool"
echo "   asr_llm_tts - Complete ASR + LLM + TTS conversation system"
echo ""
echo "To run the applications:"
echo "   cd build"
echo ""
echo "   # Real-time speech recognition"
echo "   ./bin/vad_asr"
echo ""
echo "   # Audio file processing"
echo "   ./bin/asr audio.wav"
echo "   ./bin/asr file1.wav file2.wav  # batch processing"
echo ""
echo "   # ASR with LLM integration"
echo "   ./bin/asr_llm"
echo ""
echo "   # ASR with Function Calling"
echo "   ./bin/asr_fc"
echo ""
echo "   # Text-to-Speech synthesis"
echo "   ./bin/tts --text \"你好世界\" --save_audio_path output.wav"
echo "   ./bin/tts --help"
echo ""
echo "   # Complete conversation system (ASR + LLM + TTS)"
echo "   ./bin/asr_llm_tts"
echo "   ./bin/asr_llm_tts --help"
echo ""
echo "Build artifacts:"
echo "   Executables: build/bin/vad_asr, build/bin/asr, build/bin/asr_llm, build/bin/asr_fc, build/bin/tts, build/bin/asr_llm_tts"
echo "   Build logs: build/"
echo ""
echo "Next steps:"
echo "   1. Choose the appropriate application for your use case:"
echo "      - asr: Batch audio file processing (WAV/FLAC/OGG)"
echo "      - vad_asr: Real-time microphone input with VAD"
echo "      - tts: Text-to-Speech synthesis (standalone)"
echo "      - asr_llm_tts: Complete conversation system"
echo "   2. Models will be downloaded automatically on first run"
echo "   3. For real-time apps: Press Enter to start recording when prompted"
echo "   4. For file processing: Provide audio file path(s) as argument(s)"
echo "   5. For TTS: Use --text parameter to specify text for synthesis"
