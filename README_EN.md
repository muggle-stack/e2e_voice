# End-to-End C++ Intelligent Voice Dialogue System

A complete Chinese intelligent voice dialogue system integrating Automatic Speech Recognition (ASR), Large Language Models (LLM), and Text-to-Speech (TTS) functionality. Supports real-time voice interaction with long-range recording enhancement and stereo sampling processing capabilities, suitable for conference rooms and classrooms.

## 🎯 Project Features

### 🔊 Complete Voice Dialogue Pipeline
- **ASR (Speech Recognition)**: Support for local SenseVoice model or cloud-based Aliyun real-time ASR
- **Audio File Processing**: Lightweight engine dedicated to batch processing of audio files
- **LLM (Large Language Models)**: Support for local Ollama or cloud APIs (DeepSeek/OpenAI)
- **TTS (Text-to-Speech)**: High-quality speech synthesis based on Matcha-TTS, **supporting Chinese and English**
- **Streaming Processing**: LLM streaming output + real-time TTS playback for natural dialogue experience
- **Flexible Deployment**: ASR and LLM modules support local/cloud combinations, 4 deployment modes

### 🛠️ Technical Features
- **C++ High Performance Implementation**: Using ONNX Runtime for model inference
- **Modular Design**: ASR, LLM, TTS can be used independently or combined
- **TTS API Interface**: Easy-to-use C++ library supporting external project integration
- **Multi-threading Optimization**: Parallel processing to improve response speed
- **Ordered Audio Playback**: Ensuring TTS sentences play in sequence
- **Automatic Model Management**: Automatically download required models on first run, supporting language-specific model downloads

### 🎙️ Audio Processing
- **Multiple VAD Algorithms**: Support for Energy VAD and Silero VAD
- **Multi-device Support**: Support for various audio input devices
- **Automatic Resampling**: Support for automatic conversion of various sampling rates
- **FIR Anti-aliasing Filtering**: Built-in low-pass filtering before resampling to maintain high-frequency details
- **Real-time Audio Queue**: Ensuring continuity and sequence of audio playback
- **Long-range Recording Support**: VAD buffer amplification by 100x, significantly improving long-range voice detection capability
- **Stereo Sampling**: Support for stereo input with intelligent channel mixing to optimize audio quality
- **Speaker Recognition Access Control**: Optional speaker verification chain that automatically skips LLM/TTS phases if verification fails

### 🔐 Speaker Recognition (New Feature)
- **Speaker Recognition**: High-precision speaker recognition based on 3D-Speaker CamP+ model
- **Access Control**: Only registered users can use LLM and TTS functions
- **Multi-sample Registration**: Support for multiple audio samples to improve recognition accuracy
- **Real-time Verification**: Automatic speaker identity verification after ASR
- **Flexible Configuration**: Adjustable similarity thresholds and database paths

## System Requirements

### Basic Environment
- **Operating System**: Linux (Ubuntu 18.04+) / macOS
- **Compiler**: GCC-14 (recommended) or GCC 5+
- **CMake**: 3.16+

### System Dependencies
- **PortAudio 2.0**: Audio recording and playback
- **libsndfile**: Audio file processing
- **ONNX Runtime**: AI model inference
- **cURL**: Model downloading and cloud API communication
- **FFTW3**: Audio signal processing
- **OpenSSL**: Cloud ASR signature authentication (required for cloud mode)
- **Ollama**: Local LLM service (required for local LLM mode)

## Installation Guide

### 1. Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package manager
sudo apt update

# Install compilation tools
sudo apt install gcc-14 g++-14 cmake pkg-config

# Install audio and network libraries
sudo apt install libportaudio-dev libsndfile1-dev libcurl4-openssl-dev libfftw3-dev libssl-dev espeak-ng

# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-1.20.0.tgz
sudo cp -r onnxruntime-linux-x64-1.20.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.20.0/lib/* /usr/local/lib/
sudo ldconfig
```

#### macOS (Homebrew)
```bash
# Install dependencies
brew install gcc cmake pkg-config
brew install portaudio libsndfile curl fftw onnxruntime espeak openssl
```

### 2. Install Ollama (LLM Support)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl start ollama

# Download recommended models
ollama pull qwen2.5:0.5b  # Lightweight model
ollama pull qwen2.5       # Standard model (optional)
```

### 3. Build Project
```bash
# Clone repository
git clone https://github.com/muggle-stack/e2e_Voice.git
cd e2e_Voice

# Build (default: all local mode)
./build.sh

# Or use cloud services (optional)
# Cloud ASR mode
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON ..
make -j8

# Cloud LLM mode
cmake -DUSE_CLOUD_LLM=ON ..
make -j8

# Full cloud mode
cmake -DUSE_CLOUD_ASR=ON -DUSE_CLOUD_LLM=ON ..
make -j8
```

### 4. Cloud Service Configuration (Optional)

#### Aliyun Real-time ASR Configuration

If using cloud ASR (`-DUSE_CLOUD_ASR=ON`), you need to configure Aliyun credentials:

```bash
# 1. Copy environment configuration file
cp .env.example .env

# 2. Edit .env file and fill in real credentials
# ALIYUN_ACCESS_KEY_ID=LTAI***************
# ALIYUN_ACCESS_KEY_SECRET=******************************
# ALIYUN_ASR_APPKEY=tt43P2u****
```

**Get Credentials**:

1.  **Get AccessKey**:
    * Visit https://ram.console.aliyun.com/manage/ak
    * Click "Create AccessKey"
    * Save the AccessKey ID and AccessKey Secret

2.  **Get AppKey**:
    * Visit https://nls-portal.console.aliyun.com/
    * Create a project and select the "Short Sentence Recognition" service.
    * Copy the project's AppKey.

#### Cloud LLM API Configuration

If using cloud LLM (`-DUSE_CLOUD_LLM=ON`), you need to configure API keys:

```bash
# Edit .env file and add LLM configuration
# API_KEY=sk-***************************
# API_URL=https://api.deepseek.com/chat/completions
```

**Supported APIs**:
- DeepSeek: https://platform.deepseek.com/api_keys
- OpenAI: https://platform.openai.com/api-keys
- Other OpenAI-compatible services

## Usage Instructions

### 🎯 Complete Dialogue System (Recommended)

**Important**: `asr_llm_tts` now supports cloud ASR and cloud LLM integration! Flexible switching via compile-time options:
- **Default Mode**: Local ASR + Local Ollama
- **Cloud ASR Mode** (`-DUSE_CLOUD_ASR=ON`): Aliyun ASR + Local Ollama
- **Cloud LLM Mode** (`-DUSE_CLOUD_LLM=ON`): Local ASR + Cloud API (DeepSeek/OpenAI)
- **Full Cloud Mode** (`-DUSE_CLOUD_ASR=ON -DUSE_CLOUD_LLM=ON`): Aliyun ASR + Cloud API

> 💡 `asr_llm_tts_api` has been integrated into `asr_llm_tts` through conditional compilation for unified functionality.

#### Full Local Mode (Default)
```bash
# Build (default: all local)
./build.sh

# View help
./build/bin/asr_llm_tts --help

# Run with default parameters
./build/bin/asr_llm_tts

# Run with custom parameters
./build/bin/asr_llm_tts \
  --device_index 7 \
  --sample_rate 48000 \
  --channels 2 \
  --vad_type silero \
  --model qwen2.5:0.5b \
  --tts_speed 1.0 \
  --tts_type zh

# Enable speaker recognition (registration required first)
./build/bin/asr_llm_tts \
  --enable_speaker \
  --speaker_threshold 0.5 \
  --speaker_database speakers.db \
  --device_index 1

# Long-range recording optimization configuration
./build/bin/asr_llm_tts \
  --channels 2 \
  --sample_rate 16000 \
  --vad_type energy \
  --trigger_threshold 0.6
```

#### Cloud ASR Mode
```bash
# Build cloud ASR version
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON ..
make -j8
cd ..

# Configure Aliyun credentials (.env file)
cp .env.example .env
# Edit .env and fill in: ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ALIYUN_ASR_APPKEY

# Run (Cloud ASR + Local Ollama)
./build/bin/asr_llm_tts --model qwen2.5:0.5b
```

#### Cloud LLM Mode
```bash
# Build cloud LLM version
mkdir -p build && cd build
cmake -DUSE_CLOUD_LLM=ON ..
make -j8
cd ..

# Configure cloud LLM credentials (.env file)
# Edit .env and fill in: API_KEY, API_URL

# Run (Local ASR + Cloud API)
./build/bin/asr_llm_tts --model deepseek-chat --max_tokens 500
```

#### Full Cloud Mode
```bash
# Build full cloud version
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON -DUSE_CLOUD_LLM=ON ..
make -j8
cd ..

# Configure all cloud service credentials (.env file)
# Edit .env and fill in: Aliyun ASR + Cloud LLM configuration

# Run (Cloud ASR + Cloud API)
./build/bin/asr_llm_tts \
  --model deepseek-chat \
  --max_tokens 500 \
  --device_index 7 \
  --channels 2 \
  --tts_type zh
```

#### Standalone Cloud API Program (asr_llm_tts_api)

> **Note**: `asr_llm_tts_api` is a standalone cloud LLM-only program. Its functionality has been integrated into `asr_llm_tts` cloud LLM mode.
> We recommend using `asr_llm_tts` with compile options for more flexible deployment.

```bash
# Use DeepSeek API
./build/bin/asr_llm_tts_api \
  --api_key YOUR_DEEPSEEK_KEY \
  --api_url https://api.deepseek.com/chat/completions \
  --model deepseek-chat \
  --device_index 7 \
  --channels 2 \
  --vad_type silero \
  --tts_type zh

# Use OpenAI API
./build/bin/asr_llm_tts_api \
  --api_key YOUR_OPENAI_KEY \
  --api_url https://api.openai.com/v1/chat/completions \
  --model gpt-3.5-turbo \
  --max_tokens 500 \
  --tts_type en

# Load configuration from environment variables or .env file
export API_KEY=YOUR_KEY
export API_URL=https://api.deepseek.com/chat/completions
./build/bin/asr_llm_tts_api --model deepseek-chat
```

### 🔐 Speaker Recognition

#### Register Speaker
```bash
# Register speaker using audio files
./build/bin/register_speaker -n zhang_san sample1.wav sample2.wav sample3.wav

# Register via microphone recording (automatically records 3 times, 4 seconds each)
./build/bin/register_speaker -n zhang_san

# Use custom database
./build/bin/register_speaker -d company.db -n employee001 voice.wav

# Force overwrite existing speaker
./build/bin/register_speaker -f -n zhang_san new_sample.wav
```

#### View Registered Speakers
```bash
# List speakers in default database
./build/bin/list_speakers speakers.db

# List speakers in specified database
./build/bin/list_speakers company.db
```

### 🎙️ Speech Recognition

#### Cloud Real-time ASR Testing (New Feature)
```bash
# Use .env file configuration (recommended)
cp .env.example .env
# Edit .env with Aliyun credentials and run directly
./build/bin/asr_realtime_test

# Test audio file
./build/bin/asr_realtime_test --audio_file test.wav

# Custom recording parameters
./build/bin/asr_realtime_test \
  --device_index 0 \
  --sample_rate 16000 \
  --silence_duration 1.5 \
  --max_record_time 60 \
  --vad_type energy
```

**Features**:
- Support for automatic resampling of any sample rate to 16kHz
- 1-2 second low-latency recognition
- Support for local audio files and real-time recording

#### Streaming ASR (New Feature)
```bash
# Continuous speech recognition with automatic segmentation
./build/bin/streaming_asr \
  --device_index 7 \
  --channels 2 \
  --max_duration 300 \
  --silence_threshold 0.5 \
  --num_threads 4 \
  --vad_type silero

# Streaming recognition with energy VAD
./build/bin/streaming_asr \
  --vad_type energy \
  --vad_threshold 0.005 \
  --pre_speech_buffer 0.25
```

#### Traditional ASR Mode
```bash
# VAD+ASR (real-time microphone)
./build/bin/vad_asr --device-index 6 --vad-type silero

# ASR + LLM (no TTS)
./build/bin/asr_llm --model qwen2.5
```

### 🎵 Audio File Processing
```bash
# Process single audio file
./build/bin/asr audio_file.wav

# Batch process audio files
./build/bin/asr file1.wav file2.wav file3.wav

# Supported audio formats: WAV, FLAC, OGG
# Automatic resampling to 16kHz for recognition
```

### 🔊 Text-to-Speech (TTS)
```bash
# Chinese TTS basic usage
./build/bin/tts --text "Hello World" --tts_type zh

# English TTS basic usage
./build/bin/tts --text "Hello, World!" --tts_type en

# Save as WAV file
./build/bin/tts --text "Welcome to the TTS system" --tts_type zh --save_audio_path output.wav

# English speech synthesis
./build/bin/tts --text "Welcome to the TTS system" --tts_type en --save_audio_path english_output.wav

# Adjust speech rate and speaker
./build/bin/tts --text "This is a test" --tts_type zh --tts_speed 1.2 --tts_speaker_id 0 --save_audio_path slow.wav

# View help
./build/bin/tts --help
```

#### TTS API Usage (for C++ project integration)
```cpp
#include "tts_demo.hpp"

// Chinese TTS configuration
TTSDemo::Params zh_params;
zh_params.tts_speed = 1.0f;
zh_params.tts_speaker_id = 0;
zh_params.tts_type = "zh";

// English TTS configuration
TTSDemo::Params en_params;
en_params.tts_speed = 1.0f;
en_params.tts_speaker_id = 0;
en_params.tts_type = "en";

// Create Chinese TTS instance
TTSDemo zh_tts(zh_params);
if (zh_tts.initialize()) {
    zh_tts.run("Hello World", "chinese_output.wav");
}

// Create English TTS instance
TTSDemo en_tts(en_params);
if (en_tts.initialize()) {
    en_tts.run("Hello World", "english_output.wav");
}
```

### 📱 Find Audio Devices
```bash
# Python audio device search
python search_device.py
```

## Configuration Parameters

### ASR-LLM-TTS Complete System
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--sample_rate` | Audio sampling rate | 16000 | 48000 |
| `--device_index` | Audio device index | 6 | 7 |
| `--channels` | Number of audio channels (1=mono, 2=stereo) | 1 | 2 |
| `--vad_type` | VAD type | energy | silero |
| `--model` | LLM model name | qwen2.5:0.5b | qwen2.5 |
| `--tts_speed` | TTS speech speed | 1.0 | 0.8 |
| `--tts_speaker` | TTS speaker ID | 0 | 0 |
| `--tts_type` | TTS language type | zh | en |

### Cloud API System (asr_llm_tts_api)
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--api_key` | API key | - | sk-xxx |
| `--api_url` | API endpoint URL | - | https://api.deepseek.com/chat/completions |
| `--model` | Model name | deepseek-chat | gpt-3.5-turbo |
| `--max_tokens` | Maximum generated tokens | 500 | 1000 |
| `--channels` | Number of audio channels (1=mono, 2=stereo) | 1 | 2 |
| `--tts_type` | TTS language type | zh | en |
| `--env_file` | Environment configuration file path | .env | config.env |

### Streaming ASR System (streaming_asr)
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--max_duration` | Maximum recording duration (seconds) | 60 | 300 |
| `--silence_threshold` | Silence segmentation threshold (seconds) | 0.5 | 1.0 |
| `--pre_speech_buffer` | Pre-speech buffer (seconds) | 0.25 | 0.5 |
| `--num_threads` | ASR processing threads | 2 | 4 |
| `--channels` | Number of audio channels (1=mono, 2=stereo) | 1 | 2 |
| `--vad_threshold` | Energy VAD threshold | 0.005 | 0.01 |

### TTS Standalone Tool Parameters
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--text` | Text to convert | - | "Hello World" |
| `--save_audio_path` | Audio file save path | - | "output.wav" |
| `--tts_speed` | TTS speech speed | 1.0 | 1.2 |
| `--tts_speaker_id` | TTS speaker ID | 0 | 0 |
| `--tts_type` | TTS language type | zh | en |

### Audio Recording Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--silence_duration` | Silence duration threshold (seconds) | 1.0 |
| `--max_record_time` | Maximum recording time (seconds) | 5.0 |
| `--trigger_threshold` | VAD trigger threshold | 0.6 |
| `--stop_threshold` | VAD stop threshold | 0.35 |

## Project Architecture

### Core Modules
```
src/
├── main_asr_llm_tts.cpp    # Complete dialogue system main program (Ollama)
├── main_asr_llm_tts_api.cpp # Cloud API dialogue system (DeepSeek/OpenAI)
├── main_streaming_asr.cpp  # Streaming ASR main program
├── main_ase.cpp            # Audio file processing engine (asr)
├── main_llm.cpp            # ASR+LLM system
├── main_asr.cpp            # VAD+ASR real-time system (vad_asr)
├── main_tts.cpp            # TTS standalone tool main program
├── tts_demo.cpp            # TTS API implementation (available externally)
├── audio_recorder.cpp      # Audio recording module
├── streaming_audio_recorder.cpp # Streaming audio recorder
├── api_comm.cpp            # Cloud API communication module
├── vad_detector.cpp        # Voice activity detection
├── asr_model.cpp           # Speech recognition model
├── asr_thread_pool.cpp     # ASR multi-thread pool
├── text_buffer.cpp         # Streaming text buffer
├── ordered_audio_queue.cpp # Ordered audio playback queue
└── tts/
    ├── tts_model.cpp           # TTS model implementation
    └── tts_model_downloader.cpp # TTS model downloader

include/
├── tts_demo.hpp            # TTS API header file (external interface)
├── streaming_audio_recorder.hpp # Streaming recorder interface
├── api_comm.hpp            # API communication interface
├── asr_thread_pool.hpp     # ASR thread pool interface
└── ...
```

### Workflow
```
User voice → ASR recognition → LLM streaming generation → Sentence segmentation → TTS synthesis → Ordered playback
   ↓           ↓          ↓            ↓         ↓         ↓
Recording buffer → Feature extraction → Streaming output → TextBuffer → Multi-thread TTS → AudioQueue
```

## Model Description

### ASR Model (SenseVoice)
- **Model Path**: `~/.cache/sensevoice/`
- **Main Files**:
  - `model_quant_optimized.onnx` - Quantized ASR model
  - `config.json` - Model configuration
  - `vocab.txt` - Vocabulary
  - `decoder_path` - Decoder path

### VAD Model (Silero)
- **Model File**: `silero_vad.onnx`
- **Function**: Voice activity detection, improving recognition accuracy

### TTS Model (Matcha)
- **Model Path**: `~/.cache/matcha-tts/`
- **Chinese TTS Model**: `matcha-icefall-zh-baker/`
  - `model-steps=6.onnx` - Chinese acoustic model
  - `lexicon.txt` - Chinese pronunciation dictionary
  - `tokens.txt` - Chinese phoneme markers
  - `dict/` - Chinese dictionary directory
- **English TTS Model**: `matcha-icefall-en_US-ljspeech/`
  - `model-steps=6.onnx` - English acoustic model
  - `tokens.txt` - English phoneme markers
  - `espeak-ng-data/` - English pronunciation data
- **Shared Vocoder**: `vocos-22khz-univ.onnx` - Universal vocoder model
- **Automatic Download**: Downloads corresponding models based on language type on first use

## Technical Highlights

### 🚀 Real-time Streaming Processing
- **LLM Streaming Output**: Display while generating to reduce perceived latency
- **Sentence Segmentation**: Intelligent sentence splitting based on Chinese/English punctuation
- **Parallel TTS**: Multi-threaded generation to improve efficiency

### 🎵 Ordered Audio Playback
- **Sequence Guarantee**: Strict sentence order playback regardless of TTS generation speed
- **Queue Management**: `OrderedAudioQueue` ensures continuous audio playback
- **Memory Optimization**: Timely release of played audio to save memory

### 🔧 Multi-language TTS Optimization
- **Chinese TTS**:
  - **Jieba Word Segmentation**: Precise Chinese text word segmentation
  - **Phoneme Mapping**: Complete pinyin to phoneme conversion
  - **ISTFT Post-processing**: High-quality time-domain audio reconstruction from frequency domain
- **English TTS**:
  - **espeak-ng Integration**: IPA phoneme-based English pronunciation
  - **Phoneme Filtering**: Remove problematic zero-width connector characters
  - **Language-specific Optimization**: Avoid double smoothing to maintain natural sound quality
- **Shared Vocoder**: High-quality Vocos model supporting multi-language audio generation

### 🎙️ Long-range Recording Enhancement
- **VAD Buffer 100x Amplification**: Significantly improving long-range voice detection capability
- **Adaptive Amplification Algorithm**: Dynamically adjust amplification factor based on signal strength (50x-300x)
- **Dual Protection Mechanism**: Prevent memory out-of-bounds and data loss
- **Circular Buffer Optimization**: Efficient processing of long-duration recording data

### 📡 Stereo Sampling + Speaker Recognition Chain
- **Intelligent Channel Mixing**: Left and right channel average fusion to optimize audio quality
- **Stereo Compatibility**: Automatic detection and processing of mono/stereo input
- **FIR Anti-aliasing Filtering**: Automatic low-pass filtering before resampling to avoid high-frequency folding
- **Speaker Recognition Raw Audio**: Unclipped raw waveforms fed independently into speaker recognition chain for improved accuracy
- **Safety Boundary Checks**: Multi-layer protection against array out-of-bounds access
- **Real-time Performance Optimization**: Minimize processing delay while maintaining real-time capability

## FAQ

### 1. Audio Related
**Q: How to find the correct audio device?**
```bash
python search_device.py
# Select device index with input channels
```

**Q: No sound during recording?**
A: Check microphone permissions, device index, sampling rate settings

**Q: How to enable long-range recording function?**
A: Long-range recording is automatically enabled with VAD buffer 100x amplification algorithm. Suitable for conference rooms and classrooms.

**Q: What are the advantages of stereo recording?**
A: Stereo recording can:
- Provide better noise suppression effects
- Improve long-range voice detection accuracy
- Enhance overall audio quality
- Automatically mix left and right channels for optimization

**Q: How to switch between mono/stereo?**
```bash
# Use stereo
./build/bin/asr_llm_tts --channels 2 --device_index 7

# Use mono (default)
./build/bin/asr_llm_tts --channels 1 --device_index 7
```

### 2. LLM Related
**Q: LLM connection failed?**
```bash
# Check Ollama service status
sudo systemctl status ollama

# Restart Ollama service
sudo systemctl restart ollama
```

**Q: Model download failed?**
```bash
# Manually download model
ollama pull qwen2.5:0.5b
```

### 3. TTS Related
**Q: TTS has no sound or poor quality?**
A:
- Check if TTS models are downloaded correctly
- Confirm correct language type selection (`--tts_type zh` or `--tts_type en`)
- Adjust TTS speech rate parameters
- Confirm audio output device is working properly

**Q: English TTS pronunciation inaccurate?**
A:
- Ensure espeak-ng is installed: `sudo apt install espeak-ng` (Ubuntu) or `brew install espeak` (macOS)
- English TTS relies on espeak-ng for phoneme conversion

**Q: Playback order messed up?**
A: The project uses `OrderedAudioQueue` to solve this problem

**Q: How to switch between Chinese/English TTS?**
```bash
# Use Chinese TTS
./build/bin/asr_llm_tts --tts_type zh

# Use English TTS
./build/bin/asr_llm_tts --tts_type en
```

### 4. Performance Optimization
**Q: Response too slow?**
A:
- Use lightweight LLM model (`qwen2.5:0.5b`)
- Choose energy VAD instead of Silero VAD
- Adjust sampling rate to 16kHz
- Run on high-performance devices

## TTS API Integration Guide

### Using TTS API in Your C++ Project

#### 1. Copy Required Files to Your Project
```bash
# Copy header files
cp include/tts_demo.hpp your_project/include/

# Copy source files
cp src/tts_demo.cpp your_project/src/
cp src/tts/tts_model.cpp your_project/src/
cp src/tts/tts_model_downloader.cpp your_project/src/
```

#### 2. Modify Your CMakeLists.txt
```cmake
# Add TTS source files
add_executable(your_app
    your_main.cpp
    src/tts_demo.cpp
    src/tts_model.cpp
    src/tts_model_downloader.cpp
)

# Link required libraries
target_link_libraries(your_app
    onnxruntime
    sndfile
    curl
    pthread
)

# Add header file paths
target_include_directories(your_app PRIVATE include)
```

#### 3. Basic API Usage Example
See `TTS_API_USAGE.md` documentation for detailed examples.

```cpp
#include "tts_demo.hpp"

int main() {
    // Configuration parameters
    TTSDemo::Params params;
    params.tts_speed = 1.0f;        // Normal speech rate
    params.tts_speaker_id = 0;      // Default speaker
    params.tts_type = "zh";         // Chinese TTS, options "zh" or "en"

    // Create TTS instance
    TTSDemo tts(params);

    // Initialize (automatically downloads models on first run)
    if (!tts.initialize()) {
        return -1;  // Initialization failed
    }

    // Generate speech
    tts.run("Hello, welcome to the TTS system!", "greeting.wav");

    return 0;
}
```

## Development Guide

### Adding New TTS Models
1. Inherit the `TTSModel` base class
2. Implement model loading and inference interfaces
3. Add download logic in `TTSModelDownloader`

### Integrating New LLMs
1. Extend `ollama.hpp` interface
2. Implement streaming generation callbacks
3. Adapt `TextBuffer` sentence segmentation logic

### Custom Audio Processing
1. Modify `AudioRecorder` recording parameters
2. Adjust `VADDetector` detection algorithm
3. Optimize `OrderedAudioQueue` playback strategy

## License Agreement

### Open Source License
This project uses open source license.

### Usage Scope

**Free use allowed:**
- Personal learning and research
- Academic research and educational purposes
- Technical evaluation and testing
- Open source project integration

**Commercial use requires license:**
- Integration into commercial products/services
- Development of commercial applications
- Sale of products containing this software
- Provision of commercial speech recognition services
- Production environment use by profit-making organizations

### Commercial License Application

For commercial use license, please contact the project maintainer.

**Email**: [promuggle@gmail.com]  
**GitHub**: [muggle-stack]

### Important Reminder

**Individual Developers**: Can use freely, but please keep copyright notices  
**Enterprise Users**: It is recommended to contact me for formal license before use  
**Open Source Contributions**: Welcome PR and Issue submissions to improve the project


## Contribution

Welcome to submit Issues and Pull Requests!

## Changelog

### v3.0.0 (Current Version)
- ✅ **Cloud ASR Integration**: Support Aliyun real-time ASR with 1-2 second low-latency recognition
- ✅ **Cloud LLM Integration**: Support cloud APIs (DeepSeek/OpenAI) as LLM backend
- ✅ **Flexible Modular Architecture**: ASR and LLM support independent local/cloud switching, 4 deployment modes
- ✅ **Conditional Compilation**: Control compilation via CMake options (`USE_CLOUD_ASR`, `USE_CLOUD_LLM`)
- ✅ **.env Configuration Support**: Unified configuration management, support reading cloud service credentials from .env file
- ✅ **Automatic Resampling**: ASR API layer automatically handles audio of any sample rate
- ✅ **Cloud Testing Tool**: Added `asr_realtime_test` for cloud ASR functionality testing

**Deployment Mode Comparison**:
- All Local: Local ASR + Ollama (offline available)
- Cloud ASR: Cloud ASR + Ollama (high accuracy ASR)
- Cloud LLM: Local ASR + Cloud API (powerful LLM capability)
- All Cloud: Cloud ASR + Cloud API (highest accuracy and capability)

### v2.6.0
- ✅ **Speaker Recognition Integration**: Added speaker recognition functionality based on 3D-Speaker CamP+ model
- ✅ **Access Control Mechanism**: Only registered users can use LLM and TTS functions
- ✅ **Raw Audio Path**: Preserve unclipped raw waveform dedicated to speaker embedding inference
- ✅ **FIR Anti-aliasing Resampling**: Built-in FIR low-pass filter to solve frequency aliasing during downsampling
- ✅ **Mono/Stereo Automatic Processing**: Intelligent detection and mixing of mono/stereo input, automatic conversion to 16kHz mono format

### v2.5.0
- ✅ **English TTS Support**: Added English speech synthesis functionality based on Matcha-TTS
- ✅ **Multi-language Model Management**: Language-specific model downloads, obtain Chinese/English TTS models on demand
- ✅ **espeak-ng Integration**: Support IPA phoneme-based English text-to-speech conversion
- ✅ **Intelligent Language Switching**: Easy switching between Chinese and English TTS via `--tts_type` parameter
- ✅ **Audio Quality Optimization**: Language-specific audio post-processing to eliminate English TTS popping issues

### v2.4.0
- ✅ **Long-range Recording Enhancement**: VAD buffer amplification by 100x, supporting ultra-long distance voice detection
- ✅ **Stereo Sampling Processing**: Intelligent left/right channel mixing, significantly improving audio quality
- ✅ **Adaptive Amplification Algorithm**: Dynamically adjust amplification factor based on signal strength (50x-300x)
- ✅ **Safety Boundary Checks**: Multi-layer protection against memory out-of-bounds and data loss
- ✅ **Circular Buffer Optimization**: Efficient processing of long-duration recording data, optimizing memory usage

### v2.3.0
- ✅ Added streaming ASR functionality (`streaming_asr`), supporting real-time continuous speech recognition
- ✅ Added cloud LLM API interface (`asr_llm_tts_api`), supporting DeepSeek, OpenAI and other APIs
- ✅ Implemented circular buffer and sliding window VAD, improving real-time performance
- ✅ Support multi-thread ASR processing pool, parallel processing of audio segments
- ✅ Added universal API interface layer, unified support for multiple LLM service providers

### v2.2.0
- ✅ Added standalone TTS tool (`tts`), supporting command-line text-to-speech
- ✅ Provided TTSDemo C++ API interface, supporting external project integration
- ✅ Restructured TTS module architecture, separating headers and implementations
- ✅ Output standard WAV format audio files, compatible with various players
- ✅ Added detailed API usage documentation and example code

### v2.1.0
- ✅ Restructured executable naming: asr (file processing), vad_asr (real-time microphone)
- ✅ Added audio file batch processing engine
- ✅ Support batch speech recognition for audio files
- ✅ Automatic audio format conversion and resampling functionality
- ✅ Optimized modular architecture design

### v2.0.0
- ✅ Integrated complete ASR-LLM-TTS dialogue system
- ✅ Implemented streaming LLM output and real-time TTS playback
- ✅ Added ordered audio playback queue
- ✅ Support Matcha-TTS Chinese speech synthesis
- ✅ Optimized multi-thread performance and memory usage

### v1.0.0
- ✅ Initial ASR+LLM system
- ✅ SenseVoice speech recognition
- ✅ Ollama LLM integration
- ✅ Multi-platform support
