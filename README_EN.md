# E2E Voice - C++ End-to-End Voice Dialogue System

A complete Chinese intelligent voice dialogue system integrating ASR (Speech Recognition), LLM (Large Language Model), and TTS (Text-to-Speech).

## Features

- **ASR**: Local SenseVoice or Aliyun real-time ASR
- **LLM**: Local Ollama or cloud APIs (DeepSeek/OpenAI)
- **TTS**: Matcha-TTS with Chinese, English, and bilingual support
- **Speaker Recognition**: 3D-Speaker based speaker verification
- **Streaming**: LLM streaming output + real-time TTS playback

## Quick Start

### Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install gcc g++ cmake pkg-config
sudo apt install libportaudio-dev libsndfile1-dev libcurl4-openssl-dev libfftw3-dev libssl-dev espeak-ng

# ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-1.20.0.tgz
sudo cp -r onnxruntime-linux-x64-1.20.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.20.0/lib/* /usr/local/lib/
sudo ldconfig
```

**macOS:**
```bash
brew install gcc cmake pkg-config portaudio libsndfile curl fftw onnxruntime espeak openssl
```

**Install Ollama (Local LLM):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:0.5b
```

### Build

```bash
git clone https://github.com/muggle-stack/e2e_Voice.git
cd e2e_Voice
./build.sh
```

**Cloud mode (optional):**
```bash
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON ..    # Aliyun ASR
cmake -DUSE_CLOUD_LLM=ON ..    # Cloud LLM API
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### Run

```bash
# Full dialogue system
./build/bin/asr_llm_tts --help
./build/bin/asr_llm_tts --device_index 1 --vad_type silero

# Speech recognition (file)
./build/bin/asr audio.wav

# Text-to-speech
./build/bin/tts --text "你好世界" --tts_type zh
./build/bin/tts --text "Hello" --tts_type en
./build/bin/tts --text "今天学Python" --tts_type zh-en

# Find audio devices
python search_device.py
```

### Speaker Recognition

```bash
# Register speaker
./build/bin/register_speaker -n john sample1.wav sample2.wav

# Enable speaker verification
./build/bin/asr_llm_tts --enable_speaker --speaker_threshold 0.5
```

## Cloud Configuration

### Aliyun ASR Configuration

1. **Get AccessKey**
   - Visit https://ram.console.aliyun.com/manage/ak
   - Click "Create AccessKey"
   - Save the AccessKey ID and AccessKey Secret

2. **Get AppKey**
   - Visit https://nls-portal.console.aliyun.com/
   - Create a project, select "Short Sentence Recognition" service
   - Copy the project's AppKey

3. **Configure Environment Variables**

```bash
cp .env.example .env
```

Edit `.env` file:

```bash
# Aliyun ASR
ALIYUN_ACCESS_KEY_ID=LTAI***************
ALIYUN_ACCESS_KEY_SECRET=******************************
ALIYUN_ASR_APPKEY=tt43P2u****

# Cloud LLM (DeepSeek/OpenAI)
API_KEY=sk-xxx
API_URL=https://api.deepseek.com/chat/completions
```

## Models

Auto-downloaded on first run to `~/.cache/`:
- ASR: `sensevoice/`
- TTS: `matcha-tts/`
- VAD: `silero_vad.onnx`

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgements

- [ollama-hpp](https://github.com/jmont-dev/ollama-hpp) - C++ Ollama Client
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - High-performance Inference Engine
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - Speech Recognition Model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Text-to-Speech Model
- [cppjieba](https://github.com/yanyiwu/cppjieba) - C++ Chinese Word Segmentation Library
- [cpp-pinyin](https://github.com/wolfgitpr/cpp-pinyin) - Chinese to Pinyin Library
- [dengcunqin](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010) - Chinese-English Matcha model
