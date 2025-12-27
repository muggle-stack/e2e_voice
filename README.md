# E2E Voice - C++ 端到端语音对话系统

一个完整的中文智能语音对话系统，集成 ASR（语音识别）、LLM（大语言模型）和 TTS（语音合成）。

## 特性

- **ASR**：本地 SenseVoice 或阿里云实时 ASR
- **LLM**：本地 Ollama 或云端 API（DeepSeek/OpenAI）
- **TTS**：Matcha-TTS，支持中文、英文、中英混合
- **声纹识别**：基于 3D-Speaker 的说话人验证
- **流式处理**：LLM 流式输出 + 实时 TTS 播放

## 快速开始

### 安装依赖

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

**安装 Ollama（本地 LLM）:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:0.5b
```

### 构建

```bash
git clone https://github.com/muggle-stack/e2e_Voice.git
cd e2e_Voice
./build.sh
```

**云端模式（可选）:**
```bash
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON ..    # 阿里云 ASR
cmake -DUSE_CLOUD_LLM=ON ..    # 云端 LLM API
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### 运行

```bash
# 完整对话系统
./build/bin/asr_llm_tts --help
./build/bin/asr_llm_tts --device_index 1 --vad_type silero

# 语音识别（文件）
./build/bin/asr audio.wav

# 语音合成
./build/bin/tts --text "你好世界" --tts_type zh
./build/bin/tts --text "Hello" --tts_type en
./build/bin/tts --text "今天学Python" --tts_type zh-en

# 查找音频设备
python search_device.py
```

### 声纹识别

```bash
# 注册说话人
./build/bin/register_speaker -n 张三 sample1.wav sample2.wav

# 启用声纹验证
./build/bin/asr_llm_tts --enable_speaker --speaker_threshold 0.5
```

## 云端配置

### 阿里云 ASR 配置

1. **获取 AccessKey**
   - 访问 https://ram.console.aliyun.com/manage/ak
   - 点击"创建 AccessKey"
   - 保存 AccessKey ID 和 AccessKey Secret

2. **获取 AppKey**
   - 访问 https://nls-portal.console.aliyun.com/
   - 创建项目，选择"一句话识别"服务
   - 复制项目的 AppKey

3. **配置环境变量**

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 阿里云 ASR
ALIYUN_ACCESS_KEY_ID=LTAI***************
ALIYUN_ACCESS_KEY_SECRET=******************************
ALIYUN_ASR_APPKEY=tt43P2u****

# 云端 LLM（DeepSeek/OpenAI）
API_KEY=sk-xxx
API_URL=https://api.deepseek.com/chat/completions
```

## 模型

首次运行时自动下载到 `~/.cache/`：
- ASR: `sensevoice/`
- TTS: `matcha-tts/`
- VAD: `silero_vad.onnx`

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 致谢

- [ollama-hpp](https://github.com/jmont-dev/ollama-hpp) - C++ Ollama 客户端
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - 高性能推理引擎
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 语音识别模型
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - 语音合成模型
- [cppjieba](https://github.com/yanyiwu/cppjieba) - C++ 中文分词库
- [cpp-pinyin](https://github.com/wolfgitpr/cpp-pinyin) - 中文转拼音库
- [dengcunqin](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010) - 中英文 Matcha 模型
