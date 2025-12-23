# End_to_End C++ 智能语音对话系统

一个完整的中文智能语音对话系统，集成了自动语音识别(ASR)、大语言模型(LLM)和文本转语音(TTS)功能。支持实时语音交互，具备远场录音增强和双声道采样处理能力，适用于会议室、教室等大空间环境。

## 🎯 项目特性

### 🔊 完整的语音对话链路
- **ASR (语音识别)**：支持本地SenseVoice模型或云端阿里云实时ASR
- **音频文件处理**：专用于音频文件批量处理的轻量级引擎
- **LLM (大语言模型)**：支持本地Ollama或云端API（DeepSeek/OpenAI）
- **TTS (文本转语音)**：基于Matcha-TTS的高质量语音合成，**支持中文和英文**
- **流式处理**：LLM流式输出+实时TTS播放，自然对话体验
- **灵活部署**：ASR和LLM模块支持本地/云端自由组合，4种部署模式

### 🛠️ 技术特性
- **C++高性能实现**：使用ONNX Runtime进行模型推理
- **模块化设计**：ASR、LLM、TTS可独立使用或组合
- **TTS API接口**：提供易用的C++类库，支持外部项目集成
- **多线程优化**：并行处理提升响应速度
- **有序音频播放**：确保TTS按句子顺序播放
- **自动模型管理**：首次运行自动下载所需模型，支持语言特定模型下载

### 🎙️ 音频处理
- **多种VAD算法**：支持能量VAD和Silero VAD
- **多设备支持**：支持各种音频输入设备
- **自动重采样**：支持多种采样率自动转换
- **FIR抗混叠滤波**：重采样前内置低通滤波，保持高频细节稳定
- **实时音频队列**：保证音频播放的连续性和顺序性
- **远场录音支持**：VAD buffer放大100倍，显著提升远场语音检测能力
- **双声道采样**：支持立体声输入，智能通道混合优化音频质量
- **声纹访问控制**：可选声纹验证链路，未通过验证自动跳过 LLM/TTS 阶段

### 🔐 说话人识别（新功能）
- **声纹识别**：基于3D-Speaker CamP+模型的高精度声纹识别
- **访问控制**：只有注册用户才能使用LLM和TTS功能
- **多样本注册**：支持多个音频样本提升识别准确率
- **实时验证**：在ASR后自动进行说话人身份验证
- **灵活配置**：可调节识别阈值和数据库路径

## 系统要求

### 基础环境
- **操作系统**：Linux (Ubuntu 18.04+) / macOS
- **编译器**：GCC-14 (推荐) 或 GCC 5+
- **CMake**：3.16+

### 系统依赖
- **PortAudio 2.0**：音频录制和播放
- **libsndfile**：音频文件处理
- **ONNX Runtime**：AI模型推理
- **cURL**：模型下载和云端API通信
- **FFTW3**：音频信号处理
- **OpenSSL**：云端ASR签名认证（云端模式需要）
- **Ollama**：本地LLM服务（本地LLM模式需要）

## 安装指南

### 1. 安装系统依赖

#### Ubuntu/Debian
```bash
# 更新包管理器
sudo apt update

# 安装编译工具
# Ubuntu 22.04实测 gcc-14 和 g++-14 已取消，用默认 g++ 即可
sudo apt install gcc-14 g++-14 cmake pkg-config

# 安装音频和网络库
# Ubuntu 22.04 实测 libportaudio-dev 包名改为 portaudio19-dev
sudo apt install libportaudio-dev libsndfile1-dev libcurl4-openssl-dev libfftw3-dev libssl-dev espeak-ng

# 安装ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-1.20.0.tgz
sudo cp -r onnxruntime-linux-x64-1.20.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.20.0/lib/* /usr/local/lib/
sudo ldconfig
```

#### macOS (Homebrew)
```bash
# 安装依赖
brew install gcc cmake pkg-config
brew install portaudio libsndfile curl fftw onnxruntime espeak openssl
```

### 2. 安装Ollama (LLM支持)
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动Ollama服务
sudo systemctl start ollama

# 下载推荐模型
ollama pull qwen2.5:0.5b  # 轻量级模型
ollama pull qwen2.5       # 标准模型（可选）
```

### 3. 构建项目
```bash
# 克隆项目
git clone https://github.com/muggle-stack/e2e_Voice.git
cd e2e_Voice

# 构建（默认全本地模式）
./build.sh

# 或者使用云端服务（可选）
# 云端ASR模式
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON ..
make -j8

# 云端LLM模式
cmake -DUSE_CLOUD_LLM=ON ..
make -j8

# 全云端模式
cmake -DUSE_CLOUD_ASR=ON -DUSE_CLOUD_LLM=ON ..
make -j8
```

### 4. 云端服务配置（可选）

#### 阿里云实时ASR配置

如果使用云端ASR（`-DUSE_CLOUD_ASR=ON`），需要配置阿里云密钥：

```bash
# 1. 复制环境配置文件
cp .env.example .env

# 2. 编辑 .env 文件，填入真实密钥
# ALIYUN_ACCESS_KEY_ID=LTAI***************
# ALIYUN_ACCESS_KEY_SECRET=******************************
# ALIYUN_ASR_APPKEY=tt43P2u****
```

**获取密钥**：

1. **获取 AccessKey**：
   - 访问 https://ram.console.aliyun.com/manage/ak
   - 点击"创建AccessKey"
   - 保存 AccessKey ID 和 AccessKey Secret

2. **获取 AppKey**：
   - 访问 https://nls-portal.console.aliyun.com/
   - 创建项目，选择"一句话识别"服务
   - 复制项目的 AppKey

#### 云端LLM API配置

如果使用云端LLM（`-DUSE_CLOUD_LLM=ON`），需要配置API密钥：

```bash
# 编辑 .env 文件，添加LLM配置
# API_KEY=sk-***************************
# API_URL=https://api.deepseek.com/chat/completions
```

**支持的API**：
- DeepSeek: https://platform.deepseek.com/api_keys
- OpenAI: https://platform.openai.com/api-keys
- 其他兼容OpenAI API的服务

## 使用说明

### 🎯 完整对话系统 (推荐)

**重要说明**：`asr_llm_tts` 现已支持云端ASR和云端LLM集成！通过编译时选项可灵活切换：
- **默认模式**：本地ASR + 本地Ollama
- **云端ASR模式** (`-DUSE_CLOUD_ASR=ON`)：阿里云ASR + 本地Ollama
- **云端LLM模式** (`-DUSE_CLOUD_LLM=ON`)：本地ASR + 云端API（DeepSeek/OpenAI）
- **全云端模式** (`-DUSE_CLOUD_ASR=ON -DUSE_CLOUD_LLM=ON`)：阿里云ASR + 云端API

> 💡 `asr_llm_tts_api` 已被集成到 `asr_llm_tts` 中，通过条件编译实现功能统一。

#### 全本地模式（默认）
```bash
# 编译（默认全本地）
./build.sh

# 查看帮助
./build/bin/asr_llm_tts --help

# 使用默认参数运行
./build/bin/asr_llm_tts

# 自定义参数运行
./build/bin/asr_llm_tts \
  --device_index 7 \
  --sample_rate 48000 \
  --channels 2 \
  --vad_type silero \
  --model qwen2.5:0.5b \
  --tts_speed 1.0 \
  --tts_type zh

# 启用说话人识别（需要先注册）
./build/bin/asr_llm_tts \
  --enable_speaker \
  --speaker_threshold 0.5 \
  --speaker_database speakers.db \
  --device_index 1

# 远场录音优化配置
./build/bin/asr_llm_tts \
  --channels 2 \
  --sample_rate 16000 \
  --vad_type energy \
  --trigger_threshold 0.6
```

#### 云端ASR模式
```bash
# 编译云端ASR版本
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON ..
make -j8
cd ..

# 配置阿里云密钥（.env文件）
cp .env.example .env
# 编辑.env填入：ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ALIYUN_ASR_APPKEY

# 运行（云端ASR + 本地Ollama）
./build/bin/asr_llm_tts --model qwen2.5:0.5b
```

#### 云端LLM模式
```bash
# 编译云端LLM版本
mkdir -p build && cd build
cmake -DUSE_CLOUD_LLM=ON ..
make -j8
cd ..

# 配置云端LLM密钥（.env文件）
# 编辑.env填入：API_KEY, API_URL

# 运行（本地ASR + 云端API）
./build/bin/asr_llm_tts --model deepseek-chat --max_tokens 500
```

#### 全云端模式
```bash
# 编译全云端版本
mkdir -p build && cd build
cmake -DUSE_CLOUD_ASR=ON -DUSE_CLOUD_LLM=ON ..
make -j8
cd ..

# 配置所有云端服务密钥（.env文件）
# 编辑.env填入：阿里云ASR + 云端LLM配置

# 运行（云端ASR + 云端API）
./build/bin/asr_llm_tts \
  --model deepseek-chat \
  --max_tokens 500 \
  --device_index 7 \
  --channels 2 \
  --tts_type zh
```

#### 独立云端API程序 (asr_llm_tts_api)

> **注意**：`asr_llm_tts_api` 是独立的云端LLM专用程序，功能已被集成到 `asr_llm_tts` 的云端LLM模式中。
> 推荐使用 `asr_llm_tts` 配合编译选项，获得更灵活的部署方式。

```bash
# 使用DeepSeek API
./build/bin/asr_llm_tts_api \
  --api_key YOUR_DEEPSEEK_KEY \
  --api_url https://api.deepseek.com/chat/completions \
  --model deepseek-chat \
  --device_index 7 \
  --channels 2 \
  --vad_type silero \
  --tts_type zh

# 使用OpenAI API
./build/bin/asr_llm_tts_api \
  --api_key YOUR_OPENAI_KEY \
  --api_url https://api.openai.com/v1/chat/completions \
  --model gpt-3.5-turbo \
  --max_tokens 500 \
  --tts_type en

# 从环境变量或.env文件加载配置
export API_KEY=YOUR_KEY
export API_URL=https://api.deepseek.com/chat/completions
./build/bin/asr_llm_tts_api --model deepseek-chat
```

### 🔐 说话人识别

#### 注册说话人
```bash
# 使用音频文件注册说话人
./build/bin/register_speaker -n zhang_san sample1.wav sample2.wav sample3.wav

# 通过麦克风录音注册（会录制3次，每次4秒）
./build/bin/register_speaker -n zhang_san

# 使用自定义数据库
./build/bin/register_speaker -d company.db -n employee001 voice.wav

# 强制覆盖已存在的说话人
./build/bin/register_speaker -f -n zhang_san new_sample.wav
```

#### 查看已注册说话人
```bash
# 列出默认数据库中的说话人
./build/bin/list_speakers speakers.db

# 列出指定数据库中的说话人
./build/bin/list_speakers company.db
```

### 🎙️ 语音识别

#### 云端实时ASR测试 (新功能)
```bash
# 使用.env文件配置（推荐）
cp .env.example .env
# 编辑 .env 填入阿里云密钥后直接运行
./build/bin/asr_realtime_test

# 测试音频文件
./build/bin/asr_realtime_test --audio_file test.wav

# 自定义录音参数
./build/bin/asr_realtime_test \
  --device_index 0 \
  --sample_rate 16000 \
  --silence_duration 1.5 \
  --max_record_time 60 \
  --vad_type silero
```

**特性**：
- 支持任意采样率自动重采样到16kHz
- 1-2秒低延迟识别
- 支持本地音频文件和实时录音

#### 实时流式ASR (新功能)
```bash
# 连续语音识别，自动分段
./build/bin/streaming_asr \
  --device_index 7 \
  --channels 2 \
  --max_duration 300 \
  --silence_threshold 0.5 \
  --num_threads 4 \
  --vad_type silero

# 使用能量VAD的流式识别
./build/bin/streaming_asr \
  --vad_type energy \
  --vad_threshold 0.005 \
  --pre_speech_buffer 0.25
```

#### 传统ASR模式
```bash
# VAD+ASR (实时麦克风)
./build/bin/vad_asr --device-index 6 --vad-type silero

# ASR + LLM (无TTS)
./build/bin/asr_llm --model qwen2.5
```

### 🎵 音频文件处理
```bash
# 处理单个音频文件
./build/bin/asr audio_file.wav

# 批量处理音频文件
./build/bin/asr file1.wav file2.wav file3.wav

# 支持的音频格式：WAV, FLAC, OGG
# 自动重采样到16kHz进行识别
```

### 🔊 文本转语音 (TTS)
```bash
# 中文TTS基本用法
./build/bin/tts --text "你好世界" --tts_type zh

# 英文TTS基本用法
./build/bin/tts --text "Hello, World!" --tts_type en

# 保存为WAV文件
./build/bin/tts --text "欢迎使用语音合成系统" --tts_type zh --save_audio_path output.wav

# 英文语音合成
./build/bin/tts --text "Welcome to the TTS system" --tts_type en --save_audio_path english_output.wav

# 调整语速和说话人
./build/bin/tts --text "这是一个测试" --tts_type zh --tts_speed 1.2 --tts_speaker_id 0 --save_audio_path slow.wav

# 查看帮助
./build/bin/tts --help
```

#### TTS API 使用 (用于C++项目集成)
```cpp
#include "tts_demo.hpp"

// 中文TTS配置
TTSDemo::Params zh_params;
zh_params.tts_speed = 1.0f;
zh_params.tts_speaker_id = 0;
zh_params.tts_type = "zh";

// 英文TTS配置  
TTSDemo::Params en_params;
en_params.tts_speed = 1.0f;
en_params.tts_speaker_id = 0;
en_params.tts_type = "en";

// 创建中文TTS实例
TTSDemo zh_tts(zh_params);
if (zh_tts.initialize()) {
    zh_tts.run("你好世界", "chinese_output.wav");
}

// 创建英文TTS实例
TTSDemo en_tts(en_params);
if (en_tts.initialize()) {
    en_tts.run("Hello World", "english_output.wav");
}
```

### 📱 查找音频设备
```bash
# Python版音频设备搜索
python search_device.py
```

## 配置参数

### ASR-LLM-TTS 完整系统
| 参数 | 描述 | 默认值 | 示例 |
|------|------|--------|------|
| `--sample_rate` | 音频采样率 | 16000 | 48000 |
| `--device_index` | 音频设备索引 | 6 | 7 |
| `--channels` | 音频通道数 (1=单声道, 2=双声道) | 1 | 2 |
| `--vad_type` | VAD类型 | energy | silero |
| `--model` | LLM模型名称 | qwen2.5:0.5b | qwen2.5 |
| `--tts_speed` | TTS语速 | 1.0 | 0.8 |
| `--tts_speaker` | TTS说话人ID | 0 | 0 |
| `--tts_type` | TTS语言类型 | zh | en |

### 云端API系统 (asr_llm_tts_api)
| 参数 | 描述 | 默认值 | 示例 |
|------|------|--------|------|
| `--api_key` | API密钥 | - | sk-xxx |
| `--api_url` | API端点URL | - | https://api.deepseek.com/chat/completions |
| `--model` | 模型名称 | deepseek-chat | gpt-3.5-turbo |
| `--max_tokens` | 最大生成令牌数 | 500 | 1000 |
| `--channels` | 音频通道数 (1=单声道, 2=双声道) | 1 | 2 |
| `--tts_type` | TTS语言类型 | zh | en |
| `--env_file` | 环境配置文件路径 | .env | config.env |

### 流式ASR系统 (streaming_asr)
| 参数 | 描述 | 默认值 | 示例 |
|------|------|--------|------|
| `--max_duration` | 最大录音时长(秒) | 60 | 300 |
| `--silence_threshold` | 静音分段阈值(秒) | 0.5 | 1.0 |
| `--pre_speech_buffer` | 语音前缓冲(秒) | 0.25 | 0.5 |
| `--num_threads` | ASR处理线程数 | 2 | 4 |
| `--channels` | 音频通道数 (1=单声道, 2=双声道) | 1 | 2 |
| `--vad_threshold` | 能量VAD阈值 | 0.005 | 0.01 |

### TTS 独立工具参数
| 参数 | 描述 | 默认值 | 示例 |
|------|------|--------|------|
| `--text` | 要转换的文本 | - | "你好世界" |
| `--save_audio_path` | 保存音频文件路径 | - | "output.wav" |
| `--tts_speed` | TTS语速 | 1.0 | 1.2 |
| `--tts_speaker_id` | TTS说话人ID | 0 | 0 |
| `--tts_type` | TTS语言类型 | zh | en |

### 音频录制参数
| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--silence_duration` | 静音持续时间(秒) | 1.0 |
| `--max_record_time` | 最大录音时间(秒) | 5.0 |
| `--trigger_threshold` | VAD触发阈值 | 0.6 |
| `--stop_threshold` | VAD停止阈值 | 0.35 |

## 项目架构

### 核心模块
```
example/
├── main_asr_llm_tts.cpp    # 完整对话系统主程序 (Ollama)
├── main_asr_llm_tts_api.cpp # 云端API对话系统 (DeepSeek/OpenAI)
├── main_streaming_asr.cpp  # 流式ASR主程序
├── main_ase.cpp            # 音频文件处理引擎 (asr)
├── main_llm.cpp            # ASR+LLM系统
├── main_asr.cpp            # VAD+ASR实时系统 (vad_asr)
└── main_tts.cpp            # TTS独立工具主程序

src/
├── tts_demo.cpp            # TTS API实现 (外部可用)
├── audio_recorder.cpp      # 音频录制模块
├── streaming_audio_recorder.cpp # 流式音频录制器
├── api_comm.cpp            # 云端API通信模块
├── vad_detector.cpp        # 语音活动检测
├── asr_model.cpp           # 语音识别模型
├── asr_thread_pool.cpp     # ASR多线程处理池
├── text_buffer.cpp         # 流式文本缓冲
├── ordered_audio_queue.cpp # 有序音频播放队列
└── tts/
    ├── tts_model.cpp           # TTS模型实现
    └── tts_model_downloader.cpp # TTS模型下载器

include/
├── tts_demo.hpp            # TTS API头文件 (外部接口)
├── streaming_audio_recorder.hpp # 流式录制器接口
├── api_comm.hpp            # API通信接口
├── asr_thread_pool.hpp     # ASR线程池接口
└── ...
```

### 工作流程
```
用户语音 → ASR识别 → LLM流式生成 → 句子分割 → TTS合成 → 有序播放
   ↓           ↓          ↓            ↓         ↓         ↓
 录音缓冲 → 特征提取 → 流式输出 → TextBuffer → 多线程TTS → AudioQueue
```

## 模型说明

### ASR模型 (SenseVoice)
- **模型路径**：`~/.cache/sensevoice/`
- **主要文件**：
  - `model_quant_optimized.onnx` - 量化ASR模型
  - `config.json` - 模型配置
  - `vocab.txt` - 词汇表

### VAD模型 (Silero)
- **模型文件**：`silero_vad.onnx`
- **功能**：语音活动检测，提高识别准确率

### TTS模型 (Matcha)
- **模型路径**：`~/.cache/matcha-tts/`
- **中文TTS模型**：`matcha-icefall-zh-baker/`
  - `model-steps-3.onnx` - 中文声学模型
  - `lexicon.txt` - 中文发音词典
  - `tokens.txt` - 中文音素标记
  - `dict/` - 中文词典目录
- **英文TTS模型**：`matcha-icefall-en_US-ljspeech/`
  - `model-steps-3.onnx` - 英文声学模型
  - `tokens.txt` - 英文音素标记
  - `espeak-ng-data/` - 英文发音数据
- **中英双语TTS模型**：`matcha-icefall-zh-en/` (自动下载)
  - `model-steps-3.onnx` - 中英双语声学模型
  - `vocab_tts.txt` - 中英双语词表
  - 基于 [dengcunqin](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010/summary) 微调的模型
- **共享声码器**：
  - `vocos-22khz-univ.onnx` - 22kHz通用声码器（中文/英文）
  - `vocos-16khz-univ.onnx` - 16kHz通用声码器（中英双语）
- **自动下载**：首次使用时根据语言类型自动下载对应模型

## 技术亮点

### 🚀 实时流式处理
- **LLM流式输出**：边生成边显示，降低感知延迟
- **句子分割**：基于中英文标点符号的智能分句
- **并行TTS**：多线程生成音频，提升效率

### 🎵 有序音频播放
- **顺序保证**：无论TTS生成速度如何，严格按句子顺序播放
- **队列管理**：`OrderedAudioQueue`确保音频连续播放
- **内存优化**：及时释放已播放音频，节省内存

### 🔧 多语言TTS优化
- **中文TTS**：
  - **Jieba分词**：精确的中文文本分词
  - **音素映射**：完整的拼音到音素转换
  - **ISTFT后处理**：频域到时域的高质量音频重建
- **英文TTS**：
  - **espeak-ng集成**：基于IPA音素的英文发音
  - **音素过滤**：去除零宽连接符等问题字符
  - **语言特定优化**：避免双重平滑处理，保持自然音质
- **共享声码器**：高质量Vocos模型支持多语言音频生成

### 🎙️ 远场录音增强
- **VAD Buffer 100倍放大**：显著提升远场语音检测能力
- **自适应放大算法**：根据信号强度动态调整放大倍数 (50x-300x)
- **双重保护机制**：防止内存越界和数据丢失
- **环形缓冲优化**：高效处理长时间录音数据

### 📡 双声道采样 + 声纹链路
- **智能通道混合**：左右声道平均融合，优化音频质量
- **立体声兼容性**：自动检测并处理单声道/双声道输入
- **FIR抗混叠滤波**：重采样前自动应用低通滤波，避免高频折叠
- **声纹原始音频**：未裁剪的原始波形独立送入声纹识别链路，提升准确率
- **安全边界检查**：多层保护防止数组越界访问
- **实时性能优化**：最小化处理延迟，保持实时性

## 常见问题

### 1. 音频相关
**Q: 如何查找正确的音频设备？**
```bash
python search_device.py
# 选择有输入通道的设备索引
```

**Q: 录音无声音？**
A: 检查麦克风权限、设备索引、采样率设置

**Q: 如何启用远场录音功能？**
A: 远场录音已自动启用，通过VAD buffer 100倍放大算法提升检测灵敏度。适用于会议室、教室等大空间环境。

**Q: 双声道录音有什么优势？**
A: 双声道录音可以：
- 提供更好的噪声抑制效果
- 提升远场语音检测准确性
- 改善整体音频质量
- 自动混合左右声道优化输出

**Q: 如何切换单声道/双声道？**
```bash
# 使用双声道
./build/bin/asr_llm_tts --channels 2 --device_index 7

# 使用单声道（默认）
./build/bin/asr_llm_tts --channels 1 --device_index 7
```

### 2. LLM相关
**Q: LLM连接失败？**
```bash
# 检查Ollama服务状态
sudo systemctl status ollama

# 重启Ollama服务
sudo systemctl restart ollama
```

**Q: 模型下载失败？**
```bash
# 手动下载模型
ollama pull qwen2.5:0.5b
```

### 3. TTS相关
**Q: TTS无声音或音质差？**
A: 
- 检查TTS模型是否正确下载
- 确认选择了正确的语言类型 (`--tts_type zh` 或 `--tts_type en`)
- 调整TTS语速参数
- 确认音频输出设备正常

**Q: 英文TTS发音不准确？**
A: 
- 确保已安装espeak-ng：`sudo apt install espeak-ng` (Ubuntu) 或 `brew install espeak` (macOS)
- 英文TTS依赖espeak-ng进行音素转换

**Q: 播放顺序错乱？**
A: 项目已使用`OrderedAudioQueue`解决此问题

**Q: 如何切换中文/英文TTS？**
```bash
# 使用中文TTS
./build/bin/asr_llm_tts --tts_type zh

# 使用英文TTS
./build/bin/asr_llm_tts --tts_type en

# 使用中英双语TTS（推荐，支持混合文本）
./build/bin/asr_llm_tts --tts_type zh-en
```

**Q: 如何使用中英双语TTS？**
```bash
# 完整语音对话系统
./build/bin/asr_llm_tts \
    --device_index 1 \
    --channels 1 \
    --sample_rate 48000 \
    --tts_type zh-en \
    --model qwen2.5:0.5b \
    --vad_type silero

# 单独TTS测试
./build/bin/tts --text "今天学习Python，价格99元" --tts_type zh-en
```
A: 中英双语TTS支持：
- 中英混合文本自动识别
- 阿拉伯数字自动转中文朗读（123 → 一百二十三）
- 罗马数字支持（III → 三，需2位及以上）
- 模型首次使用时自动下载到 `~/.cache/matcha-tts/matcha-icefall-zh-en/`

### 4. 性能优化
**Q: 响应速度慢？**
A:
- 使用轻量级LLM模型 (`qwen2.5:0.5b`)
- 选择能量VAD而非Silero VAD
- 调整采样率到16kHz
- 在高性能设备上运行

## TTS API 集成指南

### 在您的C++项目中使用TTS API

#### 1. 复制所需文件到您的项目
```bash
# 复制头文件
cp include/tts_demo.hpp your_project/include/

# 复制源文件
cp src/tts_demo.cpp your_project/src/
cp src/tts/tts_model.cpp your_project/src/
cp src/tts/tts_model_downloader.cpp your_project/src/
```

#### 2. 修改您的CMakeLists.txt
```cmake
# 添加TTS源文件
add_executable(your_app
    your_main.cpp
    src/tts_demo.cpp
    src/tts_model.cpp
    src/tts_model_downloader.cpp
)

# 链接必需的库
target_link_libraries(your_app 
    onnxruntime
    sndfile
    curl
    pthread
)

# 添加头文件路径
target_include_directories(your_app PRIVATE include)
```

#### 3. 基本API使用示例
详细示例请参考 `TTS_API_USAGE.md` 文档。

```cpp
#include "tts_demo.hpp"

int main() {
    // 配置参数
    TTSDemo::Params params;
    params.tts_speed = 1.0f;        // 正常语速
    params.tts_speaker_id = 0;      // 默认说话人
    params.tts_type = "zh";         // 中文TTS，可选 "zh" 或 "en"
    
    // 创建TTS实例
    TTSDemo tts(params);
    
    // 初始化（首次运行会自动下载模型）
    if (!tts.initialize()) {
        return -1;  // 初始化失败
    }
    
    // 生成语音
    tts.run("你好，欢迎使用TTS系统！", "greeting.wav");
    
    return 0;
}
```

## 开发指南

### 添加新的TTS模型
1. 继承`TTSModel`基类
2. 实现模型加载和推理接口
3. 在`TTSModelDownloader`中添加下载逻辑

### 集成新的LLM
1. 扩展`ollama.hpp`接口
2. 实现流式生成回调
3. 适配`TextBuffer`分句逻辑

### 自定义音频处理
1. 修改`AudioRecorder`录制参数
2. 调整`VADDetector`检测算法
3. 优化`OrderedAudioQueue`播放策略

## 许可证协议

### 开源许可
本项目采用开源许可证。

### 使用范围

** 允许免费使用：**
- 个人学习和研究
- 学术研究和教育用途
- 技术评估和测试
- 开源项目集成

** 商业使用需要许可：**
- 集成到商业产品/服务
- 开发商业应用程序  
- 销售包含本软件的产品
- 提供商业化语音识别服务
- 营利性组织生产环境使用

### 商业许可申请

如需商业使用许可，请联系项目维护者。

**邮箱**: [promuggle@gmail.com]  
**GitHub**: [muggle-stack]  

### 重要提醒

**个人开发者**: 可以自由使用，但请保留版权声明  
**企业用户**: 建议在使用前联系我的获得正式许可  
**开源贡献**: 欢迎提交PR和Issue，共同完善项目


## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v3.1.0 (当前版本)
- ✅ **中英双语TTS支持**：新增基于Matcha-TTS的中英混合语音合成功能
- ✅ **cpp-pinyin集成**：使用cpp-pinyin库进行高精度中文转拼音（带声调）
- ✅ **espeak-ng美式发音**：英文部分使用美式IPA音素，发音更准确
- ✅ **阿拉伯数字朗读**：自动将数字转换为中文读法（如 123 → 一百二十三，3.14 → 三点一四）
- ✅ **罗马数字支持**：2位及以上罗马数字自动转中文（如 III → 三，IV → 四）
- ✅ **智能语言检测**：自动识别中文、英文、数字和标点，无缝处理混合文本
- ✅ **模型自动下载**：zh-en模型首次使用时自动下载，无需手动配置
- 🙏 **感谢 [dengcunqin](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010/summary) 微调并开源中英文Matcha模型**

### v3.0.0
- ✅ **云端ASR集成**：支持阿里云实时ASR，1-2秒低延迟识别
- ✅ **云端LLM集成**：支持云端API（DeepSeek/OpenAI）作为LLM后端
- ✅ **灵活模块化**：ASR和LLM支持本地/云端独立切换，4种部署模式
- ✅ **条件编译**：通过CMake选项(`USE_CLOUD_ASR`, `USE_CLOUD_LLM`)控制编译
- ✅ **.env配置支持**：统一配置管理，支持从.env文件读取云端服务密钥
- ✅ **自动重采样**：ASR API层自动处理任意采样率音频
- ✅ **云端测试工具**：新增`asr_realtime_test`用于云端ASR功能测试

**部署模式对比**：
- 全本地: 本地ASR + Ollama (离线可用)
- 云端ASR: 云端ASR + Ollama (高准确率ASR)
- 云端LLM: 本地ASR + 云端API (强大LLM能力)
- 全云端: 云端ASR + 云端API (最高准确率和能力)

### v2.6.0
- ✅ **声纹识别集成**：新增基于3D-Speaker CamP+模型的声纹识别功能
- ✅ **访问控制机制**：只有注册用户才能使用LLM和TTS功能
- ✅ **原始音频链路**：录音阶段保留未裁剪信号并专供声纹推理使用，避免大幅增益导致的特征失真
- ✅ **FIR抗混叠重采样**：内置FIR低通滤波器，解决下采样时的频率混叠问题，提高音频质量
- ✅ **单双声道自动处理**：智能检测并混合单双声道输入，自动转换为16kHz单声道格式

### v2.5.0
- ✅ **英文TTS支持**：新增基于Matcha-TTS的英文语音合成功能
- ✅ **多语言模型管理**：语言特定模型下载，按需获取中文/英文TTS模型
- ✅ **espeak-ng集成**：支持IPA音素的英文文本到语音转换
- ✅ **智能语言切换**：通过`--tts_type`参数轻松切换中英文TTS
- ✅ **音频质量优化**：语言特定的音频后处理，消除英文TTS爆音问题

### v2.4.0
- ✅ **远场录音增强**：VAD buffer放大100倍，支持超远距离语音检测
- ✅ **双声道采样处理**：智能左右声道混合，显著提升音频质量
- ✅ **自适应放大算法**：根据信号强度动态调整放大倍数 (50x-300x)
- ✅ **安全边界检查**：多层保护防止内存越界和数据丢失
- ✅ **环形缓冲优化**：高效处理长时间录音数据，优化内存使用

### v2.3.0
- ✅ 新增流式ASR功能 (`streaming_asr`)，支持实时连续语音识别
- ✅ 新增云端LLM API接口 (`asr_llm_tts_api`)，支持DeepSeek、OpenAI等多种API
- ✅ 实现环形缓冲区和滑动窗口VAD，提升实时性能
- ✅ 支持多线程ASR处理池，并行处理音频片段
- ✅ 添加API通用接口层，统一支持多种LLM服务商

### v2.2.0
- ✅ 新增独立TTS工具 (`tts`)，支持命令行文本转语音
- ✅ 提供TTSDemo C++ API接口，支持外部项目集成
- ✅ 重构TTS模块架构，分离头文件和实现
- ✅ 输出标准WAV格式音频文件，兼容各种播放器
- ✅ 添加详细的API使用文档和示例代码

### v2.1.0
- ✅ 重构可执行文件命名：asr (文件处理)、vad_asr (实时麦克风)
- ✅ 新增音频文件批量处理引擎
- ✅ 支持批量音频文件语音识别
- ✅ 自动音频格式转换和重采样功能
- ✅ 优化模块化架构设计

### v2.0.0
- ✅ 集成完整ASR-LLM-TTS对话系统
- ✅ 实现流式LLM输出和实时TTS播放
- ✅ 添加有序音频播放队列
- ✅ 支持Matcha-TTS中文语音合成
- ✅ 优化多线程性能和内存使用

### v1.0.0
- ✅ 初始ASR+LLM系统
- ✅ SenseVoice语音识别
- ✅ Ollama LLM集成
- ✅ 多平台支持
