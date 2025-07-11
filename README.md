# ASR-LLM-TTS C++ 智能语音对话系统

一个完整的中文智能语音对话系统，集成了自动语音识别(ASR)、大语言模型(LLM)和文本转语音(TTS)功能，支持实时语音交互。

## 🎯 项目特性

### 🔊 完整的语音对话链路
- **ASR (语音识别)**：基于SenseVoice模型的中文语音识别
- **LLM (大语言模型)**：集成Ollama支持多种开源模型
- **TTS (文本转语音)**：基于Matcha-TTS的高质量语音合成
- **流式处理**：LLM流式输出+实时TTS播放，自然对话体验

### 🛠️ 技术特性
- **C++高性能实现**：使用ONNX Runtime进行模型推理
- **模块化设计**：ASR、LLM、TTS可独立使用或组合
- **多线程优化**：并行处理提升响应速度
- **有序音频播放**：确保TTS按句子顺序播放
- **自动模型管理**：首次运行自动下载所需模型

### 🎙️ 音频处理
- **多种VAD算法**：支持能量VAD和Silero VAD
- **多设备支持**：支持各种音频输入设备
- **自动重采样**：支持多种采样率自动转换
- **实时音频队列**：保证音频播放的连续性和顺序性

## 系统要求

### 基础环境
- **操作系统**：Linux (Ubuntu 18.04+) / macOS
- **编译器**：GCC-14 (推荐) 或 GCC 5+
- **CMake**：3.16+

### 系统依赖
- **PortAudio 2.0**：音频录制和播放
- **libsndfile**：音频文件处理
- **ONNX Runtime**：AI模型推理
- **cURL**：模型下载
- **FFTW3**：音频信号处理
- **Ollama**：LLM服务 (可选)

## 安装指南

### 1. 安装系统依赖

#### Ubuntu/Debian
```bash
# 更新包管理器
sudo apt update

# 安装编译工具
sudo apt install gcc-14 g++-14 cmake pkg-config

# 安装音频和网络库
sudo apt install libportaudio-dev libsndfile1-dev libcurl4-openssl-dev libfftw3-dev

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
brew install portaudio libsndfile curl fftw onnxruntime
```

### 2. 安装Ollama (LLM支持)
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动Ollama服务
sudo systemctl start ollama

# 下载推荐模型
ollama pull qwen2.5:0.5b  # 轻量级模型
ollama pull qwen2.5       # 标准模型
```

### 3. 构建项目
```bash
# 克隆项目
git clone <repository-url>
cd asr_llm_tts_cpp/ai

# 构建
chmod +x build.sh
./build.sh
```

## 使用说明

### 🎯 完整对话系统 (推荐)
```bash
# 查看帮助
./build/bin/asr_llm_tts --help

# 使用默认参数运行
./build/bin/asr_llm_tts

# 自定义参数运行
./build/bin/asr_llm_tts \
  --device_index 7 \
  --sample_rate 48000 \
  --vad_type silero \
  --model qwen2.5:0.5b \
  --tts_speed 1.0
```

### 🎙️ 仅语音识别
```bash
# ASR基础功能
./build/bin/asr_cpp --device-index 6 --vad-type silero

# ASR + LLM (无TTS)
./build/bin/asr_llm --model qwen2.5
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
| `--vad_type` | VAD类型 | energy | silero |
| `--model` | LLM模型名称 | qwen2.5:0.5b | qwen2.5 |
| `--tts_speed` | TTS语速 | 1.0 | 0.8 |
| `--tts_speaker` | TTS说话人ID | 0 | 0 |

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
src/
├── main_asr_llm_tts.cpp    # 完整对话系统主程序
├── main_llm.cpp            # ASR+LLM系统
├── main.cpp                # 纯ASR系统
├── audio_recorder.cpp      # 音频录制模块
├── vad_detector.cpp        # 语音活动检测
├── asr_model.cpp           # 语音识别模型
├── text_buffer.cpp         # 流式文本缓冲
├── ordered_audio_queue.cpp # 有序音频播放队列
└── tts/
    ├── tts_model.cpp           # TTS模型实现
    └── tts_model_downloader.cpp # TTS模型下载器
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
- **主要文件**：
  - `matcha-icefall-zh-baker/model-steps-3.onnx` - 声学模型
  - `vocos-22khz-univ.onnx` - 声码器模型
  - `lexicon.txt` - 发音词典
  - `tokens.txt` - 音素标记

## 技术亮点

### 🚀 实时流式处理
- **LLM流式输出**：边生成边显示，降低感知延迟
- **句子分割**：基于中英文标点符号的智能分句
- **并行TTS**：多线程生成音频，提升效率

### 🎵 有序音频播放
- **顺序保证**：无论TTS生成速度如何，严格按句子顺序播放
- **队列管理**：`OrderedAudioQueue`确保音频连续播放
- **内存优化**：及时释放已播放音频，节省内存

### 🔧 中文TTS优化
- **Jieba分词**：精确的中文文本分词
- **音素映射**：完整的拼音到音素转换
- **ISTFT后处理**：频域到时域的高质量音频重建

## 常见问题

### 1. 音频相关
**Q: 如何查找正确的音频设备？**
```bash
python search_device.py
# 选择有输入通道的设备索引
```

**Q: 录音无声音？**
A: 检查麦克风权限、设备索引、采样率设置

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
- 调整TTS语速参数
- 确认音频输出设备正常

**Q: 播放顺序错乱？**
A: 项目已使用`OrderedAudioQueue`解决此问题

### 4. 性能优化
**Q: 响应速度慢？**
A:
- 使用轻量级LLM模型 (`qwen2.5:0.5b`)
- 选择能量VAD而非Silero VAD
- 调整采样率到16kHz
- 在高性能设备上运行

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

## 许可证

MIT License - 详见 LICENSE 文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v2.0.0 (当前版本)
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

如需商业使用许可，请联系项目维护者（`不收费，给个star，告诉我一声就可以拿走`）：

**邮箱**: [3510297507@qq.com]  
**GitHub**: [muggle-stack]  

### 重要提醒

**个人开发者**: 可以自由使用，但请保留版权声明  
**企业用户**: 建议在使用前联系我的获得正式许可  
**开源贡献**: 欢迎提交PR和Issue，共同完善项目

 如果您觉得本项目对您有帮助，希望您在 GitHub 上给本项目一个 Star。
