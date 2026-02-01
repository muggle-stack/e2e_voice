/**
 * 带 AEC 的语音对话系统 Demo (全双工模式)
 *
 * 基于 main_voice_chat.cpp，使用 AecDuplexProcessor 实现回声消除
 * 支持 barge-in（用户打断 TTS 播放）
 *
 * 用法:
 *   ./voice_chat_aec [--tts zh|en|zh-en] [--model qwen2.5:0.5b] [--input-device 0] [--output-device 0]
 */

#include <iostream>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <deque>
#include <sstream>
#include <functional>
#include <memory>
#include <queue>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
#include <fstream>

// AEC 处理器
#include "aec_duplex_processor.hpp"

// 全双工音频（用于列出设备）
#include "space_audio_duplex.hpp"

// Resampler (48kHz -> 16kHz)
#include "resampler.hpp"

// STT
#include "SpaceAudioSDK.h"

// TTS
#include "SpaceTtsSDK.h"

// VAD
#include "SpaceVadSDK.h"

// LLM
#include "ollama.hpp"

// 流式 TTS 分句
#include "text_buffer.hpp"

// ============================================================================
// 时间戳辅助函数
// ============================================================================

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    std::ostringstream oss;
    oss << "[" << std::setfill('0')
        << std::setw(2) << tm_now.tm_hour << ":"
        << std::setw(2) << tm_now.tm_min << ":"
        << std::setw(2) << tm_now.tm_sec << "."
        << std::setw(3) << ms.count() << "]";
    return oss.str();
}

// ============================================================================
// 全局状态
// ============================================================================

std::atomic<bool> g_running{true};
std::atomic<bool> g_processing{false};  // 正在处理 LLM/TTS
std::atomic<bool> g_barge_in{false};    // barge-in 触发标志

void signalHandler(int sig) {
    (void)sig;
    std::cout << "\n" << getTimestamp() << " [退出中...]" << std::endl;
    g_running = false;
}

// ============================================================================
// 参数配置
// ============================================================================

struct Config {
    std::string tts_type = "zh";           // zh, en, zh-en
    std::string llm_model = "qwen2.5:0.5b";
    std::string llm_url = "";              // 空 = 使用 ollama
    int input_device = -1;
    int output_device = -1;
    float vad_threshold = 0.8f;
    float silence_duration = 0.5f;         // 静音时长触发识别 (秒)
    int max_tokens = 150;
    bool list_devices = false;

    // AEC 配置
    bool aec_enabled = true;
    bool ns_enabled = true;
    bool agc_enabled = false;                  // 默认禁用 AGC，避免低能量信号被激进放大
    int aec_delay_ms = 50;                 // AEC 延迟补偿 (毫秒)
    int buffer_frames = 0;                 // 音频缓冲帧数 (0 = 使用平台默认值)

    // 调试：音频录制
    bool save_audio = false;
    std::string audio_file = "aec_debug.wav";
};

Config parseArgs(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tts") == 0 && i + 1 < argc) {
            cfg.tts_type = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.llm_model = argv[++i];
        } else if (strcmp(argv[i], "--llm-url") == 0 && i + 1 < argc) {
            cfg.llm_url = argv[++i];
        } else if ((strcmp(argv[i], "--input-device") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            cfg.input_device = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--output-device") == 0 || strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
            cfg.output_device = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--list-devices") == 0 || strcmp(argv[i], "-l") == 0) {
            cfg.list_devices = true;
        } else if (strcmp(argv[i], "--no-aec") == 0) {
            cfg.aec_enabled = false;
        } else if (strcmp(argv[i], "--no-ns") == 0) {
            cfg.ns_enabled = false;
        } else if (strcmp(argv[i], "--agc") == 0) {
            cfg.agc_enabled = true;  // 启用 AGC（默认禁用）
        } else if (strcmp(argv[i], "--aec-delay") == 0 && i + 1 < argc) {
            cfg.aec_delay_ms = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--buffer-frames") == 0 && i + 1 < argc) {
            cfg.buffer_frames = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--save-audio") == 0) {
            cfg.save_audio = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.audio_file = argv[++i];
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "\n音频设备:\n"
                      << "  -i, --input-device <id>   输入设备索引 (默认: 系统默认)\n"
                      << "  -o, --output-device <id>  输出设备索引 (默认: 系统默认)\n"
                      << "  -l, --list-devices        列出可用音频设备\n"
                      << "\nLLM:\n"
                      << "  --model <name>            LLM模型 (默认: qwen2.5:0.5b)\n"
                      << "  --llm-url <url>           LLM API地址 (默认: 使用ollama)\n"
                      << "\nTTS:\n"
                      << "  --tts <type>              TTS后端: zh, en, zh-en (默认: zh)\n"
                      << "\nAEC:\n"
                      << "  --no-aec                  禁用回声消除\n"
                      << "  --no-ns                   禁用噪声抑制\n"
                      << "  --agc                     启用自动增益控制 (默认禁用)\n"
                      << "  --aec-delay <ms>          AEC延迟补偿 (默认: 50ms, 范围: 20-100ms)\n"
                      << "  --buffer-frames <n>       音频缓冲帧数 (默认: macOS 480, Linux 960)\n"
                      << "\n调试:\n"
                      << "  --save-audio [file]       保存AEC处理后的音频 (默认: aec_debug.wav)\n"
                      << "\n其他:\n"
                      << "  -h, --help                显示帮助\n";
            exit(0);
        }
    }
    return cfg;
}

// ============================================================================
// 列出音频设备
// ============================================================================

void listAudioDevices() {
    std::cout << getTimestamp() << " ========================================\n";
    std::cout << getTimestamp() << "            可用音频设备\n";
    std::cout << getTimestamp() << " ========================================\n\n";

    std::cout << getTimestamp() << " 输入设备 (麦克风):\n";
    auto input_devices = SpaceAudio::AudioDuplex::ListInputDevices();
    if (input_devices.empty()) {
        std::cout << getTimestamp() << "   (无可用设备)\n";
    } else {
        for (const auto& dev : input_devices) {
            std::cout << getTimestamp() << "   [" << dev.first << "] " << dev.second << "\n";
        }
    }

    std::cout << getTimestamp() << " \n输出设备 (扬声器):\n";
    auto output_devices = SpaceAudio::AudioDuplex::ListOutputDevices();
    if (output_devices.empty()) {
        std::cout << getTimestamp() << "   (无可用设备)\n";
    } else {
        for (const auto& dev : output_devices) {
            std::cout << getTimestamp() << "   [" << dev.first << "] " << dev.second << "\n";
        }
    }

    std::cout << getTimestamp() << " \n使用方法:\n";
    std::cout << getTimestamp() << "   voice_chat_aec -i <输入设备ID> -o <输出设备ID>\n";
    std::cout << getTimestamp() << " ========================================\n";
}

// ============================================================================
// OpenAI 兼容 API 客户端
// ============================================================================

bool parseUrl(const std::string& url, std::string& host, int& port, std::string& path) {
    std::string u = url;
    if (u.find("http://") == 0) u = u.substr(7);
    else if (u.find("https://") == 0) u = u.substr(8);

    size_t slash_pos = u.find('/');
    if (slash_pos != std::string::npos) {
        path = u.substr(slash_pos);
        u = u.substr(0, slash_pos);
    } else {
        path = "/v1/chat/completions";
    }

    size_t colon_pos = u.find(':');
    if (colon_pos != std::string::npos) {
        host = u.substr(0, colon_pos);
        port = std::stoi(u.substr(colon_pos + 1));
    } else {
        host = u;
        port = 8080;
    }
    return true;
}

void callOpenAICompatibleAPI(
    const std::string& url,
    const std::string& model,
    const std::string& prompt,
    int max_tokens,
    std::function<bool(const std::string&)> on_token
) {
    std::string host;
    int port;
    std::string path;
    parseUrl(url, host, port, path);

    httplib::Client cli(host, port);
    cli.set_read_timeout(60, 0);

    nlohmann::json request_body = {
        {"model", model},
        {"messages", {{{"role", "user"}, {"content", prompt}}}},
        {"max_tokens", max_tokens},
        {"stream", true}
    };

    std::string body = request_body.dump();

    auto res = cli.Post(path.c_str(), body, "application/json",
        [&](const char* data, size_t len) {
            std::string chunk(data, len);
            std::istringstream ss(chunk);
            std::string line;
            while (std::getline(ss, line)) {
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (line.find("data: ") == 0) {
                    std::string json_str = line.substr(6);
                    if (json_str == "[DONE]") continue;
                    try {
                        auto j = nlohmann::json::parse(json_str);
                        if (j.contains("choices") && !j["choices"].empty()) {
                            auto& delta = j["choices"][0]["delta"];
                            if (delta.contains("content")) {
                                std::string content = delta["content"];
                                if (!on_token(content)) {
                                    return false;
                                }
                            }
                        }
                    } catch (...) {}
                }
            }
            return g_running.load();
        }
    );

    if (!res || res->status != 200) {
        // Barge-in 导致的中断不是错误
        if (g_barge_in) {
            return;
        }
        throw std::runtime_error("API 请求失败");
    }
}

// ============================================================================
// 重采样工具
// ============================================================================

std::vector<float> resample48kTo16k(const float* data, size_t frames) {
    // 48kHz -> 16kHz = 3:1，使用线性插值避免混叠
    const double ratio = 3.0;
    size_t output_frames = frames / 3;
    std::vector<float> output(output_frames);

    for (size_t i = 0; i < output_frames; ++i) {
        double src_pos = i * ratio;
        size_t idx = static_cast<size_t>(src_pos);
        double frac = src_pos - idx;

        if (idx + 1 < frames) {
            output[i] = static_cast<float>(
                data[idx] * (1.0 - frac) + data[idx + 1] * frac);
        } else if (idx < frames) {
            output[i] = data[idx];
        } else {
            output[i] = 0.0f;
        }
    }

    return output;
}

std::vector<float> resampleTo48k(const std::vector<float>& input, int from_rate) {
    if (from_rate == 48000) return input;

    double ratio = 48000.0 / from_rate;
    size_t output_size = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(output_size);

    for (size_t i = 0; i < output_size; ++i) {
        double src_pos = i / ratio;
        size_t src_idx = static_cast<size_t>(src_pos);
        double frac = src_pos - src_idx;

        if (src_idx + 1 < input.size()) {
            output[i] = static_cast<float>(
                input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac
            );
        } else if (src_idx < input.size()) {
            output[i] = input[src_idx];
        }
    }

    return output;
}

// ============================================================================
// TTS 音频转换
// ============================================================================

std::vector<float> pcm16BytesToFloat(const std::vector<uint8_t>& bytes) {
    size_t num_samples = bytes.size() / 2;
    std::vector<float> output(num_samples);
    const int16_t* samples = reinterpret_cast<const int16_t*>(bytes.data());

    for (size_t i = 0; i < num_samples; ++i) {
        output[i] = samples[i] / 32768.0f;
    }

    return output;
}

// ============================================================================
// WAV 文件写入（用于调试）
// ============================================================================

void saveWav(const std::string& filename, const std::vector<int16_t>& data, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }

    uint32_t data_size = static_cast<uint32_t>(data.size() * sizeof(int16_t));
    uint32_t file_size = 36 + data_size;

    // RIFF header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels = 1;
    uint32_t sr = static_cast<uint32_t>(sample_rate);
    uint32_t byte_rate = sr * 2;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;

    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sr), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    file.write(reinterpret_cast<const char*>(data.data()), data_size);
}

// ============================================================================
// 主程序
// ============================================================================

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);

    Config cfg = parseArgs(argc, argv);

    // 列出设备模式
    if (cfg.list_devices) {
        listAudioDevices();
        return 0;
    }

    std::cout << getTimestamp() << " ========================================\n";
    std::cout << getTimestamp() << "    带 AEC 的语音对话系统 (全双工模式)\n";
    std::cout << getTimestamp() << " ========================================\n";
    std::cout << getTimestamp() << " TTS后端: " << cfg.tts_type << "\n";
    std::cout << getTimestamp() << " LLM模型: " << cfg.llm_model << "\n";
    if (!cfg.llm_url.empty()) {
        std::cout << getTimestamp() << " LLM URL: " << cfg.llm_url << "\n";
    } else {
        std::cout << getTimestamp() << " LLM后端: Ollama\n";
    }
    std::cout << getTimestamp() << " AEC: " << (cfg.aec_enabled ? "ON" : "OFF") << "\n";
    std::cout << getTimestamp() << " AEC延迟补偿: " << cfg.aec_delay_ms << " ms\n";
    std::cout << getTimestamp() << " 噪声抑制: " << (cfg.ns_enabled ? "ON" : "OFF") << "\n";
    std::cout << getTimestamp() << " AGC: " << (cfg.agc_enabled ? "ON" : "OFF") << "\n";
    std::cout << getTimestamp() << " 采样率: 48000 Hz (AEC) -> 16000 Hz (VAD/ASR)\n";
    std::cout << getTimestamp() << " 按 Ctrl+C 退出\n";
    std::cout << getTimestamp() << " ========================================\n\n";

    // -------------------------------------------------------------------------
    // 1. 检查 LLM 后端
    // -------------------------------------------------------------------------
    if (cfg.llm_url.empty()) {
        std::cout << getTimestamp() << " [1/5] 检查 Ollama..." << std::flush;
        if (!ollama::is_running()) {
            std::cerr << "\n" << getTimestamp() << " 错误: Ollama 未运行，请先启动: ollama serve\n";
            return 1;
        }
        std::cout << " OK (v" << ollama::get_version() << ")\n";

        auto models = ollama::list_models();
        bool model_found = false;
        for (const auto& m : models) {
            if (m.find(cfg.llm_model) != std::string::npos) {
                model_found = true;
                break;
            }
        }
        if (!model_found) {
            std::cout << getTimestamp() << "   下载模型 " << cfg.llm_model << "...\n";
            ollama::pull_model(cfg.llm_model);
        }
    } else {
        std::cout << getTimestamp() << " [1/5] LLM 后端: " << cfg.llm_url << " OK\n";
    }

    // -------------------------------------------------------------------------
    // 2. 初始化 VAD
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [2/5] 初始化 VAD..." << std::flush;

    auto vad_config = SpacemiT::VadConfig::Silero()
        .withTriggerThreshold(cfg.vad_threshold)
        .withStopThreshold(cfg.vad_threshold - 0.15f);

    auto vad = std::make_shared<SpacemiT::VadEngine>(vad_config);
    if (!vad->IsInitialized()) {
        std::cerr << "\n" << getTimestamp() << " 错误: VAD 初始化失败\n";
        return 1;
    }
    std::cout << " OK (" << vad->GetEngineName() << ")\n";

    // -------------------------------------------------------------------------
    // 3. 初始化 ASR
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [3/5] 初始化 ASR..." << std::flush;
    auto asr = std::make_shared<SpacemiT::AsrEngine>();
    if (!asr->IsInitialized()) {
        std::cerr << "\n" << getTimestamp() << " 错误: ASR 初始化失败\n";
        return 1;
    }
    std::cout << " OK\n";

    // -------------------------------------------------------------------------
    // 4. 初始化 TTS
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [4/5] 初始化 TTS (" << cfg.tts_type << ")..." << std::flush;

    SpacemiT::TtsConfig tts_cfg;
    int tts_sample_rate = 22050;

    if (cfg.tts_type == "en") {
        tts_cfg = SpacemiT::TtsConfig::MatchaEN();
        tts_sample_rate = 22050;
    } else if (cfg.tts_type == "zh-en") {
        tts_cfg = SpacemiT::TtsConfig::MatchaZHEN();
        tts_sample_rate = 16000;
    } else {
        tts_cfg = SpacemiT::TtsConfig::MatchaZH();
        tts_sample_rate = 22050;
    }

    auto tts = std::make_shared<SpacemiT::TtsEngine>(tts_cfg);
    if (!tts->IsInitialized()) {
        std::cerr << "\n" << getTimestamp() << " 错误: TTS 初始化失败\n";
        return 1;
    }
    std::cout << " OK\n";

    // -------------------------------------------------------------------------
    // 5. 初始化 AEC 处理器
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [5/5] 初始化 AEC 音频处理器..." << std::flush;

    AecDuplexProcessor::Config aec_cfg;
    aec_cfg.sample_rate = 48000;
    aec_cfg.channels = 1;
    // 使用用户指定的缓冲帧数，或平台默认值
    if (cfg.buffer_frames > 0) {
        aec_cfg.frames_per_buffer = cfg.buffer_frames;
    }
    // else: 使用 AecDuplexProcessor::Config 的平台默认值 (macOS: 480, Linux: 960)
    aec_cfg.input_device = cfg.input_device;
    aec_cfg.output_device = cfg.output_device;
    aec_cfg.aec_enabled = cfg.aec_enabled;
    aec_cfg.ns_enabled = cfg.ns_enabled;
    aec_cfg.agc_enabled = cfg.agc_enabled;
    aec_cfg.estimated_delay_ms = cfg.aec_delay_ms;

    AecDuplexProcessor aec_processor(aec_cfg);
    if (!aec_processor.initialize()) {
        std::cerr << "\n" << getTimestamp() << " 错误: AEC 初始化失败\n";
        return 1;
    }
    std::cout << " OK\n\n";

    // -------------------------------------------------------------------------
    // 状态变量
    // -------------------------------------------------------------------------
    std::vector<float> audio_buffer;
    std::mutex buffer_mutex;
    int silence_frames = 0;
    const int silence_frames_threshold = static_cast<int>(cfg.silence_duration * 16000 / 512);  // 512 samples per VAD frame
    bool is_speaking = false;
    int frame_count = 0;

    // 预缓冲区（增大以避免 barge-in 丢帧）
    const size_t PRE_BUFFER_FRAMES = 20;  // 20 帧 (~640ms) 确保 barge-in 确认期间的音频不丢失
    std::deque<std::vector<float>> pre_buffer;

    // Barge-in 连续帧检测（避免假阳性）
    int barge_in_confirm_frames = 0;
    const int BARGE_IN_CONFIRM_THRESHOLD = 3;  // 需要连续 3 帧 (~100ms) 才触发

    // Barge-in 后冷却期（让 AEC 稳定，避免过早触发静音检测）
    int post_barge_in_cooldown = 0;
    const int COOLDOWN_FRAMES = 15;  // 冷却期 15 帧 (~480ms) 让 AEC 有足够时间重新收敛

    // 音频录制缓冲区（用于调试）
    std::vector<int16_t> recorded_audio;
    std::mutex record_mutex;

    // VAD 帧累积缓冲区（512 samples = 32ms @ 16kHz，Silero VAD 推荐帧大小）
    const size_t VAD_FRAME_SIZE = 512;
    std::vector<float> vad_frame_buffer;

    // -------------------------------------------------------------------------
    // 处理识别结果 -> LLM -> TTS (流式：边生成边播放)
    // -------------------------------------------------------------------------
    auto processText = [&](const std::string& text) {
        if (text.empty()) return;

        g_processing = true;
        g_barge_in = false;

        std::cout << "\n" << getTimestamp() << " [你]: " << text << "\n";
        std::cout << getTimestamp() << " [LLM] 开始生成...\n";
        std::cout << getTimestamp() << " [AI]: " << std::flush;

        TextBuffer text_buffer;
        std::string full_response;
        int sentence_count = 0;

        // 流式 TTS 合成函数：每完成一句立即合成并播放
        auto synthesizeSentence = [&](const std::string& sentence) {
            if (sentence.empty() || g_barge_in) return;

            sentence_count++;
            auto result = tts->Call(sentence);
            if (result && result->IsSuccess()) {
                auto audio_bytes = result->GetAudioData();
                if (!audio_bytes.empty()) {
                    auto float_samples = pcm16BytesToFloat(audio_bytes);
                    auto audio_48k = resampleTo48k(float_samples, tts_sample_rate);
                    aec_processor.enqueuePlayback(audio_48k, 48000);
                }
            }
        };

        try {
            if (!cfg.llm_url.empty()) {
                // OpenAI 兼容 API
                callOpenAICompatibleAPI(cfg.llm_url, cfg.llm_model, text, cfg.max_tokens,
                    [&](const std::string& chunk) -> bool {
                        if (g_barge_in) return false;  // Barge-in 时停止 LLM

                        if (!chunk.empty()) {
                            std::cout << chunk << std::flush;
                            full_response += chunk;
                            text_buffer.addText(chunk);

                            // 检测完整句子并立即 TTS
                            while (text_buffer.hasSentence() && !g_barge_in) {
                                std::string sentence = text_buffer.getNextSentence();
                                synthesizeSentence(sentence);
                            }
                        }
                        return g_running && !g_barge_in;
                    });
            } else {
                // Ollama
                ollama::options opts;
                opts["num_predict"] = cfg.max_tokens;
                opts["temperature"] = 0.7;

                ollama::generate(cfg.llm_model, text,
                    [&](const ollama::response& r) -> bool {
                        if (g_barge_in) return false;  // Barge-in 时停止 LLM

                        std::string chunk = r.as_simple_string();
                        if (!chunk.empty()) {
                            std::cout << chunk << std::flush;
                            full_response += chunk;
                            text_buffer.addText(chunk);

                            // 检测完整句子并立即 TTS
                            while (text_buffer.hasSentence() && !g_barge_in) {
                                std::string sentence = text_buffer.getNextSentence();
                                synthesizeSentence(sentence);
                            }
                        }
                        return g_running && !g_barge_in;
                    }, opts);
            }
        } catch (const std::exception& e) {
            // 区分 barge-in 中断和真正的错误
            if (g_barge_in) {
                std::cout << "\n" << getTimestamp() << " [LLM] 已因 barge-in 中断生成\n";
            } else {
                std::cerr << "\n" << getTimestamp() << " [LLM错误] " << e.what() << std::endl;
            }
            g_processing = false;
            return;
        }

        std::cout << std::endl;

        // 处理 LLM 结束后可能残留的不完整句子
        if (!g_barge_in) {
            // 强制获取最后一个可能不完整的句子
            text_buffer.stop();
            std::string remaining = text_buffer.getNextSentence();
            if (!remaining.empty()) {
                synthesizeSentence(remaining);
            }
        }

        if (sentence_count > 0) {
            std::cout << getTimestamp() << " [TTS] 流式合成完成 (" << sentence_count << " 句)\n";
        }

        // 等待播放完成（barge-in 会在 VAD 中自动触发 clearPlayback）
        while (aec_processor.isPlaying() && g_running && !g_barge_in) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // 只有在非 barge-in 情况下才清理缓冲区
        if (!g_barge_in) {
            // TTS 播放完成后清理音频缓冲区，防止旧音频被识别
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                audio_buffer.clear();
                pre_buffer.clear();
                silence_frames = 0;
                is_speaking = false;
            }
            vad_frame_buffer.clear();  // 清空 VAD 帧缓冲区
            vad->Reset();  // 重置 VAD 状态
            // 短暂冷却期，让 AEC 稳定
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << getTimestamp() << " [TTS] 播放完成，缓冲区已清理\n";
        } else {
            // Barge-in 情况下，不清理缓冲区，让用户的语音继续被累积
            std::cout << getTimestamp() << " [TTS] Barge-in 打断，保留音频缓冲区\n";
            g_barge_in = false;  // 重置标志
        }

        g_processing = false;
        std::cout << getTimestamp() << " [等待语音输入...]\n" << std::flush;
    };

    // -------------------------------------------------------------------------
    // 设置 AEC 处理器回调
    // -------------------------------------------------------------------------
    aec_processor.setAudioCallback([&](const float* data, size_t frames, int sample_rate) {
        if (!g_running) return;

        // 降采样到 16kHz 给 VAD/ASR
        auto samples_16k = resample48kTo16k(data, frames);
        if (samples_16k.empty()) return;

        // 录制音频（用于调试）
        if (cfg.save_audio) {
            std::lock_guard<std::mutex> lock(record_mutex);
            for (float s : samples_16k) {
                recorded_audio.push_back(static_cast<int16_t>(std::clamp(s, -1.0f, 1.0f) * 32767.0f));
            }
        }

        // 累积音频到 VAD 帧缓冲区
        vad_frame_buffer.insert(vad_frame_buffer.end(), samples_16k.begin(), samples_16k.end());

        // 当缓冲区达到 VAD 帧大小时进行检测
        while (vad_frame_buffer.size() >= VAD_FRAME_SIZE && g_running) {
            // 取出一帧
            std::vector<float> vad_frame(vad_frame_buffer.begin(), vad_frame_buffer.begin() + VAD_FRAME_SIZE);
            vad_frame_buffer.erase(vad_frame_buffer.begin(), vad_frame_buffer.begin() + VAD_FRAME_SIZE);

            // VAD 检测
            auto vad_result = vad->Detect(vad_frame);
            float vad_prob = vad_result ? vad_result->GetProbability() : 0.0f;

            // 每10帧打印一次 VAD 状态
            frame_count++;
            if (frame_count % 10 == 0) {
                std::cout << "\r" << getTimestamp() << " [VAD] prob=" << std::fixed
                          << std::setprecision(2) << vad_prob
                          << " speaking=" << (is_speaking ? "Y" : "N")
                          << " buffer=" << audio_buffer.size()
                          << " playing=" << (aec_processor.isPlaying() ? "Y" : "N")
                          << "      " << std::flush;
            }

            // TTS 播放期间：检测 barge-in（需要连续多帧确认，避免假阳性）
            if (g_processing) {
                if (aec_processor.isPlaying() && vad_prob > cfg.vad_threshold) {
                    barge_in_confirm_frames++;
                    // 在确认期间也累积音频到预缓冲区
                    pre_buffer.push_back(vad_frame);
                    if (pre_buffer.size() > PRE_BUFFER_FRAMES + BARGE_IN_CONFIRM_THRESHOLD) {
                        pre_buffer.pop_front();
                    }

                    if (barge_in_confirm_frames >= BARGE_IN_CONFIRM_THRESHOLD) {
                        std::cout << "\n" << getTimestamp() << " [Barge-in] 用户打断 (连续"
                                  << barge_in_confirm_frames << "帧, prob=" << vad_prob << ")，停止播放\n";
                        aec_processor.clearPlayback();
                        g_barge_in = true;
                        barge_in_confirm_frames = 0;

                        // 设置冷却期
                        post_barge_in_cooldown = COOLDOWN_FRAMES;

                        // Barge-in 后立即开始累积用户语音
                        std::lock_guard<std::mutex> lock(buffer_mutex);
                        is_speaking = true;
                        audio_buffer.clear();
                        // 添加预缓冲区的音频（包含确认期间的帧）
                        for (const auto& frame : pre_buffer) {
                            audio_buffer.insert(audio_buffer.end(), frame.begin(), frame.end());
                        }
                        pre_buffer.clear();
                        silence_frames = 0;
                    }
                } else {
                    barge_in_confirm_frames = 0;  // 重置确认计数
                    // 继续维护预缓冲区
                    pre_buffer.push_back(vad_frame);
                    if (pre_buffer.size() > PRE_BUFFER_FRAMES) {
                        pre_buffer.pop_front();
                    }
                }
                continue;  // 继续跳过正常流程
            }

            std::lock_guard<std::mutex> lock(buffer_mutex);

            if (vad_prob > cfg.vad_threshold) {
                if (!is_speaking) {
                    is_speaking = true;
                    audio_buffer.clear();

                    // 添加预缓冲区的音频
                    for (const auto& frame : pre_buffer) {
                        audio_buffer.insert(audio_buffer.end(), frame.begin(), frame.end());
                    }
                    pre_buffer.clear();

                    std::cout << "\n" << getTimestamp() << " [VAD] 开始说话 (prob=" << vad_prob << ")...\n";
                }
                audio_buffer.insert(audio_buffer.end(), vad_frame.begin(), vad_frame.end());
                silence_frames = 0;
            } else if (is_speaking) {
                audio_buffer.insert(audio_buffer.end(), vad_frame.begin(), vad_frame.end());

                // 冷却期内不增加静音计数（给 AEC 时间稳定）
                if (post_barge_in_cooldown > 0) {
                    post_barge_in_cooldown--;
                } else {
                    silence_frames++;
                }

                if (silence_frames >= silence_frames_threshold) {
                    is_speaking = false;
                    std::cout << "\n" << getTimestamp() << " [VAD] 停止说话，触发识别\n";

                    if (audio_buffer.size() > 8000) {
                        std::cout << getTimestamp() << " [ASR] 开始识别...\n";
                        auto result = asr->Recognize(audio_buffer, 16000);
                        if (result && !result->IsEmpty()) {
                            std::string text = result->GetText();
                            std::cout << getTimestamp() << " [ASR] 识别完成: \"" << text << "\"\n";
                            std::thread([&processText, text]() {
                                processText(text);
                            }).detach();
                        } else {
                            std::cout << getTimestamp() << " [ASR] 识别完成: (无结果)\n";
                        }
                    }

                    audio_buffer.clear();
                    silence_frames = 0;
                }
            } else {
                pre_buffer.push_back(vad_frame);
                if (pre_buffer.size() > PRE_BUFFER_FRAMES) {
                    pre_buffer.pop_front();
                }
            }
        }
    });

    // -------------------------------------------------------------------------
    // 开始对话
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [等待语音输入...]\n" << std::flush;

    if (!aec_processor.start()) {
        std::cerr << getTimestamp() << " 错误: 无法启动音频处理\n";
        return 1;
    }

    // 主循环
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 清理
    aec_processor.stop();

    // 保存录制的音频
    if (cfg.save_audio && !recorded_audio.empty()) {
        std::cout << getTimestamp() << " [保存音频] " << cfg.audio_file
                  << " (" << recorded_audio.size() << " samples, "
                  << (recorded_audio.size() / 16000.0f) << " 秒)\n";
        saveWav(cfg.audio_file, recorded_audio, 16000);
    }

    std::cout << "\n" << getTimestamp() << " [已退出]\n";
    return 0;
}
