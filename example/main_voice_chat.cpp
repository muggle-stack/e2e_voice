/**
 * 简洁语音对话系统 Demo
 *
 * 整合: audio (采集/播放) + stt (识别) + tts (合成) + ollama (LLM)
 *
 * 用法:
 *   ./voice_chat [--tts zh|en|zh-en] [--model qwen2.5:0.5b] [--input-device 0] [--output-device 0]
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
#include <map>
#include <set>
#include <vector>
#include <chrono>
#include <ctime>

// Audio
#include "space_audio.hpp"
#include "resampler.hpp"

// STT
#include "SpaceAudioSDK.h"

// TTS
#include "SpaceTtsSDK.h"

// VAD (新 SDK 接口)
#include "SpaceVadSDK.h"

// LLM
#include "ollama.hpp"

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
std::atomic<bool> g_speaking{false};  // TTS 正在播放

void signalHandler(int sig) {
    std::cout << "\n" << getTimestamp() << " [退出中...]" << std::endl;
    g_running = false;
}

// ============================================================================
// 参数配置
// ============================================================================

struct Config {
    std::string tts_type = "zh";           // zh, en, zh-en
    std::string llm_model = "qwen2.5:0.5b";
    std::string llm_url = "";              // 空 = 使用 ollama, 否则使用 OpenAI 兼容 API
    int input_device = -1;                  // 输入设备索引 (-1 = 默认)
    int output_device = -1;                 // 输出设备索引 (-1 = 默认)
    int input_sample_rate = 16000;          // 输入采样率
    int output_sample_rate = 0;             // 输出采样率 (0 = 跟随TTS)
    int channels = 1;                       // 声道数 (1=mono, 2=stereo)
    float vad_threshold = 0.5f;             // VAD 阈值
    float silence_duration = 0.3f;          // 静音时长触发识别 (秒)
    int max_tokens = 150;
    bool list_devices = false;              // 是否列出设备
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
        } else if (strcmp(argv[i], "--input-rate") == 0 && i + 1 < argc) {
            cfg.input_sample_rate = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--output-rate") == 0 && i + 1 < argc) {
            cfg.output_sample_rate = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--channels") == 0 || strcmp(argv[i], "-c") == 0) && i + 1 < argc) {
            cfg.channels = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--list-devices") == 0 || strcmp(argv[i], "-l") == 0) {
            cfg.list_devices = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "\n音频设备:\n"
                      << "  -i, --input-device <id>   输入设备索引 (默认: 系统默认)\n"
                      << "  -o, --output-device <id>  输出设备索引 (默认: 系统默认)\n"
                      << "  -l, --list-devices        列出可用音频设备\n"
                      << "\n采样率和声道:\n"
                      << "  --input-rate <rate>       输入采样率 (默认: 16000)\n"
                      << "  --output-rate <rate>      输出采样率 (默认: 跟随TTS)\n"
                      << "  -c, --channels <n>        声道数: 1=单声道, 2=立体声 (默认: 1)\n"
                      << "\nLLM:\n"
                      << "  --model <name>            LLM模型 (默认: qwen2.5:0.5b)\n"
                      << "  --llm-url <url>           LLM API地址 (默认: 使用ollama)\n"
                      << "                            例: http://localhost:8080 (llama-server)\n"
                      << "\nTTS:\n"
                      << "  --tts <type>              TTS后端: zh, en, zh-en (默认: zh)\n"
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
    auto input_devices = SpaceAudio::AudioCapture::ListDevices();
    if (input_devices.empty()) {
        std::cout << getTimestamp() << "   (无可用设备)\n";
    } else {
        for (const auto& dev : input_devices) {
            std::cout << getTimestamp() << "   [" << dev.first << "] " << dev.second << "\n";
        }
    }

    std::cout << getTimestamp() << " \n输出设备 (扬声器):\n";
    auto output_devices = SpaceAudio::AudioPlayer::ListDevices();
    if (output_devices.empty()) {
        std::cout << getTimestamp() << "   (无可用设备)\n";
    } else {
        for (const auto& dev : output_devices) {
            std::cout << getTimestamp() << "   [" << dev.first << "] " << dev.second << "\n";
        }
    }

    std::cout << getTimestamp() << " \n使用方法:\n";
    std::cout << getTimestamp() << "   voice_chat -i <输入设备ID> -o <输出设备ID>\n";
    std::cout << getTimestamp() << " ========================================\n";
}

// ============================================================================
// OpenAI 兼容 API 客户端 (用于 llama-server 等)
// ============================================================================

// 解析 URL 为 host 和 port
bool parseUrl(const std::string& url, std::string& host, int& port, std::string& path) {
    // 格式: http://host:port 或 http://host:port/path
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

// 调用 OpenAI 兼容 API (流式)
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

    // 构建请求体
    nlohmann::json request_body = {
        {"model", model},
        {"messages", {{{"role", "user"}, {"content", prompt}}}},
        {"max_tokens", max_tokens},
        {"stream", true}
    };

    std::string body = request_body.dump();

    // 发送流式请求 (使用 ContentReceiver 接收响应)
    auto res = cli.Post(path.c_str(), body, "application/json",
        [&](const char* data, size_t len) {
            std::string chunk(data, len);

            // 解析 SSE 格式: data: {...}
            std::istringstream ss(chunk);
            std::string line;
            while (std::getline(ss, line)) {
                // 去除行尾的 \r
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
                    } catch (...) {
                        // 忽略解析错误
                    }
                }
            }
            return g_running.load();
        }
    );

    if (!res || res->status != 200) {
        throw std::runtime_error("API 请求失败: " + (res ? std::to_string(res->status) : "无响应"));
    }
}

// ============================================================================
// 有序音频播放队列 - 支持并行合成，按序号顺序播放
// ============================================================================

class OrderedAudioQueue {
public:
    OrderedAudioQueue(SpaceAudio::AudioPlayer& player)
        : player_(player)
        , running_(false)
        , all_done_(false)
        , next_play_seq_(0)
        , total_sentences_(UINT32_MAX)
    {}

    ~OrderedAudioQueue() {
        stop();
    }

    void start(int sample_rate, int channels = 1) {
        running_ = true;
        all_done_ = false;
        next_play_seq_ = 0;
        total_sentences_ = UINT32_MAX;

        // 清空队列
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            audio_map_.clear();
            completed_seqs_.clear();
        }

        player_.Start(sample_rate, channels);
        playback_thread_ = std::thread(&OrderedAudioQueue::playbackLoop, this);
    }

    // 推送音频数据（带序号）
    void push(uint32_t seq, const std::vector<uint8_t>& audio_data) {
        if (!running_ || audio_data.empty()) return;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            audio_map_[seq].push_back(audio_data);
        }
        queue_cv_.notify_one();
    }

    // 标记某个序号的句子合成完成
    void markSentenceComplete(uint32_t seq) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            completed_seqs_.insert(seq);
        }
        queue_cv_.notify_one();
    }

    // 标记所有合成完成
    void markAllDone(uint32_t total) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            total_sentences_ = total;
            all_done_ = true;
        }
        queue_cv_.notify_one();
    }

    void waitForCompletion() {
        if (playback_thread_.joinable()) {
            playback_thread_.join();
        }
    }

    void stop() {
        running_ = false;
        queue_cv_.notify_all();

        if (playback_thread_.joinable()) {
            playback_thread_.join();
        }
    }

private:
    void playbackLoop() {
        while (running_) {
            std::vector<uint8_t> audio_data;
            bool got_data = false;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // 等待条件：当前序号有数据，或者已完成
                queue_cv_.wait(lock, [this] {
                    // 检查当前序号是否有数据可播放
                    auto it = audio_map_.find(next_play_seq_);
                    if (it != audio_map_.end() && !it->second.empty()) {
                        return true;
                    }
                    // 检查当前序号是否已完成（没有更多数据）
                    if (completed_seqs_.count(next_play_seq_) > 0) {
                        return true;
                    }
                    // 检查是否全部完成
                    if (all_done_ && next_play_seq_ >= total_sentences_) {
                        return true;
                    }
                    return !running_;
                });

                if (!running_) break;

                // 检查是否全部播放完成
                if (all_done_ && next_play_seq_ >= total_sentences_) {
                    break;
                }

                auto it = audio_map_.find(next_play_seq_);
                if (it != audio_map_.end() && !it->second.empty()) {
                    // 取出当前序号的一块音频
                    audio_data = std::move(it->second.front());
                    it->second.erase(it->second.begin());
                    got_data = true;

                    // 如果当前序号的所有数据都播放完了，且已标记完成，移动到下一个
                    if (it->second.empty() && completed_seqs_.count(next_play_seq_) > 0) {
                        audio_map_.erase(it);
                        completed_seqs_.erase(next_play_seq_);
                        next_play_seq_++;
                    }
                } else if (completed_seqs_.count(next_play_seq_) > 0) {
                    // 当前序号已完成但没有数据，跳到下一个
                    completed_seqs_.erase(next_play_seq_);
                    next_play_seq_++;
                    continue;
                }
            }

            if (got_data && !audio_data.empty()) {
                player_.Write(audio_data);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        player_.Stop();
    }

    SpaceAudio::AudioPlayer& player_;
    std::atomic<bool> running_;
    std::atomic<bool> all_done_;

    uint32_t next_play_seq_;
    uint32_t total_sentences_;

    std::map<uint32_t, std::vector<std::vector<uint8_t>>> audio_map_;  // 序号 -> 音频块列表
    std::set<uint32_t> completed_seqs_;  // 已完成合成的序号

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread playback_thread_;
};

// ============================================================================
// TTS 回调 - 将合成音频推送到有序队列 (支持重采样)
// ============================================================================

class TtsPlayCallback : public SpacemiT::TtsResultCallback {
public:
    TtsPlayCallback(OrderedAudioQueue& queue, uint32_t seq, int tts_sample_rate, int output_sample_rate)
        : queue_(queue)
        , seq_(seq)
        , tts_sample_rate_(tts_sample_rate)
        , output_sample_rate_(output_sample_rate > 0 ? output_sample_rate : tts_sample_rate)
        , need_resample_(output_sample_rate > 0 && output_sample_rate != tts_sample_rate)
    {
        if (need_resample_) {
            Resampler::Config rs_cfg;
            rs_cfg.input_sample_rate = tts_sample_rate_;
            rs_cfg.output_sample_rate = output_sample_rate_;
            rs_cfg.channels = 1;
            rs_cfg.method = (output_sample_rate_ > tts_sample_rate_)
                ? ResampleMethod::LINEAR_UPSAMPLE
                : ResampleMethod::LINEAR_DOWNSAMPLE;
            resampler_ = std::make_unique<Resampler>(rs_cfg);
            resampler_->initialize();
        }
    }

    void OnOpen() override {}

    void OnEvent(std::shared_ptr<SpacemiT::TtsEngineResult> result) override {
        if (result && result->IsSuccess()) {
            auto audio = result->GetAudioData();
            if (!audio.empty()) {
                if (need_resample_ && resampler_) {
                    const int16_t* samples = reinterpret_cast<const int16_t*>(audio.data());
                    size_t num_samples = audio.size() / 2;

                    std::vector<float> float_in(num_samples);
                    for (size_t i = 0; i < num_samples; i++) {
                        float_in[i] = samples[i] / 32768.0f;
                    }

                    auto float_out = resampler_->process(float_in);

                    std::vector<uint8_t> pcm_out(float_out.size() * 2);
                    int16_t* out_samples = reinterpret_cast<int16_t*>(pcm_out.data());
                    for (size_t i = 0; i < float_out.size(); i++) {
                        float val = std::max(-1.0f, std::min(1.0f, float_out[i]));
                        out_samples[i] = static_cast<int16_t>(val * 32767.0f);
                    }

                    queue_.push(seq_, pcm_out);
                } else {
                    queue_.push(seq_, audio);
                }
            }
        }
    }

    void OnComplete() override {}

    void OnError(const std::string& msg) override {
        std::cerr << getTimestamp() << " [TTS错误] " << msg << std::endl;
    }

    void OnClose() override {
        // 标记这个句子合成完成
        queue_.markSentenceComplete(seq_);
    }

private:
    OrderedAudioQueue& queue_;
    uint32_t seq_;
    int tts_sample_rate_;
    int output_sample_rate_;
    bool need_resample_;
    std::unique_ptr<Resampler> resampler_;
};

// ============================================================================
// 流式 TTS 播放器 - 支持并行合成，按序号顺序播放
// ============================================================================

class StreamingTtsPlayer {
public:
    StreamingTtsPlayer(
        std::shared_ptr<SpacemiT::TtsEngine> tts,
        SpaceAudio::AudioPlayer& player,
        int tts_sample_rate,
        int output_sample_rate,
        int num_synthesis_threads = 2  // 并行合成线程数
    ) : tts_(tts)
      , player_(player)
      , tts_sample_rate_(tts_sample_rate)
      , output_sample_rate_(output_sample_rate)
      , audio_queue_(player)
      , num_threads_(num_synthesis_threads)
      , running_(false)
      , text_finished_(false)
      , next_seq_(0)
      , total_sentences_(0)
    {}

    ~StreamingTtsPlayer() {
        stop();
    }

    void start() {
        running_ = true;
        text_finished_ = false;
        text_buffer_.clear();
        next_seq_ = 0;
        total_sentences_ = 0;

        // 清空句子队列
        {
            std::lock_guard<std::mutex> lock(sentence_mutex_);
            while (!sentence_queue_.empty()) sentence_queue_.pop();
        }

        g_speaking = true;

        // 启动有序音频播放队列
        int out_rate = output_sample_rate_ > 0 ? output_sample_rate_ : tts_sample_rate_;
        audio_queue_.start(out_rate, 1);

        // 启动多个合成线程
        for (int i = 0; i < num_threads_; i++) {
            synthesis_threads_.emplace_back(&StreamingTtsPlayer::synthesisWorker, this);
        }
    }

    void pushToken(const std::string& token) {
        if (!running_) return;

        text_buffer_ += token;

        std::string sentence;
        while (extractSentence(sentence)) {
            uint32_t seq = next_seq_++;
            {
                std::lock_guard<std::mutex> lock(sentence_mutex_);
                sentence_queue_.push({seq, sentence});
            }
            sentence_cv_.notify_one();
        }
    }

    void flush() {
        if (!text_buffer_.empty()) {
            uint32_t seq = next_seq_++;
            {
                std::lock_guard<std::mutex> lock(sentence_mutex_);
                sentence_queue_.push({seq, text_buffer_});
            }
            text_buffer_.clear();
            sentence_cv_.notify_one();
        }

        // 记录总句子数并标记文本完成
        {
            std::lock_guard<std::mutex> lock(sentence_mutex_);
            total_sentences_ = next_seq_;
            text_finished_ = true;
        }
        sentence_cv_.notify_all();
    }

    void stop() {
        if (!running_) return;

        running_ = false;
        sentence_cv_.notify_all();

        for (auto& t : synthesis_threads_) {
            if (t.joinable()) t.join();
        }
        synthesis_threads_.clear();

        audio_queue_.stop();
        g_speaking = false;
    }

    void waitForCompletion() {
        // 等待所有合成线程完成
        for (auto& t : synthesis_threads_) {
            if (t.joinable()) t.join();
        }
        synthesis_threads_.clear();

        // 等待音频播放完成
        audio_queue_.waitForCompletion();
        g_speaking = false;
    }

private:
    bool isSentenceEnd(const std::string& text, size_t pos) {
        if (pos >= text.size()) return false;

        unsigned char c = text[pos];

        if (c == '.' || c == '!' || c == '?' || c == ';' || c == ':') {
            return true;
        }

        if (pos + 2 < text.size()) {
            if ((c & 0xE0) == 0xE0) {
                std::string punct = text.substr(pos, 3);
                if (punct == "。" || punct == "！" || punct == "？" ||
                    punct == "；" || punct == "：" || punct == "，" ||
                    punct == "、") {
                    return true;
                }
            }
        }

        return false;
    }

    bool extractSentence(std::string& sentence) {
        size_t i = 0;
        while (i < text_buffer_.size()) {
            unsigned char c = text_buffer_[i];

            size_t char_len = 1;
            if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;

            if (isSentenceEnd(text_buffer_, i)) {
                sentence = text_buffer_.substr(0, i + char_len);
                text_buffer_ = text_buffer_.substr(i + char_len);

                while (!text_buffer_.empty() &&
                       (text_buffer_[0] == ' ' || text_buffer_[0] == '\n')) {
                    text_buffer_ = text_buffer_.substr(1);
                }

                return true;
            }

            i += char_len;
        }

        return false;
    }

    // 合成工作线程 - 多个线程并行合成
    void synthesisWorker() {
        while (running_) {
            std::pair<uint32_t, std::string> task;
            bool got_task = false;

            {
                std::unique_lock<std::mutex> lock(sentence_mutex_);
                sentence_cv_.wait(lock, [this] {
                    return !sentence_queue_.empty() || text_finished_ || !running_;
                });

                if (!running_) break;

                if (!sentence_queue_.empty()) {
                    task = sentence_queue_.front();
                    sentence_queue_.pop();
                    got_task = true;
                } else if (text_finished_ && sentence_queue_.empty()) {
                    break;
                }
            }

            if (got_task && !task.second.empty() && running_) {
                synthesize(task.first, task.second);
            }
        }

        // 最后一个退出的线程通知播放队列
        {
            std::lock_guard<std::mutex> lock(sentence_mutex_);
            threads_done_++;
            if (threads_done_ >= num_threads_) {
                audio_queue_.markAllDone(total_sentences_);
                threads_done_ = 0;
            }
        }
    }

    void synthesize(uint32_t seq, const std::string& text) {
        // 使用阻塞式 Call() 一次性生成完整音频，避免流式回调的微卡顿
        std::cout << getTimestamp() << " [TTS] 合成句子 #" << seq << ": \"" << text << "\"" << std::endl;
        auto result = tts_->Call(text);
        if (!result || !result->IsSuccess()) {
            std::cerr << getTimestamp() << " [TTS错误] 合成失败: " << text << std::endl;
            audio_queue_.markSentenceComplete(seq);
            return;
        }

        auto audio = result->GetAudioData();
        if (audio.empty()) {
            std::cout << getTimestamp() << " [TTS] 句子 #" << seq << " 无音频输出" << std::endl;
            audio_queue_.markSentenceComplete(seq);
            return;
        }

        std::cout << getTimestamp() << " [TTS] 句子 #" << seq << " 合成完成 (" << audio.size() << " bytes)" << std::endl;

        // 重采样（如果需要）
        if (output_sample_rate_ > 0 && output_sample_rate_ != tts_sample_rate_) {
            Resampler::Config rs_cfg;
            rs_cfg.input_sample_rate = tts_sample_rate_;
            rs_cfg.output_sample_rate = output_sample_rate_;
            rs_cfg.channels = 1;
            rs_cfg.method = (output_sample_rate_ > tts_sample_rate_)
                ? ResampleMethod::LINEAR_UPSAMPLE
                : ResampleMethod::LINEAR_DOWNSAMPLE;
            Resampler resampler(rs_cfg);
            resampler.initialize();

            const int16_t* samples = reinterpret_cast<const int16_t*>(audio.data());
            size_t num_samples = audio.size() / 2;

            std::vector<float> float_in(num_samples);
            for (size_t i = 0; i < num_samples; i++) {
                float_in[i] = samples[i] / 32768.0f;
            }

            auto float_out = resampler.process(float_in);

            std::vector<uint8_t> pcm_out(float_out.size() * 2);
            int16_t* out_samples = reinterpret_cast<int16_t*>(pcm_out.data());
            for (size_t i = 0; i < float_out.size(); i++) {
                float val = std::max(-1.0f, std::min(1.0f, float_out[i]));
                out_samples[i] = static_cast<int16_t>(val * 32767.0f);
            }

            audio_queue_.push(seq, pcm_out);
        } else {
            audio_queue_.push(seq, audio);
        }

        audio_queue_.markSentenceComplete(seq);
    }

    std::shared_ptr<SpacemiT::TtsEngine> tts_;
    SpaceAudio::AudioPlayer& player_;
    int tts_sample_rate_;
    int output_sample_rate_;

    OrderedAudioQueue audio_queue_;

    int num_threads_;
    std::vector<std::thread> synthesis_threads_;
    int threads_done_ = 0;

    std::atomic<bool> running_;
    std::atomic<bool> text_finished_;
    std::atomic<uint32_t> next_seq_;
    uint32_t total_sentences_;

    std::string text_buffer_;
    std::queue<std::pair<uint32_t, std::string>> sentence_queue_;  // <序号, 句子>
    std::mutex sentence_mutex_;
    std::condition_variable sentence_cv_;
};

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
    std::cout << getTimestamp() << "        简洁语音对话系统 Demo\n";
    std::cout << getTimestamp() << " ========================================\n";
    std::cout << getTimestamp() << " TTS后端: " << cfg.tts_type << "\n";
    std::cout << getTimestamp() << " LLM模型: " << cfg.llm_model << "\n";
    if (!cfg.llm_url.empty()) {
        std::cout << getTimestamp() << " LLM URL: " << cfg.llm_url << "\n";
    } else {
        std::cout << getTimestamp() << " LLM后端: Ollama\n";
    }
    std::cout << getTimestamp() << " 输入设备: " << (cfg.input_device < 0 ? "默认" : std::to_string(cfg.input_device)) << "\n";
    std::cout << getTimestamp() << " 输出设备: " << (cfg.output_device < 0 ? "默认" : std::to_string(cfg.output_device)) << "\n";
    std::cout << getTimestamp() << " 输入采样率: " << cfg.input_sample_rate << " Hz, " << cfg.channels << " 声道\n";
    if (cfg.output_sample_rate > 0) {
        std::cout << getTimestamp() << " 输出采样率: " << cfg.output_sample_rate << " Hz\n";
    }
    std::cout << getTimestamp() << " 按 Ctrl+C 退出\n";
    std::cout << getTimestamp() << " ========================================\n\n";

    // -------------------------------------------------------------------------
    // 1. 检查 LLM 后端
    // -------------------------------------------------------------------------
    if (cfg.llm_url.empty()) {
        // 使用 Ollama
        std::cout << getTimestamp() << " [1/5] 检查 Ollama..." << std::flush;
        if (!ollama::is_running()) {
            std::cerr << "\n" << getTimestamp() << " 错误: Ollama 未运行，请先启动: ollama serve\n";
            return 1;
        }
        std::cout << " OK (v" << ollama::get_version() << ")\n";

        // 检查模型
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
        // 使用 OpenAI 兼容 API
        std::cout << getTimestamp() << " [1/5] LLM 后端: " << cfg.llm_url << " OK\n";
    }

    // -------------------------------------------------------------------------
    // 2. 初始化 VAD
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [2/5] 初始化 VAD..." << std::flush;

    auto vad_config = SpacemiT::VadConfig::Silero()
        .withTriggerThreshold(cfg.vad_threshold)
        .withStopThreshold(cfg.vad_threshold - 0.15f);  // stop threshold 比 trigger 低一些

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
    // 5. 初始化 Audio 和重采样器
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [5/5] 初始化音频设备..." << std::flush;
    SpaceAudio::Init(cfg.input_sample_rate, cfg.channels, 3200, cfg.input_device, cfg.output_device);

    SpaceAudio::AudioCapture capture(cfg.input_device);
    SpaceAudio::AudioPlayer player(cfg.output_device);

    // 输入重采样器 (如果输入采样率不是 16kHz，需要重采样到 16kHz 给 VAD/ASR)
    std::unique_ptr<Resampler> input_resampler;
    const int ASR_SAMPLE_RATE = 16000;
    bool need_input_resample = (cfg.input_sample_rate != ASR_SAMPLE_RATE);

    if (need_input_resample) {
        Resampler::Config rs_cfg;
        rs_cfg.input_sample_rate = cfg.input_sample_rate;
        rs_cfg.output_sample_rate = ASR_SAMPLE_RATE;
        rs_cfg.channels = cfg.channels;
        rs_cfg.method = (ASR_SAMPLE_RATE > cfg.input_sample_rate)
            ? ResampleMethod::LINEAR_UPSAMPLE
            : ResampleMethod::LINEAR_DOWNSAMPLE;
        input_resampler = std::make_unique<Resampler>(rs_cfg);
        input_resampler->initialize();
        std::cout << " OK (重采样: " << cfg.input_sample_rate << " -> " << ASR_SAMPLE_RATE << " Hz)\n\n";
    } else {
        std::cout << " OK\n\n";
    }

    // -------------------------------------------------------------------------
    // 状态变量
    // -------------------------------------------------------------------------
    std::vector<float> audio_buffer;
    std::mutex buffer_mutex;
    int silence_frames = 0;
    const int silence_frames_threshold = static_cast<int>(cfg.silence_duration * 16000 / 512);
    bool is_speaking = false;

    // 预缓冲区：保存 VAD 触发前的音频（约 0.3 秒）
    const size_t PRE_BUFFER_FRAMES = 5;  // 每帧约 100ms (1600 samples @ 16kHz)
    std::deque<std::vector<float>> pre_buffer;

    // -------------------------------------------------------------------------
    // 处理识别结果 -> LLM -> TTS
    // -------------------------------------------------------------------------
    auto processText = [&](const std::string& text) {
        if (text.empty()) return;

        std::cout << "\n" << getTimestamp() << " [你]: " << text << "\n";
        std::cout << getTimestamp() << " [LLM] 开始生成...\n";
        std::cout << getTimestamp() << " [AI]: " << std::flush;

        // 创建流式 TTS 播放器
        StreamingTtsPlayer tts_player(tts, player, tts_sample_rate, cfg.output_sample_rate);
        tts_player.start();

        // LLM 流式生成，同时推送到 TTS
        std::string full_response;

        try {
            if (!cfg.llm_url.empty()) {
                // 使用 OpenAI 兼容 API (llama-server 等)
                callOpenAICompatibleAPI(cfg.llm_url, cfg.llm_model, text, cfg.max_tokens,
                    [&](const std::string& chunk) -> bool {
                        if (!chunk.empty()) {
                            std::cout << chunk << std::flush;
                            full_response += chunk;
                            tts_player.pushToken(chunk);  // 实时推送到 TTS
                        }
                        return g_running;
                    });
            } else {
                // 使用 Ollama
                ollama::options opts;
                opts["num_predict"] = cfg.max_tokens;
                opts["temperature"] = 0.7;

                ollama::generate(cfg.llm_model, text,
                    [&](const ollama::response& r) -> bool {
                        std::string chunk = r.as_simple_string();
                        if (!chunk.empty()) {
                            std::cout << chunk << std::flush;
                            full_response += chunk;
                            tts_player.pushToken(chunk);  // 实时推送到 TTS
                        }
                        return g_running;
                    }, opts);
            }
        } catch (const std::exception& e) {
            std::cerr << "\n" << getTimestamp() << " [LLM错误] " << e.what() << std::endl;
            tts_player.stop();
            std::cout << "\n" << getTimestamp() << " [等待语音输入...]\n" << std::flush;
            return;
        }

        std::cout << std::endl;
        std::cout << getTimestamp() << " [LLM] 生成完成 (" << full_response.size() << " 字符)" << std::endl;

        // 刷新剩余文本并等待播放完成
        std::cout << getTimestamp() << " [TTS] 等待播放完成..." << std::endl;
        tts_player.flush();
        tts_player.waitForCompletion();
        std::cout << getTimestamp() << " [TTS] 播放完成" << std::endl;

        std::cout << getTimestamp() << " [等待语音输入...]\n" << std::flush;
    };

    // -------------------------------------------------------------------------
    // 音频采集回调
    // -------------------------------------------------------------------------
    int frame_count = 0;  // 用于控制打印频率
    capture.SetCallback([&](const uint8_t* data, size_t size) {
        if (!g_running || g_speaking) return;

        // PCM16 -> float (支持多声道转单声道)
        const int16_t* samples = reinterpret_cast<const int16_t*>(data);
        size_t total_samples = size / 2;
        size_t num_frames = total_samples / cfg.channels;

        std::vector<float> float_samples(num_frames);
        if (cfg.channels == 1) {
            // 单声道：直接转换
            for (size_t i = 0; i < num_frames; i++) {
                float_samples[i] = samples[i] / 32768.0f;
            }
        } else {
            // 多声道：平均各声道转为单声道
            for (size_t i = 0; i < num_frames; i++) {
                float sum = 0.0f;
                for (int ch = 0; ch < cfg.channels; ch++) {
                    sum += samples[i * cfg.channels + ch] / 32768.0f;
                }
                float_samples[i] = sum / cfg.channels;
            }
        }

        // 重采样到 16kHz (如果需要)
        std::vector<float> resampled_samples;
        const std::vector<float>& vad_samples = need_input_resample
            ? (resampled_samples = input_resampler->process(float_samples), resampled_samples)
            : float_samples;

        // VAD 检测 (使用 16kHz 采样率)
        auto vad_result = vad->Detect(vad_samples);
        float vad_prob = vad_result ? vad_result->GetProbability() : 0.0f;

        // 每10帧打印一次 VAD 状态
        frame_count++;
        if (frame_count % 10 == 0) {
            std::string vad_state_str = "?";
            if (vad_result) {
                switch (vad_result->GetState()) {
                    case SpacemiT::VadState::SILENCE: vad_state_str = "SILENCE"; break;
                    case SpacemiT::VadState::SPEECH_START: vad_state_str = "START"; break;
                    case SpacemiT::VadState::SPEECH: vad_state_str = "SPEECH"; break;
                    case SpacemiT::VadState::SPEECH_END: vad_state_str = "END"; break;
                }
            }
            std::cout << "\r" << getTimestamp() << " [VAD] prob=" << std::fixed << std::setprecision(2) << vad_prob
                      << " state=" << vad_state_str
                      << " speaking=" << (is_speaking ? "Y" : "N")
                      << " buffer=" << audio_buffer.size()
                      << "      " << std::flush;
        }

        std::lock_guard<std::mutex> lock(buffer_mutex);

        if (vad_prob > cfg.vad_threshold) {
            // 检测到语音
            if (!is_speaking) {
                is_speaking = true;
                audio_buffer.clear();

                // 把预缓冲区的音频加入 (VAD 触发前的音频)
                for (const auto& frame : pre_buffer) {
                    audio_buffer.insert(audio_buffer.end(), frame.begin(), frame.end());
                }
                pre_buffer.clear();

                std::cout << "\n" << getTimestamp() << " [VAD] 开始说话 (预缓冲 " << audio_buffer.size() << " samples)..." << std::endl;
            }
            audio_buffer.insert(audio_buffer.end(), vad_samples.begin(), vad_samples.end());
            silence_frames = 0;
        } else if (is_speaking) {
            // 静音计数
            audio_buffer.insert(audio_buffer.end(), vad_samples.begin(), vad_samples.end());
            silence_frames++;

            if (silence_frames >= silence_frames_threshold) {
                // 触发识别
                is_speaking = false;
                std::cout << "\n" << getTimestamp() << " [VAD] 停止说话，触发识别 (buffer=" << audio_buffer.size() << ")" << std::endl;

                if (audio_buffer.size() > 8000) {  // 至少 0.5 秒
                    std::cout << getTimestamp() << " [ASR] 开始识别..." << std::endl;
                    auto result = asr->Recognize(audio_buffer, ASR_SAMPLE_RATE);
                    if (result && !result->IsEmpty()) {
                        std::string text = result->GetText();
                        std::cout << getTimestamp() << " [ASR] 识别完成: \"" << text << "\"" << std::endl;
                        // 异步处理，避免阻塞音频回调
                        std::thread([&processText, text]() {
                            processText(text);
                        }).detach();
                    } else {
                        std::cout << getTimestamp() << " [ASR] 识别完成: (无结果)" << std::endl;
                    }
                }

                audio_buffer.clear();
                silence_frames = 0;
            }
        } else {
            // 非说话状态：更新预缓冲区 (使用重采样后的数据)
            pre_buffer.push_back(vad_samples);
            if (pre_buffer.size() > PRE_BUFFER_FRAMES) {
                pre_buffer.pop_front();
            }
        }
    });

    // -------------------------------------------------------------------------
    // 开始对话
    // -------------------------------------------------------------------------
    std::cout << getTimestamp() << " [等待语音输入...]\n" << std::flush;

    if (!capture.Start()) {
        std::cerr << getTimestamp() << " 错误: 无法启动音频采集\n";
        return 1;
    }

    // 主循环
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 清理
    capture.Stop();
    capture.Close();
    player.Close();

    std::cout << "\n" << getTimestamp() << " [已退出]\n";
    return 0;
}
