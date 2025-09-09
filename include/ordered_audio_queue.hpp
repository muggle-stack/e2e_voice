#pragma once

#include <map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <string>

struct OrderedAudioData {
    std::vector<float> samples;
    int sample_rate;
    std::string text;
    size_t order;  // 顺序号
    std::string language;  // Language for conditional smoothing
    
    OrderedAudioData() : sample_rate(22050), order(0), language("zh") {}
    OrderedAudioData(std::vector<float> s, int sr, const std::string& t, size_t ord, const std::string& lang = "zh") 
        : samples(std::move(s)), sample_rate(sr), text(t), order(ord), language(lang) {}
};

class OrderedAudioQueue {
public:
    OrderedAudioQueue();
    ~OrderedAudioQueue();

    // 添加音频到队列，指定顺序号
    void enqueue(const OrderedAudioData& audio);
    
    // 开始播放线程
    void start();
    
    // 停止播放线程并清空队列
    void stop();
    
    // 检查队列是否为空
    bool empty() const;
    
    // 重置播放顺序（用于新的对话轮次）
    void resetOrder();

private:
    void playbackWorker();
    void playAudioBlocking(const OrderedAudioData& audio);
    void applySentenceTransitionSmoothing(std::vector<float>& samples, int sample_rate);
    void removeDCOffset(std::vector<float>& samples);
    
    std::map<size_t, OrderedAudioData> audio_map_;  // 按顺序号存储
    size_t next_play_order_;  // 下一个要播放的顺序号
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_flag_;
    std::thread playback_thread_;
    bool started_;
};