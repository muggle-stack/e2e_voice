#include "ordered_audio_queue.hpp"
#include <portaudio.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

OrderedAudioQueue::OrderedAudioQueue() : next_play_order_(0), stop_flag_(false), started_(false) {
}

OrderedAudioQueue::~OrderedAudioQueue() {
    stop();
}

void OrderedAudioQueue::enqueue(const OrderedAudioData& audio) {
    if (stop_flag_) return;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        audio_map_[audio.order] = audio;
        // Safely handle text substring
        std::string preview = audio.text.length() > 20 ? audio.text.substr(0, 20) + "..." : audio.text;
        std::cout << "[OrderedQueue] Enqueued audio #" << audio.order << ": " << preview << std::endl;
    }
    cv_.notify_one();
}

void OrderedAudioQueue::start() {
    if (started_) return;
    
    stop_flag_ = false;
    next_play_order_ = 0;
    playback_thread_ = std::thread(&OrderedAudioQueue::playbackWorker, this);
    started_ = true;
}

void OrderedAudioQueue::stop() {
    if (!started_) return;
    
    stop_flag_ = true;
    cv_.notify_all();
    
    if (playback_thread_.joinable()) {
        playback_thread_.join();
    }
    
    // 清空剩余音频
    std::lock_guard<std::mutex> lock(mutex_);
    audio_map_.clear();
    
    started_ = false;
}

bool OrderedAudioQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return audio_map_.empty();
}

void OrderedAudioQueue::resetOrder() {
    std::lock_guard<std::mutex> lock(mutex_);
    next_play_order_ = 0;
    audio_map_.clear();
    std::cout << "[OrderedQueue] Reset order to 0 for new conversation" << std::endl;
}

void OrderedAudioQueue::playbackWorker() {
    std::cout << "[OrderedQueue] Playback thread started" << std::endl;
    
    while (!stop_flag_) {
        OrderedAudioData audio;
        bool found = false;
        
        // 等待下一个按顺序的音频
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { 
                return audio_map_.find(next_play_order_) != audio_map_.end() || stop_flag_; 
            });
            
            if (stop_flag_) {
                std::cout << "[OrderedQueue] Stop signal received" << std::endl;
                break;
            }
            
            auto it = audio_map_.find(next_play_order_);
            if (it != audio_map_.end()) {
                // Make a proper copy to avoid alignment issues on RISC-V
                audio = it->second;  // Copy instead of move
                audio_map_.erase(it);
                found = true;
                // Safely handle text substring
                std::string preview = audio.text.length() > 20 ? audio.text.substr(0, 20) + "..." : audio.text;
                std::cout << "[OrderedQueue] Playing audio #" << next_play_order_ << ": " << preview << std::endl;
                next_play_order_++;
            }
        }
        
        // 在锁外播放音频
        if (found && !audio.samples.empty()) {
            playAudioBlocking(audio);
            std::cout << "[OrderedQueue] Finished playing audio #" << (next_play_order_ - 1) << std::endl;
        }
    }
    
    std::cout << "[OrderedQueue] Playback thread ended" << std::endl;
}

void OrderedAudioQueue::playAudioBlocking(const OrderedAudioData& audio) {
    if (audio.samples.empty()) {
        std::cout << "[OrderedQueue] Empty audio samples, skipping" << std::endl;
        return;
    }
    
    // Apply language-aware transition smoothing
    std::vector<float> smoothed_samples = audio.samples;  // Copy for modification
    
    // Only apply additional smoothing for English TTS
    // Chinese TTS already has smoothing applied during generation
    if (audio.language == "en") {
        applySentenceTransitionSmoothing(smoothed_samples, audio.sample_rate);
        // Apply additional DC offset removal for English TTS
        removeDCOffset(smoothed_samples);
    }
    
    PaStream* stream = nullptr;
    
    // 打开音频流
    PaError err = Pa_OpenDefaultStream(&stream,
                                      0,          // 无输入通道
                                      1,          // 单声道输出
                                      paFloat32,  // 32位浮点输出
                                      audio.sample_rate,
                                      256,        // 每缓冲区帧数
                                      nullptr,    // 使用阻塞API
                                      nullptr);   // 无用户数据
    
    if (err != paNoError) {
        std::cerr << "[OrderedQueue] Failed to open audio stream: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
    
    // 开始流
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "[OrderedQueue] Failed to start audio stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        return;
    }
    
    // 分块写入音频数据（阻塞写入）
    // Use smaller chunks for RISC-V to reduce memory pressure
    const size_t chunk_size = 512;  // Smaller chunks for RISC-V
    size_t total_written = 0;
    
    // Create aligned buffer for RISC-V
    std::vector<float> chunk_buffer;
    chunk_buffer.reserve(chunk_size);
    
    for (size_t i = 0; i < smoothed_samples.size(); i += chunk_size) {
        if (stop_flag_) {
            break;
        }
        
        size_t frames_to_write = std::min(chunk_size, smoothed_samples.size() - i);
        
        // Copy to aligned buffer
        chunk_buffer.clear();
        chunk_buffer.insert(chunk_buffer.end(), smoothed_samples.begin() + i, smoothed_samples.begin() + i + frames_to_write);
        
        err = Pa_WriteStream(stream, chunk_buffer.data(), frames_to_write);
        total_written += frames_to_write;
        
        if (err != paNoError && err != paOutputUnderflowed) {
            std::cerr << "[OrderedQueue] Error writing to audio stream: " << Pa_GetErrorText(err) << std::endl;
            break;
        }
    }
    
    // Ensure complete stream drainage before closing (especially important for English TTS)
    if (stream) {
        // Wait for all queued audio to finish playing
        PaError err = Pa_StopStream(stream);
        if (err != paNoError) {
            std::cerr << "[OrderedQueue] Warning: Error stopping stream: " << Pa_GetErrorText(err) << std::endl;
        }
        
        // Wait for stream to become inactive (ensures complete drainage)
        while (Pa_IsStreamActive(stream)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // For English TTS, add additional gap to prevent popping between sentences
        if (audio.language == "en") {
            // Longer pause to ensure complete audio buffer clearance
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Force audio system to flush buffers
            Pa_Sleep(10);
        }
        
        err = Pa_CloseStream(stream);
        if (err != paNoError) {
            std::cerr << "[OrderedQueue] Warning: Error closing stream: " << Pa_GetErrorText(err) << std::endl;
        }
    }
}

void OrderedAudioQueue::applySentenceTransitionSmoothing(std::vector<float>& samples, int sample_rate) {
    if (samples.empty()) return;
    
    // Apply fade-out at the end to prevent popping when stream closes
    // Use longer 20ms fade-out for English TTS (at 22050Hz = ~440 samples)
    const int fade_out_samples = std::min(static_cast<int>(sample_rate * 0.02f), static_cast<int>(samples.size() / 4));
    
    for (int i = 0; i < fade_out_samples && i < samples.size(); ++i) {
        int idx = samples.size() - 1 - i;
        float fade_factor = static_cast<float>(i) / fade_out_samples;
        // Use cosine fade for smoother transition (same as TTS smoothing)
        fade_factor = 0.5f * (1.0f - std::cos(M_PI * fade_factor));
        samples[idx] *= fade_factor;
    }
    
    // Apply longer fade-in at the beginning to prevent click when stream starts
    // Use 10ms fade-in for better smoothing (at 22050Hz = ~220 samples)
    const int fade_in_samples = std::min(static_cast<int>(sample_rate * 0.01f), static_cast<int>(samples.size() / 10));
    
    for (int i = 0; i < fade_in_samples && i < samples.size(); ++i) {
        float fade_factor = static_cast<float>(i) / fade_in_samples;
        // Use cosine fade for smoother transition
        fade_factor = 0.5f * (1.0f - std::cos(M_PI * fade_factor));
        samples[i] *= fade_factor;
    }
}

void OrderedAudioQueue::removeDCOffset(std::vector<float>& samples) {
    if (samples.empty()) return;
    
    // Calculate DC offset (average value)
    float dc_offset = 0.0f;
    for (float sample : samples) {
        dc_offset += sample;
    }
    dc_offset /= samples.size();
    
    // Remove DC offset from all samples
    for (float& sample : samples) {
        sample -= dc_offset;
    }
    
    // Apply gentle high-pass filter to further reduce low-frequency artifacts
    if (samples.size() > 1) {
        float prev_sample = samples[0];
        const float alpha = 0.995f;  // High-pass filter coefficient
        
        for (size_t i = 1; i < samples.size(); ++i) {
            float filtered = alpha * (prev_sample + samples[i] - samples[i-1]);
            prev_sample = samples[i];
            samples[i] = filtered;
        }
    }
}