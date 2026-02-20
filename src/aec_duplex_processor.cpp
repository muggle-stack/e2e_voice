/**
 * AecDuplexProcessor Implementation
 *
 * Integrates WebRTC APM with SpaceAudio::AudioDuplex.
 * Provides echo cancellation for full-duplex audio.
 * Barge-in detection should be done at the application layer using VAD.
 */

#include "aec_duplex_processor.hpp"
#include "audio_duplex_api.hpp"

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

// WebRTC includes
#include <webrtc/modules/audio_processing/include/audio_processing.h>

// Frame size for WebRTC APM (10ms at 48kHz = 480 samples)
constexpr int kFrameSizeMs = 10;
constexpr int kDefaultSampleRate = 48000;
constexpr int kFrameSize = kDefaultSampleRate * kFrameSizeMs / 1000;  // 480 samples

// ============================================================================
// Constructor / Destructor
// ============================================================================

AecDuplexProcessor::AecDuplexProcessor()
    : AecDuplexProcessor(Config()) {}

AecDuplexProcessor::AecDuplexProcessor(const Config& config)
    : config_(config)
    , duplex_(nullptr)
    , apm_(nullptr)
    , is_running_(false)
    , is_playing_(false)
    , audio_callback_(nullptr)
    , history_write_pos_(0)
    , delay_samples_(0) {

    // Pre-allocate buffers
    input_int16_.resize(config_.frames_per_buffer);
    output_int16_.resize(config_.frames_per_buffer);
    processed_float_.resize(config_.frames_per_buffer);
    delayed_ref_buffer_.resize(config_.frames_per_buffer);

    // Initialize playback history buffer (150ms @ 48kHz = 7200 samples)
    size_t history_samples = (config_.sample_rate * kMaxDelayMs) / 1000;
    playback_history_.resize(history_samples, 0.0f);

    // Calculate delay in samples
    delay_samples_ = (config_.sample_rate * config_.estimated_delay_ms) / 1000;
    if (delay_samples_ >= history_samples) {
        delay_samples_ = history_samples - config_.frames_per_buffer;
    }
}

AecDuplexProcessor::~AecDuplexProcessor() {
    cleanup();
}

// ============================================================================
// Initialization
// ============================================================================

bool AecDuplexProcessor::initialize() {
    std::cout << "[AecDuplex] Initializing..." << std::endl;
    std::cout << "[AecDuplex] Sample rate: " << config_.sample_rate << " Hz" << std::endl;
    std::cout << "[AecDuplex] Frame size: " << config_.frames_per_buffer << " samples" << std::endl;

    // Ensure sample rate is 48kHz for optimal AEC performance
    if (config_.sample_rate != 48000) {
        std::cerr << "[AecDuplex] Warning: Sample rate should be 48000 Hz for best AEC performance" << std::endl;
        config_.sample_rate = 48000;
    }

    // Create WebRTC APM
    rtc::scoped_refptr<webrtc::AudioProcessing> apm_ref =
        webrtc::AudioProcessingBuilder().Create();

    if (!apm_ref) {
        std::cerr << "[AecDuplex] Failed to create AudioProcessing instance" << std::endl;
        return false;
    }

    // Store the raw pointer and add a reference to prevent deletion
    apm_ = apm_ref.get();
    apm_->AddRef();

    // Configure APM
    webrtc::AudioProcessing::Config apm_config;

    // Echo cancellation
    apm_config.echo_canceller.enabled = config_.aec_enabled;
    apm_config.echo_canceller.mobile_mode = false;  // Desktop mode for better quality

    // Gain control - 默认禁用，因为 AGC 在低能量输入时会激进放大导致 AEC reduction 为负
    // 如果需要启用，请显式设置 agc_enabled = true
    if (config_.agc_enabled) {
        apm_config.gain_controller1.enabled = false;  // 禁用 AGC1 避免激进放大
        apm_config.gain_controller2.enabled = false;  // 禁用 AGC2
    }

    // High-pass filter
    apm_config.high_pass_filter.enabled = config_.highpass_enabled;

    // Noise suppression - 使用 kLow 级别，避免过度处理导致语音失真
    if (config_.ns_enabled) {
        apm_config.noise_suppression.enabled = true;
        apm_config.noise_suppression.level =
            webrtc::AudioProcessing::Config::NoiseSuppression::kLow;
    }

    apm_->ApplyConfig(apm_config);

    std::cout << "[AecDuplex] WebRTC APM configured:" << std::endl;
    std::cout << "[AecDuplex]   Echo Cancellation: " << (config_.aec_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "[AecDuplex]   Noise Suppression: " << (config_.ns_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "[AecDuplex]   Gain Control: " << (config_.agc_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "[AecDuplex]   High-pass Filter: " << (config_.highpass_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "[AecDuplex]   Delay Compensation: " << config_.estimated_delay_ms << " ms ("
              << delay_samples_ << " samples)" << std::endl;

    // Create AudioDuplex
    duplex_ = std::make_unique<SpaceAudio::AudioDuplex>(
        config_.input_device,
        config_.output_device
    );

    // Set callback
    duplex_->SetCallback([this](const float* input, float* output, size_t frames, int channels) {
        onDuplexAudio(input, output, frames, channels);
    });

    std::cout << "[AecDuplex] Initialization complete" << std::endl;
    return true;
}

void AecDuplexProcessor::cleanup() {
    stop();

    duplex_.reset();

    if (apm_) {
        apm_->Release();
        apm_ = nullptr;
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

bool AecDuplexProcessor::start() {
    if (is_running_) return true;

    if (!duplex_) {
        std::cerr << "[AecDuplex] Not initialized" << std::endl;
        return false;
    }

    // Start processing thread first
    processing_running_ = true;
    processing_thread_ = std::thread(&AecDuplexProcessor::processingLoop, this);

    if (!duplex_->Start(config_.sample_rate, config_.channels, config_.frames_per_buffer)) {
        std::cerr << "[AecDuplex] Failed to start duplex stream" << std::endl;
        // Stop processing thread on failure
        processing_running_ = false;
        queue_cv_.notify_all();
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        return false;
    }

    is_running_ = true;
    std::cout << "[AecDuplex] Started (async processing enabled)" << std::endl;
    return true;
}

void AecDuplexProcessor::stop() {
    if (!is_running_) return;

    is_running_ = false;

    if (duplex_) {
        duplex_->Stop();
    }

    // Stop processing thread
    processing_running_ = false;
    queue_cv_.notify_all();
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }

    // Clear any remaining frames in queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!audio_queue_.empty()) {
            audio_queue_.pop();
        }
    }

    std::cout << "[AecDuplex] Stopped" << std::endl;
}

// ============================================================================
// Duplex Audio Callback (runs in real-time audio thread)
// ============================================================================

void AecDuplexProcessor::onDuplexAudio(const float* input, float* output,
                                        size_t frames, int channels) {
    // Step 1: Fill output buffer from playback queue (must be fast!)
    fillOutputBuffer(output, frames);

    // Step 2: Save output to playback history ring buffer
    for (size_t i = 0; i < frames; ++i) {
        playback_history_[history_write_pos_] = output[i];
        history_write_pos_ = (history_write_pos_ + 1) % playback_history_.size();
    }

    // Step 3: Get delayed reference signal from history
    size_t history_size = playback_history_.size();
    size_t read_pos = (history_write_pos_ + history_size - delay_samples_ - frames) % history_size;

    // Step 4: Enqueue input + reference for async processing (no AEC here!)
    if (input && processing_running_) {
        AudioFrame frame;
        frame.input.assign(input, input + frames);
        frame.reference.resize(frames);
        for (size_t i = 0; i < frames; ++i) {
            frame.reference[i] = playback_history_[(read_pos + i) % history_size];
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            // Limit queue size to prevent memory growth (drop oldest if too many)
            if (audio_queue_.size() < 100) {
                audio_queue_.push(std::move(frame));
            }
        }
        queue_cv_.notify_one();
    }
}

// ============================================================================
// Processing Thread (runs in non-real-time thread)
// ============================================================================

void AecDuplexProcessor::processingLoop() {
    std::cout << "[AecDuplex] Processing thread started" << std::endl;

    while (processing_running_) {
        AudioFrame frame;

        // Wait for audio frame
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !audio_queue_.empty() || !processing_running_;
            });

            if (!processing_running_) break;
            if (audio_queue_.empty()) continue;

            frame = std::move(audio_queue_.front());
            audio_queue_.pop();
        }

        // Process through AEC (now in non-real-time thread, can take longer)
        if (apm_) {
            processInput(frame.input.data(), frame.reference.data(), frame.input.size());
        }

        // Call user callback with processed audio
        if (audio_callback_) {
            audio_callback_(processed_float_.data(), frame.input.size(), config_.sample_rate);
        }
    }

    std::cout << "[AecDuplex] Processing thread stopped" << std::endl;
}

size_t AecDuplexProcessor::fillOutputBuffer(float* output, size_t frames) {
    std::lock_guard<std::mutex> lock(playback_mutex_);

    // 检查是否处于淡出状态
    int fade_frames = fade_out_frames_.load();
    if (fade_frames > 0) {
        // 淡出处理：继续输出当前音频但逐渐降低音量
        size_t samples_written = 0;

        // 如果有音频数据，输出带淡出的音频
        if (!current_playback_.samples.empty() &&
            current_playback_.position < current_playback_.samples.size()) {

            size_t samples_available = current_playback_.samples.size() - current_playback_.position;
            size_t samples_to_copy = std::min(frames, samples_available);

            // 复制并应用淡出增益
            float gain = static_cast<float>(fade_frames) / kFadeOutFrames;
            for (size_t i = 0; i < samples_to_copy; ++i) {
                output[i] = current_playback_.samples[current_playback_.position + i] * gain;
            }
            samples_written = samples_to_copy;
            current_playback_.position += samples_to_copy;
        }

        // 填充剩余为静音
        if (samples_written < frames) {
            std::memset(output + samples_written, 0, (frames - samples_written) * sizeof(float));
        }

        // 递减淡出计数
        fade_frames--;
        fade_out_frames_.store(fade_frames);

        // 淡出完成，清空队列
        if (fade_frames == 0) {
            while (!playback_queue_.empty()) {
                playback_queue_.pop();
            }
            current_playback_.samples.clear();
            current_playback_.position = 0;
            is_playing_.store(false);
        }

        return samples_written;
    }

    // If current buffer is exhausted, get next from queue
    if (current_playback_.samples.empty() ||
        current_playback_.position >= current_playback_.samples.size()) {

        if (!playback_queue_.empty()) {
            current_playback_ = std::move(playback_queue_.front());
            playback_queue_.pop();
            current_playback_.position = 0;
        } else {
            current_playback_.samples.clear();
            current_playback_.position = 0;
        }
    }

    // Check if we have playback samples
    if (current_playback_.samples.empty() ||
        current_playback_.position >= current_playback_.samples.size()) {
        // No playback, output silence
        std::memset(output, 0, frames * sizeof(float));
        is_playing_.store(false);
        return 0;
    }

    // Copy samples to output
    size_t samples_available = current_playback_.samples.size() - current_playback_.position;
    size_t samples_to_copy = std::min(frames, samples_available);

    std::memcpy(output,
                current_playback_.samples.data() + current_playback_.position,
                samples_to_copy * sizeof(float));

    // Pad with zeros if needed
    if (samples_to_copy < frames) {
        std::memset(output + samples_to_copy, 0, (frames - samples_to_copy) * sizeof(float));
    }

    current_playback_.position += samples_to_copy;

    // Update playing state: still playing if queue has data or current buffer has remaining data
    bool still_playing = !playback_queue_.empty() ||
                         (current_playback_.position < current_playback_.samples.size());
    is_playing_.store(still_playing);

    return samples_to_copy;
}

void AecDuplexProcessor::processInput(const float* input, const float* output_ref, size_t frames) {
    webrtc::StreamConfig stream_config(config_.sample_rate, config_.channels);

    // Ensure buffers are sized correctly
    if (input_int16_.size() < frames) {
        input_int16_.resize(frames);
        output_int16_.resize(frames);
        processed_float_.resize(frames);
    }

    // Convert output to int16 for reverse stream (playback reference)
    for (size_t i = 0; i < frames; ++i) {
        float sample = std::clamp(output_ref[i], -1.0f, 1.0f);
        output_int16_[i] = static_cast<int16_t>(
            std::clamp(sample * 32768.0f, -32768.0f, 32767.0f));
    }

    // Process reverse stream (playback reference for AEC)
    apm_->ProcessReverseStream(output_int16_.data(), stream_config,
                               stream_config, output_int16_.data());

    // Convert input to int16
    for (size_t i = 0; i < frames; ++i) {
        float sample = std::clamp(input[i], -1.0f, 1.0f);
        input_int16_[i] = static_cast<int16_t>(
            std::clamp(sample * 32768.0f, -32768.0f, 32767.0f));
    }

    // Process stream (applies AEC, NS, AGC)
    apm_->ProcessStream(input_int16_.data(), stream_config,
                        stream_config, input_int16_.data());

    // Convert back to float
    for (size_t i = 0; i < frames; ++i) {
        processed_float_[i] = static_cast<float>(input_int16_[i]) / 32768.0f;
    }
}

// ============================================================================
// Playback Queue
// ============================================================================

void AecDuplexProcessor::enqueuePlayback(const std::vector<float>& samples, int sample_rate) {
    enqueuePlayback(samples.data(), samples.size(), sample_rate);
}

void AecDuplexProcessor::enqueuePlayback(const float* samples, size_t count, int sample_rate) {
    if (!samples || count == 0) return;

    std::vector<float> resampled;

    // Resample to 48kHz if needed
    if (sample_rate != config_.sample_rate) {
        resampled = resample(std::vector<float>(samples, samples + count),
                             sample_rate, config_.sample_rate);
    } else {
        resampled.assign(samples, samples + count);
    }

    PlaybackBuffer buffer;
    buffer.samples = std::move(resampled);
    buffer.position = 0;

    {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        playback_queue_.push(std::move(buffer));
        is_playing_.store(true);  // Set playing state immediately when audio is queued
    }
}

void AecDuplexProcessor::clearPlayback() {
    // 不立即清空，设置淡出标志，避免 AEC 参考信号突变
    // 淡出会在 fillOutputBuffer() 中执行
    fade_out_frames_.store(kFadeOutFrames);
}

size_t AecDuplexProcessor::getPlaybackQueueSize() const {
    std::lock_guard<std::mutex> lock(playback_mutex_);
    return playback_queue_.size();
}

// ============================================================================
// Utilities
// ============================================================================

std::vector<float> AecDuplexProcessor::resample(const std::vector<float>& input,
                                                 int from_rate, int to_rate) {
    if (from_rate == to_rate || input.empty()) {
        return input;
    }

    double ratio = static_cast<double>(to_rate) / from_rate;
    size_t output_size = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(output_size);

    // Simple linear interpolation resampling
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
        } else {
            output[i] = 0.0f;
        }
    }

    return output;
}
