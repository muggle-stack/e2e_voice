/**
 * AecDuplexProcessor - AEC processor using AudioDuplex
 *
 * Integrates WebRTC APM with SpaceAudio::AudioDuplex for echo cancellation.
 * Uses the full-duplex stream for automatic time alignment between
 * microphone input and speaker output.
 *
 * Usage:
 *   AecDuplexProcessor processor;
 *   processor.setAudioCallback([](const float* data, size_t frames) {
 *       // Process echo-cancelled audio
 *   });
 *   processor.initialize();
 *   processor.start();
 *
 *   // Enqueue TTS audio for playback
 *   processor.enqueuePlayback(tts_samples, tts_sample_rate);
 */

#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <condition_variable>

// Forward declarations
namespace webrtc {
class AudioProcessing;
}

namespace SpaceAudio {
class AudioDuplex;
}

/**
 * AEC Full-Duplex Processor
 *
 * Combines SpaceAudio::AudioDuplex with WebRTC APM for echo cancellation.
 * Barge-in detection should be done at the application layer using VAD.
 */
class AecDuplexProcessor {
public:
    // Callback for processed (echo-cancelled) audio
    using AudioCallback = std::function<void(const float* data, size_t frames, int sample_rate)>;

    struct Config {
        // Audio settings
        int sample_rate = 48000;           // 48kHz recommended for AEC
        int channels = 1;
#ifdef __linux__
        int frames_per_buffer = 960;       // Linux: 20ms @ 48kHz (避免 ALSA underrun)
#else
        int frames_per_buffer = 480;       // macOS: 10ms @ 48kHz
#endif
        int input_device = -1;             // -1 for default
        int output_device = -1;            // -1 for default

        // WebRTC APM settings
        bool aec_enabled = true;           // Echo cancellation
        bool ns_enabled = true;            // Noise suppression
        bool agc_enabled = true;           // Automatic gain control
        bool highpass_enabled = true;      // High-pass filter

        // AEC delay compensation
        int estimated_delay_ms = 50;       // Estimated system delay for AEC reference alignment
    };

    AecDuplexProcessor();
    explicit AecDuplexProcessor(const Config& config);
    ~AecDuplexProcessor();

    // Non-copyable
    AecDuplexProcessor(const AecDuplexProcessor&) = delete;
    AecDuplexProcessor& operator=(const AecDuplexProcessor&) = delete;

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    /**
     * Initialize the processor
     * @return true on success
     */
    bool initialize();

    /**
     * Clean up resources
     */
    void cleanup();

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    bool start();
    void stop();
    bool isRunning() const { return is_running_.load(); }

    // -------------------------------------------------------------------------
    // Callbacks
    // -------------------------------------------------------------------------

    /**
     * Set callback for processed audio
     * Called from processing thread (not audio thread) with echo-cancelled samples
     */
    void setAudioCallback(AudioCallback callback) { audio_callback_ = std::move(callback); }

    // -------------------------------------------------------------------------
    // Playback Queue
    // -------------------------------------------------------------------------

    /**
     * Enqueue audio for playback
     * Audio will be resampled to 48kHz if needed
     *
     * @param samples Audio samples (float, [-1.0, 1.0])
     * @param sample_rate Input sample rate
     */
    void enqueuePlayback(const std::vector<float>& samples, int sample_rate);

    /**
     * Enqueue audio from raw pointer
     */
    void enqueuePlayback(const float* samples, size_t count, int sample_rate);

    /**
     * Clear all queued playback
     */
    void clearPlayback();

    /**
     * Check if currently playing
     */
    bool isPlaying() const { return is_playing_.load(); }

    /**
     * Get playback queue size
     */
    size_t getPlaybackQueueSize() const;

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    const Config& getConfig() const { return config_; }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    /**
     * Resample audio
     */
    static std::vector<float> resample(const std::vector<float>& input,
                                        int from_rate, int to_rate);

private:
    // Full-duplex callback (runs in real-time audio thread)
    void onDuplexAudio(const float* input, float* output, size_t frames, int channels);

    // Processing thread main loop (runs in non-real-time thread)
    void processingLoop();

    // Audio frame for async processing queue
    struct AudioFrame {
        std::vector<float> input;
        std::vector<float> reference;
    };

    // Playback buffer
    struct PlaybackBuffer {
        std::vector<float> samples;
        size_t position = 0;
    };

    // Fill output buffer from playback queue
    size_t fillOutputBuffer(float* output, size_t frames);

    // Process input through AEC
    void processInput(const float* input, const float* output_ref, size_t frames);

    // Configuration
    Config config_;

    // SpaceAudio duplex stream
    std::unique_ptr<SpaceAudio::AudioDuplex> duplex_;

    // WebRTC APM
    webrtc::AudioProcessing* apm_;

    // State
    std::atomic<bool> is_running_;
    std::atomic<bool> is_playing_;

    // Processing thread (for async AEC + user callback)
    std::thread processing_thread_;
    std::atomic<bool> processing_running_{false};

    // Audio frame queue (audio thread -> processing thread)
    std::queue<AudioFrame> audio_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Playback queue
    std::queue<PlaybackBuffer> playback_queue_;
    PlaybackBuffer current_playback_;
    mutable std::mutex playback_mutex_;

    // Callbacks
    AudioCallback audio_callback_;

    // Temporary buffers (pre-allocated to avoid allocation in callback)
    std::vector<int16_t> input_int16_;
    std::vector<int16_t> output_int16_;
    std::vector<float> processed_float_;

    // Playback history ring buffer for AEC delay compensation
    static constexpr size_t kMaxDelayMs = 150;  // Max supported delay (150ms)
    std::vector<float> playback_history_;
    size_t history_write_pos_ = 0;
    size_t delay_samples_ = 0;
    std::vector<float> delayed_ref_buffer_;  // Pre-allocated buffer for delayed reference

    // Fade-out for smooth playback stop (避免 AEC 参考信号突变)
    std::atomic<int> fade_out_frames_{0};
    static constexpr int kFadeOutFrames = 5;  // 淡出 5 帧 (~50ms @ 480 samples/frame)
};
