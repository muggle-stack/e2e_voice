#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <functional>
#include <memory>
#include <string>
#include <chrono>
#include <portaudio.h>

// WebRTC forward declaration
namespace webrtc {
class AudioProcessing;
}

class VADDetector;

/**
 * AECAudioProcessor - Full-duplex audio processor with WebRTC AEC3
 *
 * This class handles:
 * 1. Full-duplex audio I/O (simultaneous recording and playback)
 * 2. WebRTC AEC3 echo cancellation
 * 3. Audio playback queue for TTS output
 * 4. VAD detection on echo-cancelled audio
 * 5. Barge-in (interrupt) detection
 */
class AECAudioProcessor {
public:
    // Callback types
    using AudioDataCallback = std::function<void(const std::vector<float>&)>;
    using BargeInCallback = std::function<void()>;

    struct Config {
        // Audio settings - all at 48kHz for AEC
        int sample_rate = 48000;
        int channels = 1;
        int frames_per_buffer = 480;  // 10ms at 48kHz
        int input_device_index = -1;  // -1 for default
        int output_device_index = -1; // -1 for default

        // Channel selection (for multi-channel input devices)
        int input_channels = 1;       // Number of channels to open on input device
        int selected_channel = 0;     // Which channel to use (0-based), -1 for mix all

        // VAD settings
        std::string vad_type = "silero";  // "energy" or "silero"
        double vad_trigger_threshold = 0.6;
        double vad_stop_threshold = 0.35;
        double silence_duration = 1.0;    // seconds of silence to stop
        double max_record_time = 10.0;    // max recording time

        // AEC settings
        bool aec_enabled = true;
        bool agc_enabled = true;
        bool ns_enabled = true;  // noise suppression
        bool highpass_enabled = true;

        // Barge-in settings
        bool barge_in_enabled = true;
        double barge_in_threshold = 0.5;  // VAD threshold for barge-in
        int barge_in_frames = 3;          // consecutive frames to trigger
    };

    AECAudioProcessor();
    explicit AECAudioProcessor(const Config& config);
    ~AECAudioProcessor();

    // Initialization
    bool initialize();
    void cleanup();

    // Start/stop audio processing
    bool start();
    void stop();
    bool isRunning() const { return is_running_.load(); }

    // Playback queue - enqueue TTS audio for playback
    // Audio will be resampled to 48kHz if needed
    void enqueuePlayback(const std::vector<float>& samples, int sample_rate);
    void clearPlaybackQueue();
    bool isPlaying() const;
    size_t getPlaybackQueueSize() const;

    // Recording - get echo-cancelled audio
    // Returns audio at 48kHz, caller should downsample if needed
    std::vector<float> getRecordedAudio();
    void clearRecordedAudio();

    // Blocking record with VAD
    // Returns when speech ends (silence detected) or max time reached
    std::vector<float> recordWithVAD();

    // Callbacks
    void setVADCallback(AudioDataCallback callback) { vad_callback_ = callback; }
    void setBargeInCallback(BargeInCallback callback) { barge_in_callback_ = callback; }
    void setVADDetector(VADDetector* vad) { vad_detector_ = vad; }

    // State queries
    bool isSpeechDetected() const { return speech_detected_.load(); }
    bool wasBargeInTriggered() const { return barge_in_triggered_.load(); }
    void resetBargeInFlag() { barge_in_triggered_.store(false); }

    // Resampling utilities
    static std::vector<float> resample(const std::vector<float>& input,
                                       int from_rate, int to_rate);
    static std::vector<int16_t> floatToInt16(const std::vector<float>& input);
    static std::vector<float> int16ToFloat(const std::vector<int16_t>& input);

private:
    // PortAudio callback (static for C API)
    static int paCallback(const void* input, void* output,
                         unsigned long frames_per_buffer,
                         const PaStreamCallbackTimeInfo* time_info,
                         PaStreamCallbackFlags status_flags,
                         void* user_data);

    // Process one frame of audio
    void processAudioFrame(const float* input, float* output, unsigned long frames);

    // VAD processing
    float computeVAD(const std::vector<float>& audio);
    void updateVADState(float vad_prob);

    // Playback buffer management
    struct PlaybackBuffer {
        std::vector<float> samples;
        size_t position = 0;
    };

    // Configuration
    Config config_;

    // WebRTC APM (stored as raw pointer, managed manually)
    webrtc::AudioProcessing* apm_;

    // PortAudio
    PaStream* stream_;
    std::atomic<bool> is_running_;

    // Playback queue
    std::queue<PlaybackBuffer> playback_queue_;
    PlaybackBuffer current_playback_;
    mutable std::mutex playback_mutex_;
    std::atomic<bool> is_playing_;

    // Recording buffer (echo-cancelled)
    std::vector<float> record_buffer_;
    std::mutex record_mutex_;
    std::condition_variable record_cv_;

    // VAD state
    VADDetector* vad_detector_;
    AudioDataCallback vad_callback_;
    BargeInCallback barge_in_callback_;
    std::atomic<bool> speech_detected_;
    std::atomic<bool> recording_active_;
    int vad_speech_count_;
    int vad_silence_count_;
    std::chrono::steady_clock::time_point last_speech_time_;
    std::chrono::steady_clock::time_point recording_start_time_;

    // Barge-in state
    std::atomic<bool> barge_in_triggered_;
    int barge_in_count_;

    // VAD buffer for Silero (needs 512 samples at 16kHz)
    std::vector<float> vad_accumulator_;

    // Temporary buffers to avoid allocation in callback
    std::vector<int16_t> input_int16_;
    std::vector<int16_t> output_int16_;
    std::vector<float> processed_float_;
    std::vector<float> mono_buffer_;  // For channel extraction
};
