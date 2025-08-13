#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <string>
#include <portaudio.h>

// Forward declaration
class VADDetector;

class AudioRecorder {
public:
    using AudioCallback = std::function<void(const std::vector<float>&)>;
    
    struct Config {
        int sample_rate = 16000;
        int channels = 1;
        int frames_per_buffer = 512;
        int device_index = -1; // -1 for default device
        double silence_duration = 1.0; // seconds
        double max_record_time = 5.0; // seconds
        double trigger_threshold = 0.6;
        double stop_threshold = 0.35;
        std::string vad_type = "energy"; // "energy" or "silero"
    };

    AudioRecorder();
    explicit AudioRecorder(const Config& config);
    ~AudioRecorder();

    bool initialize();
    void cleanup();
    
    // Blocking recording
    std::vector<float> recordAudio();
    
    // Non-blocking recording
    void startRecording();
    void stopRecording();
    bool isRecording() const { return is_recording_.load(); }
    std::vector<float> getLastRecording();
    
    // Set VAD callback
    void setVADCallback(AudioCallback callback) { vad_callback_ = callback; }
    
    // Set VAD detector for Silero VAD
    void setVADDetector(VADDetector* vad_detector) { vad_detector_ = vad_detector; }

private:
    static int audioCallback(const void* input_buffer, void* output_buffer,
                           unsigned long frames_per_buffer,
                           const PaStreamCallbackTimeInfo* time_info,
                           PaStreamCallbackFlags status_flags,
                           void* user_data);
    
    void processAudioFrame(const float* input, unsigned long frame_count);
    void recordingThread();
    
    Config config_;
    PaStream* stream_;
    std::atomic<bool> is_recording_;
    std::atomic<bool> speech_detected_;
    std::atomic<bool> should_stop_;
    
    std::vector<float> audio_buffer_;
    std::vector<float> pre_speech_buffer_;
    std::mutex buffer_mutex_;
    
    std::thread recording_thread_;
    AudioCallback vad_callback_;
    VADDetector* vad_detector_;
    
    std::chrono::steady_clock::time_point last_speech_time_;
    std::chrono::steady_clock::time_point recording_start_time_;
    
    // For thread-safe access to recorded data
    std::vector<float> last_recording_;
    std::mutex recording_mutex_;
    std::condition_variable recording_cv_;
    
    // VAD related
    bool useEnergyVAD() const { return config_.vad_type == "energy"; }
    bool useSileroVAD() const { return config_.vad_type == "silero"; }
    float computeEnergyVAD(const float* input, unsigned long frame_count);
    float computeSileroVAD(const std::vector<float>& audio_chunk);
    
    // VAD buffer for Silero VAD (moved from static variable)
    std::vector<float> vad_buffer_;
    
    // VAD smoothing (for energy-based VAD): require N consecutive frames to switch state
    int vad_speech_count_ = 0;
    int vad_silence_count_ = 0;
    int vad_required_consecutive_frames_ = 3;
};
