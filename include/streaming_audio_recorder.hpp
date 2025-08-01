#ifndef STREAMING_AUDIO_RECORDER_HPP
#define STREAMING_AUDIO_RECORDER_HPP

#include <portaudio.h>
#include "vad_detector.hpp"
#include <queue>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <atomic>
#include <thread>

struct AudioSegment {
    std::vector<float> samples;
    double timestamp_start;
    double timestamp_end;
    size_t segment_id;
    bool is_final;
};

class StreamingAudioRecorder {
public:
    using SegmentCallback = std::function<void(const AudioSegment&)>;
    
    StreamingAudioRecorder(int device_index, int sample_rate);
    ~StreamingAudioRecorder();
    
    // Start streaming session with continuous recording
    void startStreamingSession(
        SegmentCallback callback,
        double max_duration = 60.0,
        double silence_threshold = 0.5,
        double pre_speech_buffer = 0.25
    );
    
    // Stop the streaming session
    void stopStreamingSession();
    
    // Wait for session completion
    void waitForCompletion();
    
    // Set VAD parameters
    void setVADThreshold(float threshold) { vadThreshold = threshold; }
    void setUseEnergyVAD(bool use) { useEnergyVAD = use; }
    void setVADDetector(std::shared_ptr<VADDetector> detector);
    void setVADTriggerThreshold(float threshold) { vadTriggerThreshold = threshold; }
    void setVADStopThreshold(float threshold) { vadStopThreshold = threshold; }

private:
    // VAD state machine
    enum class VADState {
        WAITING_FOR_SPEECH,
        RECORDING_SPEECH,
        SILENCE_DETECTED
    };
    
    // Ring buffer for continuous audio storage
    std::vector<float> ringBuffer;
    size_t ringBufferSize;
    std::atomic<size_t> writePos{0};
    std::atomic<size_t> readPos{0};
    std::mutex bufferMutex;
    
    // VAD state tracking
    VADState vadState{VADState::WAITING_FOR_SPEECH};
    size_t speechStartPos{0};
    size_t silenceFrames{0};
    size_t currentSegmentId{0};
    
    // Session control
    std::atomic<bool> isStreaming{false};
    std::chrono::steady_clock::time_point sessionStartTime;
    double maxSessionDuration{60.0};
    double silenceThresholdSec{0.5};
    double preSpeechBufferSec{0.25};
    
    // Segment processing
    SegmentCallback segmentCallback;
    std::thread processingThread;
    
    // VAD parameters
    float vadThreshold{0.01f};
    bool useEnergyVAD{true};
    std::shared_ptr<VADDetector> sileroVAD;
    float vadTriggerThreshold{0.5f};
    float vadStopThreshold{0.35f};
    
    // Frame processing
    void processAudioFrame(const float* samples, size_t numSamples);
    void processStreamingAudio();
    
    // VAD detection
    bool detectSpeech(const float* samples, size_t numSamples);
    bool detectSpeechEnergy(const float* samples, size_t numSamples);
    bool detectSpeechSilero(const float* samples, size_t numSamples);
    
    // Ring buffer operations
    void addToRingBuffer(const float* samples, size_t numSamples);
    std::vector<float> extractFromRingBuffer(size_t startPos, size_t numSamples);
    size_t getBufferPosition(size_t samplesBack) const;
    size_t calculateBufferDistance(size_t from, size_t to) const;
    
    // Segment creation
    void createAndQueueSegment();
    
    // Session management
    bool checkSessionTimeout() const;
    
    // Audio callback for streaming
    static int streamingAudioCallback(
        const void* inputBuffer,
        void* outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData
    );
    
    // Audio configuration
    int device_index;
    int sample_rate;
    int target_sample_rate{16000};  // ASR expects 16kHz
    PaStream* stream{nullptr};
    
    // Resampling
    std::vector<float> resampleBuffer;
    void resampleAudio(const float* input, size_t inputSamples, std::vector<float>& output);
};

#endif // STREAMING_AUDIO_RECORDER_HPP