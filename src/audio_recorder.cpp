#include "audio_recorder.hpp"
#include "vad_detector.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>

AudioRecorder::AudioRecorder()
    : config_(Config{}), stream_(nullptr), is_recording_(false), 
      speech_detected_(false), should_stop_(false), vad_detector_(nullptr) {
}

AudioRecorder::AudioRecorder(const Config& config)
    : config_(config), stream_(nullptr), is_recording_(false), 
      speech_detected_(false), should_stop_(false), vad_detector_(nullptr) {
}

AudioRecorder::~AudioRecorder() {
    cleanup();
}

// ===================== Helpers: resampling =====================
namespace {
    // Linear resample a single-channel sequence
    std::vector<float> resampleMono(const std::vector<float>& input,
                                    int input_rate,
                                    int output_rate) {
        if (input_rate == output_rate || input.empty()) {
            return input;
        }
        double ratio = static_cast<double>(input_rate) / static_cast<double>(output_rate);
        size_t output_length = static_cast<size_t>(std::floor(static_cast<double>(input.size()) / ratio));
        std::vector<float> output;
        output.resize(output_length);

        for (size_t i = 0; i < output_length; ++i) {
            double src_pos = static_cast<double>(i) * ratio;
            size_t idx = static_cast<size_t>(src_pos);
            double frac = src_pos - static_cast<double>(idx);
            if (idx + 1 < input.size()) {
                float a = input[idx];
                float b = input[idx + 1];
                output[i] = static_cast<float>((1.0 - frac) * a + frac * b);
            } else {
                output[i] = input[idx];
            }
        }
        return output;
    }

    // Resample interleaved audio by splitting channels, resampling per channel, then re-interleaving
    std::vector<float> resampleInterleaved(const std::vector<float>& interleaved,
                                           int channels,
                                           int input_rate,
                                           int output_rate) {
        if (channels <= 0) {
            return interleaved;
        }
        if (input_rate == output_rate || interleaved.empty()) {
            return interleaved;
        }

        size_t input_frames = interleaved.size() / static_cast<size_t>(channels);
        if (input_frames == 0) {
            return {};
        }

        // Split channels
        std::vector<std::vector<float>> ch_data(static_cast<size_t>(channels));
        for (int c = 0; c < channels; ++c) {
            ch_data[c].reserve(input_frames);
        }
        for (size_t f = 0; f < input_frames; ++f) {
            for (int c = 0; c < channels; ++c) {
                ch_data[c].push_back(interleaved[f * channels + c]);
            }
        }

        // Resample each channel
        for (int c = 0; c < channels; ++c) {
            ch_data[c] = resampleMono(ch_data[c], input_rate, output_rate);
        }

        // Re-interleave
        size_t output_frames = ch_data[0].size();
        std::vector<float> output;
        output.resize(output_frames * static_cast<size_t>(channels));
        for (size_t f = 0; f < output_frames; ++f) {
            for (int c = 0; c < channels; ++c) {
                float sample = (f < ch_data[c].size()) ? ch_data[c][f] : ch_data[c].back();
                output[f * channels + c] = sample;
            }
        }
        return output;
    }
}

bool AudioRecorder::initialize() {
    // Suppress ALSA error messages
    FILE* null_file = std::freopen("/dev/null", "w", stderr);
    (void)null_file; // Suppress unused warning
    
    PaError err = Pa_Initialize();
    
    // Restore stderr
    FILE* tty_file = std::freopen("/dev/tty", "w", stderr);
    (void)tty_file; // Suppress unused warning
    
    if (err != paNoError) {
        std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    // Print available devices for debugging
    int numDevices = Pa_GetDeviceCount();
    std::cout << "Available audio devices:" << std::endl;
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        std::cout << "Device " << i << ": " << deviceInfo->name 
                  << " (inputs: " << deviceInfo->maxInputChannels << ")" << std::endl;
    }

    // Set up stream parameters
    PaStreamParameters inputParameters;
    inputParameters.device = (config_.device_index >= 0) ? 
        config_.device_index : Pa_GetDefaultInputDevice();
    
    if (inputParameters.device == paNoDevice) {
        std::cerr << "No default input device available." << std::endl;
        return false;
    }

    inputParameters.channelCount = config_.channels;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    // Open stream
    err = Pa_OpenStream(&stream_,
                        &inputParameters,
                        nullptr, // no output
                        config_.sample_rate,
                        config_.frames_per_buffer,
                        paClipOff,
                        audioCallback,
                        this);

    if (err != paNoError) {
        std::cerr << "Failed to open stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    return true;
}

void AudioRecorder::cleanup() {
    if (is_recording_.load()) {
        stopRecording();
    }

    if (stream_) {
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }
    
    Pa_Terminate();
}

int AudioRecorder::audioCallback(const void* input_buffer, void* output_buffer,
                                unsigned long frames_per_buffer,
                                const PaStreamCallbackTimeInfo* time_info,
                                PaStreamCallbackFlags status_flags,
                                void* user_data) {
    AudioRecorder* recorder = static_cast<AudioRecorder*>(user_data);
    const float* input = static_cast<const float*>(input_buffer);
    
    if (input) {
        recorder->processAudioFrame(input, frames_per_buffer);
    }
    
    return recorder->should_stop_.load() ? paComplete : paContinue;
}

void AudioRecorder::processAudioFrame(const float* input, unsigned long frame_count) {
    // Process audio without holding the lock for the entire duration
    // Convert to vector for easier processing
    std::vector<float> frame(input, input + frame_count * config_.channels);
    
    // First, calculate the signal level to determine amplification needed
    float max_raw_amplitude = 0.0f;
    for (unsigned long i = 0; i < frame_count * config_.channels; ++i) {
        max_raw_amplitude = std::max(max_raw_amplitude, std::abs(frame[i]));
    }
    
    // Adaptive amplification based on signal level (applied to both VAD and ASR)
    float vad_amplification = 20.0f;  // Base amplification
    if (max_raw_amplitude < 0.001f) {
        vad_amplification = 50.0f;  // Very weak signal, amplify more
    } else if (max_raw_amplitude < 0.01f) {
        vad_amplification = 30.0f;  // Weak signal
    } else if (max_raw_amplitude < 0.1f) {
        vad_amplification = 20.0f;  // Normal signal
    } else {
        vad_amplification = 10.0f;  // Strong signal, less amplification needed
    }
    
    // Apply amplification to the frame
    std::vector<float> amplified_frame = frame;
    for (auto& sample : amplified_frame) {
        sample *= vad_amplification;
        sample = std::max(-1.0f, std::min(1.0f, sample));  // Clip to prevent overflow
    }
    
    // For VAD, use amplified audio converted to mono if stereo
    std::vector<float> vad_source_mono;
    vad_source_mono.reserve(frame_count);
    if (config_.channels == 2) {
        // Average stereo to mono (already amplified)
        for (unsigned long i = 0; i < frame_count; ++i) {
            float left = amplified_frame[i * 2];
            float right = amplified_frame[i * 2 + 1];
            vad_source_mono.push_back((left + right) / 2.0f);
        }
    } else {
        // Already mono (and amplified)
        vad_source_mono.assign(amplified_frame.begin(), amplified_frame.end());
    }
    
    // Lock only when modifying shared buffers
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        // Add amplified audio to pre-speech buffer (circular buffer)
        // Use amplified audio for better ASR quality
        if (pre_speech_buffer_.size() > config_.frames_per_buffer * 10 * config_.channels) {
            pre_speech_buffer_.erase(pre_speech_buffer_.begin(), 
                                    pre_speech_buffer_.begin() + config_.frames_per_buffer * config_.channels);
        }
        pre_speech_buffer_.insert(pre_speech_buffer_.end(), amplified_frame.begin(), amplified_frame.end());
    }
    
    // Choose VAD method based on configuration
    bool is_speech = false;
    float vad_prob = 0.0f;
    
    if (useEnergyVAD()) {
        // Energy-based VAD - use mono frame
        vad_prob = computeEnergyVAD(vad_source_mono.data(), vad_source_mono.size());
        bool frame_is_speech = vad_prob > config_.trigger_threshold;

        // 3-frame hysteresis smoothing
        const int N = vad_required_consecutive_frames_;
        if (frame_is_speech) {
            vad_speech_count_ = std::min(vad_speech_count_ + 1, N);
            vad_silence_count_ = 0;
        } else {
            vad_silence_count_ = std::min(vad_silence_count_ + 1, N);
            vad_speech_count_ = 0;
        }

        if (!speech_detected_.load()) {
            // require N consecutive speech frames to start
            is_speech = (vad_speech_count_ >= N);
        } else {
            // once started, require N consecutive silence frames to stop
            is_speech = (vad_silence_count_ < N);
        }

        // Debug output disabled for cleaner interface
        /*
        static int frame_counter = 0;
        if (++frame_counter % 10 == 0) {
            std::cout << "[VAD] Energy: " << vad_prob
                      << ", speech_cnt=" << vad_speech_count_
                      << ", silence_cnt=" << vad_silence_count_
                      << " (thr: " << config_.trigger_threshold << ")" << std::endl;
        }
        */
    } else if (useSileroVAD() && vad_detector_) {
        // Silero VAD - use mono frame
        vad_prob = computeSileroVAD(vad_source_mono);
        is_speech = vad_prob > config_.trigger_threshold;
        
    } else {
        // Fallback to energy VAD if Silero not available - use mono frame
        vad_prob = computeEnergyVAD(vad_source_mono.data(), vad_source_mono.size());
        is_speech = vad_prob > config_.trigger_threshold;
    }
    
    // Call external VAD callback if available
    if (vad_callback_) {
        vad_callback_(frame);
    }
    
    auto now = std::chrono::steady_clock::now();
    
    if (is_speech) {
        last_speech_time_ = now;
        if (!speech_detected_.load()) {
            speech_detected_.store(true);
            std::cout << "▶ Speech detected, starting recording..." << std::endl;
            // Add pre-speech buffer to main buffer
            audio_buffer_.insert(audio_buffer_.end(), 
                               pre_speech_buffer_.begin(), pre_speech_buffer_.end());
        }
    }
    
    if (speech_detected_.load()) {
        // Store amplified audio for better ASR quality
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        audio_buffer_.insert(audio_buffer_.end(), amplified_frame.begin(), amplified_frame.end());
        
        // Check stopping conditions
        auto silence_duration = std::chrono::duration<double>(now - last_speech_time_).count();
        auto total_duration = std::chrono::duration<double>(now - recording_start_time_).count();
        
        if (silence_duration > config_.silence_duration) {
            std::cout << "⏹ Silence detected, stopping recording" << std::endl;
            should_stop_.store(true);
        } else if (total_duration > config_.max_record_time) {
            std::cout << "⏹ Max recording time reached, stopping recording" << std::endl;
            should_stop_.store(true);
        }
    }
}

std::vector<float> AudioRecorder::recordAudio() {
    if (!stream_) {
        std::cerr << "Stream not initialized" << std::endl;
        return {};
    }

    // Reset state
    audio_buffer_.clear();
    pre_speech_buffer_.clear();
    vad_buffer_.clear(); // Clear VAD buffer for new recording
    
    // Reset VAD state counters (for energy VAD)
    vad_speech_count_ = 0;
    vad_silence_count_ = 0;
    
    speech_detected_.store(false);
    should_stop_.store(false);
    recording_start_time_ = std::chrono::steady_clock::now();
    last_speech_time_ = recording_start_time_;
    
    // Reset VAD detector state if using Silero VAD
    if (vad_detector_) {
        vad_detector_->reset();
    }

    // Start stream
    PaError err = Pa_StartStream(stream_);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        return {};
    }

    // Wait for recording to complete
    while (!should_stop_.load()) {
        Pa_Sleep(100); // Sleep 100ms
    }

    // Stop stream
    err = Pa_StopStream(stream_);
    if (err != paNoError) {
        std::cerr << "Failed to stop stream: " << Pa_GetErrorText(err) << std::endl;
    }

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return audio_buffer_;
}

void AudioRecorder::startRecording() {
    if (is_recording_.load()) {
        return;
    }

    is_recording_.store(true);
    recording_thread_ = std::thread(&AudioRecorder::recordingThread, this);
}

void AudioRecorder::stopRecording() {
    if (!is_recording_.load()) {
        return;
    }

    should_stop_.store(true);
    is_recording_.store(false);
    
    if (recording_thread_.joinable()) {
        recording_thread_.join();
    }
}

void AudioRecorder::recordingThread() {
    auto result = recordAudio();
    
    std::unique_lock<std::mutex> lock(recording_mutex_);
    last_recording_ = std::move(result);
    lock.unlock();
    
    recording_cv_.notify_all();
}

std::vector<float> AudioRecorder::getLastRecording() {
    std::unique_lock<std::mutex> lock(recording_mutex_);
    recording_cv_.wait(lock, [this] { return !is_recording_.load(); });
    return last_recording_;
}

float AudioRecorder::computeEnergyVAD(const float* input, unsigned long frame_count) {
    // Calculate RMS energy
    float energy = 0.0f;
    float max_sample = 0.0f;
    
    for (unsigned long i = 0; i < frame_count; ++i) {
        energy += input[i] * input[i];
        max_sample = std::max(max_sample, std::abs(input[i]));
    }
    energy = std::sqrt(energy / frame_count);
    
    // Convert energy to probability-like value (0-1)
    // Adjust these thresholds based on your audio environment
    // Much lower thresholds for dual-mic setup with amplification
    const float min_energy = 0.0000005f;  // Even lower threshold for better sensitivity
    const float max_energy = 0.005f;       // Lower max for better dynamic range
    
    if (energy < min_energy) return 0.0f;
    if (energy > max_energy) return 1.0f;
    
    // Linear mapping to 0-1 range
    return (energy - min_energy) / (max_energy - min_energy);
}

float AudioRecorder::computeSileroVAD(const std::vector<float>& audio_chunk) {
    if (!vad_detector_) {
        return 0.0f;
    }
    
    // Accumulate audio chunks for Silero VAD (using member variable)
    static const size_t VAD_WINDOW_SIZE = 512; // 32ms at 16kHz
    
    // Resample audio_chunk to 16kHz if needed
    std::vector<float> resampled_chunk;
    int effective_rate = config_.sample_rate;
    
    if (effective_rate == 16000) {
        resampled_chunk = audio_chunk;
    } else {
        // Use the high-quality resampleMono function with linear interpolation
        resampled_chunk = resampleMono(audio_chunk, effective_rate, 16000);
    }
    
    // Add resampled chunk to buffer
    if (!resampled_chunk.empty()) {
        vad_buffer_.insert(vad_buffer_.end(), resampled_chunk.begin(), resampled_chunk.end());
    }
    
    // Process when we have enough data
    if (vad_buffer_.size() >= VAD_WINDOW_SIZE) {
        // Use the latest VAD_WINDOW_SIZE samples
        std::vector<float> vad_input(vad_buffer_.end() - VAD_WINDOW_SIZE, vad_buffer_.end());
        
        // Keep only recent data in buffer (sliding window)
        if (vad_buffer_.size() > VAD_WINDOW_SIZE * 2) {
            vad_buffer_.erase(vad_buffer_.begin(), vad_buffer_.end() - VAD_WINDOW_SIZE);
        }
        
        // Debug: Check audio amplitude before sending to VAD
        float max_amp = 0.0f;
        float avg_amp = 0.0f;
        for (const auto& s : vad_input) {
            float abs_s = std::abs(s);
            max_amp = std::max(max_amp, abs_s);
            avg_amp += abs_s;
        }
        avg_amp /= vad_input.size();
        
        // Print warning if audio is too quiet (adjusted for amplified signal)
        static int warn_counter = 0;
        // Reset counter for new recording
        if (vad_buffer_.size() <= VAD_WINDOW_SIZE * 2) {
            warn_counter = 0;
        }
        if (max_amp < 0.01f && ++warn_counter % 10 == 0) {
            std::cout << "[SileroVAD] Warning: Input audio very quiet (max=" << max_amp 
                      << ", avg=" << avg_amp << ")" << std::endl;
        }
        
        // Use Silero VAD detector
        float prob = vad_detector_->detectVAD(vad_input);
        return prob;
    }
    
    // Not enough data yet, return low probability
    return 0.0f;
}
