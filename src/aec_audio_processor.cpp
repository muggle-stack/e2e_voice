#include "aec_audio_processor.hpp"
#include "vad_detector.hpp"

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

// WebRTC includes - must be after other includes to avoid conflicts
#include <webrtc/modules/audio_processing/include/audio_processing.h>

// Frame size for WebRTC APM (10ms at 48kHz = 480 samples)
constexpr int kFrameSizeMs = 10;
constexpr int kSampleRate = 48000;
constexpr int kFrameSize = kSampleRate * kFrameSizeMs / 1000;  // 480 samples

AECAudioProcessor::AECAudioProcessor()
    : AECAudioProcessor(Config()) {}

AECAudioProcessor::AECAudioProcessor(const Config& config)
    : config_(config)
    , apm_(nullptr)
    , stream_(nullptr)
    , is_running_(false)
    , is_playing_(false)
    , vad_detector_(nullptr)
    , speech_detected_(false)
    , recording_active_(false)
    , vad_speech_count_(0)
    , vad_silence_count_(0)
    , barge_in_triggered_(false)
    , barge_in_count_(0) {

    // Pre-allocate buffers with default size
    // Will be resized in initialize() if needed for larger buffer on Linux
    input_int16_.resize(kFrameSize);
    output_int16_.resize(kFrameSize);
    processed_float_.resize(kFrameSize);
    mono_buffer_.resize(kFrameSize);
}

AECAudioProcessor::~AECAudioProcessor() {
    cleanup();
}

bool AECAudioProcessor::initialize() {
    std::cout << "[AEC] Initializing AEC Audio Processor..." << std::endl;
    std::cout << "[AEC] Sample rate: " << config_.sample_rate << " Hz" << std::endl;
    std::cout << "[AEC] Frame size: " << config_.frames_per_buffer << " samples" << std::endl;

    // Ensure sample rate is 48kHz for optimal AEC performance
    if (config_.sample_rate != 48000) {
        std::cerr << "[AEC] Warning: Sample rate should be 48000 Hz for best AEC performance" << std::endl;
        config_.sample_rate = 48000;
    }

    // Initialize WebRTC APM
    rtc::scoped_refptr<webrtc::AudioProcessing> apm_ref =
        webrtc::AudioProcessingBuilder().Create();

    if (!apm_ref) {
        std::cerr << "[AEC] Failed to create AudioProcessing instance" << std::endl;
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

    // Gain control
    if (config_.agc_enabled) {
        apm_config.gain_controller1.enabled = true;
        apm_config.gain_controller1.mode =
            webrtc::AudioProcessing::Config::GainController1::kAdaptiveDigital;
        apm_config.gain_controller2.enabled = true;
    }

    // High-pass filter (removes DC offset and low-frequency noise)
    apm_config.high_pass_filter.enabled = config_.highpass_enabled;

    // Noise suppression
    if (config_.ns_enabled) {
        apm_config.noise_suppression.enabled = true;
        apm_config.noise_suppression.level =
            webrtc::AudioProcessing::Config::NoiseSuppression::kModerate;
    }

    apm_->ApplyConfig(apm_config);

    std::cout << "[AEC] WebRTC APM configured:" << std::endl;
    std::cout << "[AEC]   Echo Cancellation: " << (config_.aec_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "[AEC]   Noise Suppression: " << (config_.ns_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "[AEC]   Gain Control: " << (config_.agc_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "[AEC]   High-pass Filter: " << (config_.highpass_enabled ? "ON" : "OFF") << std::endl;

    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "[AEC] Failed to initialize PortAudio: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    // Get device info
    int numDevices = Pa_GetDeviceCount();
    std::cout << "[AEC] Available audio devices: " << numDevices << std::endl;

    // Set up input device
    PaStreamParameters inputParams;
    inputParams.device = (config_.input_device_index >= 0) ?
                         config_.input_device_index : Pa_GetDefaultInputDevice();
    if (inputParams.device == paNoDevice) {
        std::cerr << "[AEC] No default input device found" << std::endl;
        return false;
    }

    // Check device capabilities for multi-channel
    const PaDeviceInfo* inputDeviceInfo = Pa_GetDeviceInfo(inputParams.device);
    int maxInputChannels = inputDeviceInfo->maxInputChannels;

    // Validate channel configuration
    if (config_.input_channels > maxInputChannels) {
        std::cerr << "[AEC] Requested " << config_.input_channels
                  << " channels but device only supports " << maxInputChannels << std::endl;
        config_.input_channels = maxInputChannels;
    }
    if (config_.selected_channel >= config_.input_channels && config_.selected_channel != -1) {
        std::cerr << "[AEC] Selected channel " << config_.selected_channel
                  << " out of range, using channel 0" << std::endl;
        config_.selected_channel = 0;
    }

    inputParams.channelCount = config_.input_channels;
    inputParams.sampleFormat = paFloat32;
    // Use higher latency on Linux to avoid ALSA issues
#ifdef __linux__
    inputParams.suggestedLatency = inputDeviceInfo->defaultHighInputLatency;
#else
    inputParams.suggestedLatency = inputDeviceInfo->defaultLowInputLatency;
#endif
    inputParams.hostApiSpecificStreamInfo = nullptr;

    std::cout << "[AEC] Input device: " << inputDeviceInfo->name << std::endl;
    std::cout << "[AEC] Input latency: " << inputParams.suggestedLatency * 1000 << " ms" << std::endl;
    std::cout << "[AEC] Input channels: " << config_.input_channels
              << " (max: " << maxInputChannels << ")" << std::endl;
    if (config_.selected_channel == -1) {
        std::cout << "[AEC] Channel mode: mix all channels" << std::endl;
    } else {
        std::cout << "[AEC] Selected channel: " << config_.selected_channel << std::endl;
    }

    // Set up output device
    PaStreamParameters outputParams;
    outputParams.device = (config_.output_device_index >= 0) ?
                          config_.output_device_index : Pa_GetDefaultOutputDevice();
    if (outputParams.device == paNoDevice) {
        std::cerr << "[AEC] No default output device found" << std::endl;
        return false;
    }
    outputParams.channelCount = config_.channels;
    outputParams.sampleFormat = paFloat32;
    // Use higher latency on Linux to avoid ALSA underruns
#ifdef __linux__
    outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultHighOutputLatency;
#else
    outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
#endif
    outputParams.hostApiSpecificStreamInfo = nullptr;

    std::cout << "[AEC] Output device: " << Pa_GetDeviceInfo(outputParams.device)->name << std::endl;
    std::cout << "[AEC] Output latency: " << outputParams.suggestedLatency * 1000 << " ms" << std::endl;

    // On Linux, use larger buffer size to avoid ALSA underruns
    int frames_per_buffer = config_.frames_per_buffer;
#ifdef __linux__
    frames_per_buffer = std::max(frames_per_buffer, 960);  // At least 20ms at 48kHz
    std::cout << "[AEC] Using larger buffer for Linux: " << frames_per_buffer << " samples" << std::endl;
#endif

    // Resize buffers if needed for larger frame size
    if (frames_per_buffer > static_cast<int>(input_int16_.size())) {
        input_int16_.resize(frames_per_buffer);
        output_int16_.resize(frames_per_buffer);
        processed_float_.resize(frames_per_buffer);
        mono_buffer_.resize(frames_per_buffer);
        std::cout << "[AEC] Resized internal buffers to " << frames_per_buffer << " samples" << std::endl;
    }

    // Open full-duplex stream
    err = Pa_OpenStream(&stream_,
                        &inputParams,
                        &outputParams,
                        config_.sample_rate,
                        frames_per_buffer,
                        paClipOff,  // Don't clip samples
                        paCallback,
                        this);

    if (err != paNoError) {
        std::cerr << "[AEC] Failed to open stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    std::cout << "[AEC] Full-duplex audio stream opened successfully" << std::endl;
    return true;
}

void AECAudioProcessor::cleanup() {
    stop();

    if (stream_) {
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }

    Pa_Terminate();

    if (apm_) {
        apm_->Release();
        apm_ = nullptr;
    }
}

bool AECAudioProcessor::start() {
    if (is_running_) return true;

    if (!stream_) {
        std::cerr << "[AEC] Stream not initialized" << std::endl;
        return false;
    }

    PaError err = Pa_StartStream(stream_);
    if (err != paNoError) {
        std::cerr << "[AEC] Failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    is_running_ = true;
    std::cout << "[AEC] Audio stream started" << std::endl;
    return true;
}

void AECAudioProcessor::stop() {
    if (!is_running_) return;

    is_running_ = false;
    recording_active_ = false;

    if (stream_) {
        Pa_StopStream(stream_);
    }

    // Notify any waiting threads
    record_cv_.notify_all();

    std::cout << "[AEC] Audio stream stopped" << std::endl;
}

int AECAudioProcessor::paCallback(const void* input, void* output,
                                   unsigned long frames_per_buffer,
                                   const PaStreamCallbackTimeInfo* time_info,
                                   PaStreamCallbackFlags status_flags,
                                   void* user_data) {
    auto* processor = static_cast<AECAudioProcessor*>(user_data);

    const float* in = static_cast<const float*>(input);
    float* out = static_cast<float*>(output);

    processor->processAudioFrame(in, out, frames_per_buffer);

    return paContinue;
}

void AECAudioProcessor::processAudioFrame(const float* input, float* output,
                                           unsigned long frames) {
    // Prepare stream config for WebRTC
    webrtc::StreamConfig stream_config(config_.sample_rate, config_.channels);

    // Step 1: Process playback (reverse stream for AEC reference)
    // Hold mutex for entire playback processing to avoid race conditions
    {
        std::lock_guard<std::mutex> lock(playback_mutex_);

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
        bool has_playback = !current_playback_.samples.empty() &&
                           current_playback_.position < current_playback_.samples.size();

        if (has_playback) {
            is_playing_ = true;

            size_t samples_available = current_playback_.samples.size() - current_playback_.position;
            size_t samples_to_use = std::min(static_cast<size_t>(frames), samples_available);

            // Convert to int16 for APM
            for (size_t i = 0; i < samples_to_use; ++i) {
                float sample = current_playback_.samples[current_playback_.position + i];
                sample = std::clamp(sample, -1.0f, 1.0f);
                output_int16_[i] = static_cast<int16_t>(sample * 32767.0f);
            }

            // Pad with zeros if needed
            for (size_t i = samples_to_use; i < frames; ++i) {
                output_int16_[i] = 0;
            }

            // Process reverse stream (playback reference for AEC)
            if (apm_) {
                apm_->ProcessReverseStream(output_int16_.data(), stream_config,
                                           stream_config, output_int16_.data());
            }

            // Copy to output
            for (size_t i = 0; i < frames; ++i) {
                output[i] = static_cast<float>(output_int16_[i]) / 32768.0f;
            }

            current_playback_.position += samples_to_use;
        } else {
            is_playing_ = false;

            // No playback, output silence
            std::memset(output, 0, frames * sizeof(float));

            // Still need to process reverse stream with silence for AEC state
            if (apm_) {
                std::memset(output_int16_.data(), 0, frames * sizeof(int16_t));
                apm_->ProcessReverseStream(output_int16_.data(), stream_config,
                                           stream_config, output_int16_.data());
            }
        }
    }

    if (apm_) {

        // Step 2: Process capture stream (microphone with echo cancellation)
        if (input) {
            // Extract mono from multi-channel input
            const float* mono_input = input;
            if (config_.input_channels > 1) {
                // Multi-channel input - extract or mix
                for (size_t i = 0; i < frames; ++i) {
                    if (config_.selected_channel == -1) {
                        // Mix all channels
                        float sum = 0.0f;
                        for (int ch = 0; ch < config_.input_channels; ++ch) {
                            sum += input[i * config_.input_channels + ch];
                        }
                        mono_buffer_[i] = sum / config_.input_channels;
                    } else {
                        // Select specific channel
                        mono_buffer_[i] = input[i * config_.input_channels + config_.selected_channel];
                    }
                }
                mono_input = mono_buffer_.data();
            }

            // Convert input to int16
            for (size_t i = 0; i < frames; ++i) {
                float sample = mono_input[i];
                sample = std::clamp(sample, -1.0f, 1.0f);
                input_int16_[i] = static_cast<int16_t>(sample * 32767.0f);
            }

            // Process stream (applies AEC, NS, AGC)
            apm_->ProcessStream(input_int16_.data(), stream_config,
                               stream_config, input_int16_.data());

            // Convert back to float
            for (size_t i = 0; i < frames; ++i) {
                processed_float_[i] = static_cast<float>(input_int16_[i]) / 32768.0f;
            }

            // VAD processing and barge-in detection
            // Only pass the valid frames to VAD
            std::vector<float> vad_input(processed_float_.begin(), processed_float_.begin() + frames);
            float vad_prob = computeVAD(vad_input);
            updateVADState(vad_prob);

            // Check for barge-in (user speaking while playback is active)
            if (config_.barge_in_enabled && is_playing_ && speech_detected_) {
                barge_in_count_++;
                if (barge_in_count_ >= config_.barge_in_frames) {
                    if (!barge_in_triggered_) {
                        barge_in_triggered_ = true;
                        std::cout << "[AEC] Barge-in detected!" << std::endl;

                        // Clear playback queue
                        {
                            std::lock_guard<std::mutex> lock(playback_mutex_);
                            while (!playback_queue_.empty()) {
                                playback_queue_.pop();
                            }
                            current_playback_.samples.clear();
                            current_playback_.position = 0;
                        }

                        // Call callback
                        if (barge_in_callback_) {
                            barge_in_callback_();
                        }
                    }
                }
            } else {
                barge_in_count_ = 0;
            }

            // Store echo-cancelled audio if recording is active
            if (recording_active_) {
                std::lock_guard<std::mutex> lock(record_mutex_);
                record_buffer_.insert(record_buffer_.end(),
                                     processed_float_.begin(),
                                     processed_float_.begin() + frames);
                record_cv_.notify_one();
            }

            // Call VAD callback
            if (vad_callback_ && speech_detected_) {
                std::vector<float> audio_chunk(processed_float_.begin(),
                                               processed_float_.begin() + frames);
                vad_callback_(audio_chunk);
            }
        }
    } else {
        // No APM - just store raw input if recording
        // (Playback is already handled above in the unified playback section)
        if (recording_active_ && input) {
            std::lock_guard<std::mutex> lock(record_mutex_);
            record_buffer_.insert(record_buffer_.end(), input, input + frames);
            record_cv_.notify_one();
        }
    }
}

float AECAudioProcessor::computeVAD(const std::vector<float>& audio) {
    if (config_.vad_type == "silero" && vad_detector_) {
        // Silero VAD needs 512 samples at 16kHz
        // We're at 48kHz, so we need to downsample and accumulate

        // Downsample 3:1 (48kHz -> 16kHz)
        for (size_t i = 0; i < audio.size(); i += 3) {
            vad_accumulator_.push_back(audio[i]);
        }

        // Process when we have enough samples (512 at 16kHz)
        if (vad_accumulator_.size() >= 512) {
            std::vector<float> vad_chunk(vad_accumulator_.begin(),
                                         vad_accumulator_.begin() + 512);
            vad_accumulator_.erase(vad_accumulator_.begin(),
                                   vad_accumulator_.begin() + 512);

            return vad_detector_->detectVAD(vad_chunk);
        }

        // Not enough samples yet - estimate from energy
        float energy = 0.0f;
        for (float sample : audio) {
            energy += sample * sample;
        }
        energy = std::sqrt(energy / audio.size());
        return std::min(energy * 4.0f, 1.0f);
    } else {
        // Energy-based VAD
        float energy = 0.0f;
        for (float sample : audio) {
            energy += sample * sample;
        }
        energy = std::sqrt(energy / audio.size());
        return std::min(energy * 3.0f, 1.0f);
    }
}

void AECAudioProcessor::updateVADState(float vad_prob) {
    const int required_frames = 3;  // Require N consecutive frames

    if (vad_prob >= config_.vad_trigger_threshold) {
        vad_speech_count_++;
        vad_silence_count_ = 0;

        if (vad_speech_count_ >= required_frames && !speech_detected_) {
            speech_detected_ = true;
            last_speech_time_ = std::chrono::steady_clock::now();
        }
    } else if (vad_prob < config_.vad_stop_threshold) {
        vad_silence_count_++;
        vad_speech_count_ = 0;

        if (speech_detected_) {
            // Check silence duration
            auto now = std::chrono::steady_clock::now();
            auto silence_time = std::chrono::duration<double>(now - last_speech_time_).count();

            if (silence_time >= config_.silence_duration) {
                speech_detected_ = false;
            }
        }
    }

    // Update last speech time if currently speaking
    if (speech_detected_ && vad_prob >= config_.vad_stop_threshold) {
        last_speech_time_ = std::chrono::steady_clock::now();
    }
}

void AECAudioProcessor::enqueuePlayback(const std::vector<float>& samples, int sample_rate) {
    if (samples.empty()) return;

    std::vector<float> resampled;

    // Resample to 48kHz if needed
    if (sample_rate != config_.sample_rate) {
        resampled = resample(samples, sample_rate, config_.sample_rate);
    } else {
        resampled = samples;
    }

    PlaybackBuffer buffer;
    buffer.samples = std::move(resampled);
    buffer.position = 0;

    {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        playback_queue_.push(std::move(buffer));
    }

    std::cout << "[AEC] Enqueued " << samples.size() << " samples for playback "
              << "(resampled from " << sample_rate << " to " << config_.sample_rate << " Hz)"
              << std::endl;
}

void AECAudioProcessor::clearPlaybackQueue() {
    std::lock_guard<std::mutex> lock(playback_mutex_);
    while (!playback_queue_.empty()) {
        playback_queue_.pop();
    }
    current_playback_.samples.clear();
    current_playback_.position = 0;
    is_playing_ = false;

    std::cout << "[AEC] Playback queue cleared" << std::endl;
}

bool AECAudioProcessor::isPlaying() const {
    return is_playing_.load();
}

size_t AECAudioProcessor::getPlaybackQueueSize() const {
    std::lock_guard<std::mutex> lock(playback_mutex_);
    return playback_queue_.size();
}

std::vector<float> AECAudioProcessor::getRecordedAudio() {
    std::lock_guard<std::mutex> lock(record_mutex_);
    std::vector<float> result = std::move(record_buffer_);
    record_buffer_.clear();
    return result;
}

void AECAudioProcessor::clearRecordedAudio() {
    std::lock_guard<std::mutex> lock(record_mutex_);
    record_buffer_.clear();
}

std::vector<float> AECAudioProcessor::recordWithVAD() {
    std::cout << "[AEC] Starting VAD-controlled recording..." << std::endl;

    // Clear previous recording
    clearRecordedAudio();

    // Reset VAD state
    speech_detected_ = false;
    vad_speech_count_ = 0;
    vad_silence_count_ = 0;
    vad_accumulator_.clear();
    barge_in_triggered_ = false;
    barge_in_count_ = 0;

    // Start recording
    recording_active_ = true;
    recording_start_time_ = std::chrono::steady_clock::now();

    std::vector<float> result;
    bool speech_started = false;

    while (recording_active_ && is_running_) {
        // Wait for audio data
        {
            std::unique_lock<std::mutex> lock(record_mutex_);
            record_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !record_buffer_.empty() || !recording_active_;
            });

            if (!recording_active_) break;

            // Copy accumulated audio
            result.insert(result.end(), record_buffer_.begin(), record_buffer_.end());
            record_buffer_.clear();
        }

        // Check if speech has started
        if (speech_detected_ && !speech_started) {
            speech_started = true;
            std::cout << "[AEC] Speech detected, recording..." << std::endl;
        }

        // Check for end conditions
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - recording_start_time_).count();

        // Check max recording time
        if (elapsed >= config_.max_record_time) {
            std::cout << "[AEC] Max recording time reached" << std::endl;
            break;
        }

        // Check if speech ended (silence after speech)
        if (speech_started && !speech_detected_) {
            auto silence_time = std::chrono::duration<double>(now - last_speech_time_).count();
            if (silence_time >= config_.silence_duration) {
                std::cout << "[AEC] Speech ended (silence detected)" << std::endl;
                break;
            }
        }
    }

    recording_active_ = false;

    std::cout << "[AEC] Recording complete: " << result.size() << " samples at "
              << config_.sample_rate << " Hz ("
              << (float)result.size() / config_.sample_rate << "s)" << std::endl;

    return result;
}

// Static utility functions for resampling
std::vector<float> AECAudioProcessor::resample(const std::vector<float>& input,
                                                int from_rate, int to_rate) {
    if (from_rate == to_rate || input.empty()) {
        return input;
    }

    double ratio = static_cast<double>(to_rate) / from_rate;
    size_t output_size = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(output_size);

    // Simple linear interpolation resampling
    // For production, consider using libsamplerate or similar
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

std::vector<int16_t> AECAudioProcessor::floatToInt16(const std::vector<float>& input) {
    std::vector<int16_t> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float sample = std::clamp(input[i], -1.0f, 1.0f);
        output[i] = static_cast<int16_t>(sample * 32767.0f);
    }
    return output;
}

std::vector<float> AECAudioProcessor::int16ToFloat(const std::vector<int16_t>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]) / 32768.0f;
    }
    return output;
}
