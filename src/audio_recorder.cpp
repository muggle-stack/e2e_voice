#include "audio_recorder.hpp"
#include "vad_detector.hpp"
#ifdef WITH_SPEAKER_RECOGNITION
#include "speaker_recognition.hpp"
#include "speaker_model_downloader.hpp"
#endif
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

    // Initialize speaker recognition if enabled
#ifdef WITH_SPEAKER_RECOGNITION
    if (config_.enable_speaker_recognition) {
        std::cout << "[Speaker Recognition] Enabled, initializing..." << std::endl;
        if (!initializeSpeakerRecognition()) {
            std::cerr << "Warning: Failed to initialize speaker recognition" << std::endl;
            // Continue without speaker recognition
            config_.enable_speaker_recognition = false;
        }
    } else {
        std::cout << "[Speaker Recognition] Disabled" << std::endl;
    }
#endif

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
    
    // Separated processing: Create dedicated high amplification audio for VAD, AFE uses original audio
    // VAD amplification strategy: More aggressive, specifically for far-field detection
    float vad_amplification = 100.0f;  // Base amplification significantly increased
    if (max_raw_amplitude < 0.0001f) {
        vad_amplification = 300.0f;  // Extremely weak signal, super high amplification
    } else if (max_raw_amplitude < 0.001f) {
        vad_amplification = 200.0f;  // Very weak signal, high amplification
    } else if (max_raw_amplitude < 0.01f) {
        vad_amplification = 150.0f;  // Weak signal, medium amplification
    } else if (max_raw_amplitude < 0.1f) {
        vad_amplification = 100.0f;  // Normal signal
    } else {
        vad_amplification = 50.0f;   // Strong signal, moderate amplification
    }
    
    // Create dedicated amplified audio for VAD (allow distortion, only for detection)
    std::vector<float> vad_amplified_frame = frame;
    for (auto& sample : vad_amplified_frame) {
        sample *= vad_amplification;
        sample = std::max(-1.0f, std::min(1.0f, sample));  // Hard clipping, allow distortion
    }
    
    // Create moderately amplified audio for recording (maintain quality)
    float recording_amplification = 20.0f;  // Moderate amplification for recording
    if (max_raw_amplitude < 0.001f) {
        recording_amplification = 40.0f;
    } else if (max_raw_amplitude < 0.01f) {
        recording_amplification = 30.0f;
    } else if (max_raw_amplitude < 0.1f) {
        recording_amplification = 20.0f;
    } else {
        recording_amplification = 15.0f;
    }
    
    std::vector<float> amplified_frame = frame;
    for (auto& sample : amplified_frame) {
        sample *= recording_amplification;
        sample = std::max(-1.0f, std::min(1.0f, sample));  // Gentle clipping
    }
    
    // VAD dedicated audio processing: Force use of highly amplified audio, ignore AFE complex logic
    std::vector<float> vad_source_mono;
    
    // Simplified logic: VAD always uses current frame's highly amplified audio for maximum sensitivity
    vad_source_mono.reserve(frame_count);
    
    // Safety check: ensure vad_amplified_frame has enough data
    size_t expected_size = frame_count * config_.channels;
    if (vad_amplified_frame.size() != expected_size) {
        std::cerr << "[VAD] Error: vad_amplified_frame size mismatch. Expected: " 
                  << expected_size << ", Got: " << vad_amplified_frame.size() << std::endl;
        // Can't return directly! Continue processing, but skip VAD processing
        vad_source_mono.clear(); // Clear VAD data, but continue other processing
    } else {
        if (config_.channels == 2) {
            // Average stereo to mono (using VAD dedicated highly amplified version)
            // Safer boundary checking: ensure no out-of-range memory access
            size_t max_frames = vad_amplified_frame.size() / 2;  // Maximum processable frames
            size_t actual_frames = std::min(static_cast<size_t>(frame_count), max_frames);
            
            vad_source_mono.reserve(actual_frames);
            for (size_t i = 0; i < actual_frames; ++i) {
                size_t left_idx = i * 2;
                size_t right_idx = i * 2 + 1;
                // Double protection: check both calculation result and actual size
                if (right_idx < vad_amplified_frame.size()) {
                    float left = vad_amplified_frame[left_idx];
                    float right = vad_amplified_frame[right_idx];
                    vad_source_mono.push_back((left + right) / 2.0f);
                } else {
                    break;  // Stop immediately if boundary issues encountered
                }
            }
        } else {
            // Already mono (using VAD dedicated highly amplified version)
            // Ensure mono data size is correct
            if (vad_amplified_frame.size() == frame_count) {
                vad_source_mono.assign(vad_amplified_frame.begin(), vad_amplified_frame.end());
            } else {
                std::cerr << "[VAD] Warning: mono frame size mismatch, expected: " << frame_count 
                          << ", got: " << vad_amplified_frame.size() << std::endl;
                // Safe copy: only copy available data
                size_t safe_size = std::min(static_cast<size_t>(frame_count), vad_amplified_frame.size());
                vad_source_mono.assign(vad_amplified_frame.begin(), vad_amplified_frame.begin() + safe_size);
            }
        }
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

    // Perform speaker identification if enabled
#ifdef WITH_SPEAKER_RECOGNITION
    if (config_.enable_speaker_recognition && speaker_embedder_ && speaker_manager_) {
        std::cout << "[Speaker Recognition] Starting identification..." << std::endl;
        std::cout << "[Speaker Recognition] Audio buffer size: " << audio_buffer_.size()
                  << " samples (" << (float)audio_buffer_.size() / config_.sample_rate << "s)" << std::endl;

        last_identified_speaker_ = identifySpeaker(audio_buffer_);

        if (!last_identified_speaker_.empty()) {
            std::cout << "[Speaker Recognition] ✓ Identified speaker: " << last_identified_speaker_ << std::endl;
            speaker_registered_ = true;
        } else {
            std::cout << "[Speaker Recognition] ✗ Unknown speaker or no match found" << std::endl;
            std::cout << "[Speaker Recognition] Threshold: " << config_.speaker_threshold << std::endl;
            speaker_registered_ = false;
        }
    } else {
        if (!config_.enable_speaker_recognition) {
            std::cout << "[Speaker Recognition] Not enabled" << std::endl;
        } else if (!speaker_embedder_) {
            std::cout << "[Speaker Recognition] Embedder not initialized" << std::endl;
        } else if (!speaker_manager_) {
            std::cout << "[Speaker Recognition] Manager not initialized" << std::endl;
        }
    }
#endif

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
        // Use dual-channel resampling if we have stereo input, otherwise mono
        if (config_.channels == 1) {
            resampled_chunk = resampleMono(audio_chunk, effective_rate, 16000);
        } else {
            // For stereo input, use proper dual-channel resampling
            resampled_chunk = resampleInterleaved(audio_chunk, config_.channels, effective_rate, 16000);
        }
    }
    
    // Convert stereo to mono for VAD (Silero VAD expects mono input)
    if (config_.channels == 2 && resampled_chunk.size() >= 2) {
        std::vector<float> mono_chunk;
        mono_chunk.reserve(resampled_chunk.size() / 2);
        
        // Average left and right channels
        for (size_t i = 0; i + 1 < resampled_chunk.size(); i += 2) {
            float left = resampled_chunk[i];
            float right = resampled_chunk[i + 1];
            mono_chunk.push_back((left + right) / 2.0f);
        }
        
        resampled_chunk = std::move(mono_chunk);
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

#ifdef WITH_SPEAKER_RECOGNITION
bool AudioRecorder::initializeSpeakerRecognition() {
    if (!config_.enable_speaker_recognition) {
        return true; // Not an error if not enabled
    }

    std::cout << "[Speaker Recognition] Initializing..." << std::endl;

    // Initialize model downloader
    SRModelDownloader downloader;
    if (!downloader.ensureModelsExist()) {
        std::cerr << "[Speaker Recognition] Failed to ensure models exist" << std::endl;
        return false;
    }

    // Create embedder
    speaker_recognition::EmbedderConfig embedder_config;
    embedder_config.model_path = downloader.getModelPath(SRModelDownloader::AR_MODEL_NAME);
    embedder_config.num_threads = 1;
    embedder_config.debug = false;
    embedder_config.provider = "cpu";

    speaker_embedder_ = speaker_recognition::SpeakerEmbedder::Create(embedder_config);
    if (!speaker_embedder_) {
        std::cerr << "[Speaker Recognition] Failed to create speaker embedder" << std::endl;
        return false;
    }

    // Create manager
    speaker_manager_ = speaker_recognition::SpeakerManager::Create(speaker_embedder_->GetEmbeddingDimension());
    if (!speaker_manager_) {
        std::cerr << "[Speaker Recognition] Failed to create speaker manager" << std::endl;
        return false;
    }

    // Load existing database
    if (!config_.speaker_database.empty()) {
        if (speaker_manager_->LoadDatabase(config_.speaker_database)) {
            std::cout << "[Speaker Recognition] Loaded " << speaker_manager_->GetSpeakerCount()
                      << " speakers from " << config_.speaker_database << std::endl;

            // List all speakers
            auto speakers = speaker_manager_->GetAllSpeakers();
            if (!speakers.empty()) {
                std::cout << "[Speaker Recognition] Registered speakers: ";
                for (size_t i = 0; i < speakers.size(); ++i) {
                    std::cout << speakers[i];
                    if (i < speakers.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "[Speaker Recognition] No existing database found or failed to load" << std::endl;
        }
    }

    std::cout << "[Speaker Recognition] Initialization complete" << std::endl;
    return true;
}

std::string AudioRecorder::identifySpeaker(const std::vector<float>& audio) {
    if (!speaker_embedder_ || !speaker_manager_ || audio.empty()) {
        if (!speaker_embedder_) std::cerr << "[Speaker Recognition] Error: Embedder is null" << std::endl;
        if (!speaker_manager_) std::cerr << "[Speaker Recognition] Error: Manager is null" << std::endl;
        if (audio.empty()) std::cerr << "[Speaker Recognition] Error: Audio is empty" << std::endl;
        return "";
    }

    constexpr int32_t kModelSampleRate = 16000;
    int32_t input_sample_rate = config_.sample_rate;

    std::cout << "[Speaker Recognition] Processing " << audio.size() << " samples at "
              << input_sample_rate << "Hz (expected 16000Hz for model)" << std::endl;

    // IMPORTANT: Speaker recognition model expects 16000Hz audio
    // If sample rate is different, resample to match the model requirement
    if (input_sample_rate != kModelSampleRate) {
        std::cout << "[Speaker Recognition] Resampling from " << input_sample_rate
                  << "Hz to " << kModelSampleRate << "Hz" << std::endl;
    }

    // Create a stream and process audio
    auto stream = speaker_embedder_->CreateStream();

    // Convert stereo to mono if needed
    std::vector<float> mono_audio;
    if (config_.channels == 2) {
        mono_audio.reserve(audio.size() / 2);
        for (size_t i = 0; i < audio.size(); i += 2) {
            float left = audio[i];
            float right = (i + 1 < audio.size()) ? audio[i + 1] : left;
            mono_audio.push_back((left + right) / 2.0f);
        }
        std::cout << "[Speaker Recognition] Converted stereo to mono: " << mono_audio.size() << " samples" << std::endl;
    } else {
        mono_audio = audio;
        std::cout << "[Speaker Recognition] Using mono audio: " << mono_audio.size() << " samples" << std::endl;
    }

    if (input_sample_rate != kModelSampleRate) {
        mono_audio = resampleMono(mono_audio, input_sample_rate, kModelSampleRate);
        input_sample_rate = kModelSampleRate;
    }

    stream->AcceptWaveform(input_sample_rate, mono_audio);
    stream->InputFinished();

    if (!speaker_embedder_->IsStreamReady(stream.get())) {
        std::cerr << "[Speaker Recognition] Audio too short for embedding (need at least 0.5s)" << std::endl;
        return "";
    }

    // Compute embedding
    std::cout << "[Speaker Recognition] Computing embedding..." << std::endl;
    auto embedding = speaker_embedder_->ComputeEmbedding(stream.get());
    if (embedding.empty()) {
        std::cerr << "[Speaker Recognition] Failed to compute embedding" << std::endl;
        return "";
    }
    std::cout << "[Speaker Recognition] Embedding computed, dimension: " << embedding.size() << std::endl;

    // Get best matches to show scores
    auto matches = speaker_manager_->GetBestMatches(embedding, 0.0f, 5);
    std::cout << "[Speaker Recognition] Top matches:" << std::endl;
    for (const auto& match : matches) {
        std::cout << "  - " << match.name << ": score=" << match.score
                  << (match.score >= config_.speaker_threshold ? " ✓" : " ✗") << std::endl;
    }

    // Search for speaker
    std::string identified = speaker_manager_->SearchSpeaker(embedding, config_.speaker_threshold);
    if (!identified.empty()) {
        std::cout << "[Speaker Recognition] Match found: " << identified << std::endl;
    } else {
        std::cout << "[Speaker Recognition] No match above threshold " << config_.speaker_threshold << std::endl;
    }

    return identified;
}
#else
// Stub implementations when speaker recognition is disabled
bool AudioRecorder::initializeSpeakerRecognition() {
    return true;
}

std::string AudioRecorder::identifySpeaker(const std::vector<float>& audio) {
    return "";
}
#endif
