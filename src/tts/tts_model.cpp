#include "tts/tts_model.hpp"
#include "audio_processor.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <cstdint>
#include <cstdlib>  // for posix_memalign
#include <chrono>
#include <cctype>   // for isalnum, ispunct
#include <fftw3.h>
#include <regex>
#include <mutex>
#include <unistd.h>  // for dup, dup2, close
#include <fcntl.h>   // for open

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Jieba for Chinese text segmentation
#include "cppjieba/Jieba.hpp"

// cpp-pinyin for Chinese to Pinyin conversion (zh-en bilingual model)
#include <cpp-pinyin/Pinyin.h>
#include <cpp-pinyin/G2pglobal.h>

namespace fs = std::filesystem;

namespace tts {

// Internal helper functions
namespace {

// Forward declarations
std::vector<float> normalizeAudio(const std::vector<float>& audio, const TTSConfig& config);
std::vector<float> removeClicksAndPops(const std::vector<float>& audio);

// Helper function to read custom metadata from ONNX model
std::string LookupCustomModelMetaData(const Ort::ModelMetadata& meta_data,
                                     const std::string& key,
                                     Ort::AllocatorWithDefaultOptions& allocator) {
    try {
        auto result = meta_data.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
        if (result) {
            return std::string(result.get());
        }
    } catch (...) {
        // Ignore exceptions and return empty string
    }
    return "";
}

// Read tokens to ID mapping from file
std::unordered_map<std::string, int64_t> readTokenToIdMap(const std::string& path) {
    std::unordered_map<std::string, int64_t> token_to_id;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open tokens file: " + path);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (!line.empty()) {
            std::istringstream iss(line);
            std::string token;
            int64_t id;

            if (iss >> token >> id) {
                // Format: "token_name token_id"
                token_to_id[token] = id;
            } else {
                // Fallback: use line number as ID (0-indexed)
                token_to_id[line] = line_num - 1;
            }
        }
    }

    return token_to_id;
}

// Read tokens to ID mapping for zh-en model (line number + 1 = ID)
std::unordered_map<std::string, int64_t> readZhEnTokenToIdMap(const std::string& path) {
    std::unordered_map<std::string, int64_t> token_to_id;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open tokens file: " + path);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (!line.empty()) {
            // For zh-en model: line number + 1 = token ID (1-indexed)
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                std::string token = line.substr(start, end - start + 1);
                token_to_id[token] = line_num;  // 1-indexed
            }
        }
    }

    return token_to_id;
}

// Convert IPA to gruut en-us format for zh-en model
std::string convertToGruutEnUs(const std::string& ipa) {
    std::string text = ipa;

    // First, remove zero-width joiner (U+200D) that espeak-ng sometimes adds
    {
        std::string zwj = "\xe2\x80\x8d";  // Zero-width joiner UTF-8
        size_t pos = 0;
        while ((pos = text.find(zwj, pos)) != std::string::npos) {
            text.erase(pos, zwj.length());
        }
    }

    // R-colored vowels (standard IPA -> Gruut US decomposed)
    std::vector<std::pair<std::string, std::string>> replacements = {
        {"ɝ", "ɜɹ"},   // nurse
        {"ɚ", "əɹ"},   // letter

        // Diphthongs (diphthong -> single uppercase letter)
        // Must process longer patterns first
        {"eɪ", "A"},   // face
        {"aɪ", "I"},   // price
        {"ɔɪ", "Y"},   // choice
        {"oʊ", "O"},   // goat (American)
        {"əʊ", "O"},   // goat (British compatibility)
        {"ɛʊ", "O"},   // goat variant
        {"aʊ", "W"},   // mouth

        // Affricates
        {"tʃ", "ʧ"},   // cheese
        {"dʒ", "ʤ"},   // joy

        // Consonant normalization
        {"g", "ɡ"},    // Standard g -> Script g (U+0261)
        {"r", "ɹ"},    // Standard r -> Turned r (U+0279)
    };

    for (const auto& rep : replacements) {
        size_t pos = 0;
        while ((pos = text.find(rep.first, pos)) != std::string::npos) {
            text.replace(pos, rep.first.length(), rep.second);
            pos += rep.second.length();
        }
    }

    return text;
}

// Read lexicon from file
std::unordered_map<std::string, std::string> readLexicon(const std::string& path) {
    std::unordered_map<std::string, std::string> lexicon;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open lexicon file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string word = line.substr(0, space_pos);
                std::string phones = line.substr(space_pos + 1);
                lexicon[word] = phones;
            }
        }
    }
    
    return lexicon;
}

// Convert mel spectrogram to audio using vocoder
std::vector<float> vocoderInference(Ort::Session& session, const std::vector<float>& mel, int mel_dim, const TTSConfig& config) {
    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    
    
    // Prepare input tensor
    int64_t num_frames = mel.size() / mel_dim;
    std::vector<int64_t> input_shape = {1, mel_dim, num_frames};
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(mel.data()), mel.size(),
        input_shape.data(), input_shape.size()
    );
    
    // Use correct vocoder interface - get all outputs to see which one is audio
    const char* input_names[] = {"mels"};
    const char* output_names[] = {"mag", "x", "y"};
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                     input_names, &input_tensor, 1,
                                     output_names, 3);
    
    
    // Vocos outputs frequency domain data that needs ISTFT post-processing
    // Extract the three outputs: mag, x (real), y (imag)
    float* mag_data = output_tensors[0].GetTensorMutableData<float>();
    float* x_data = output_tensors[1].GetTensorMutableData<float>();
    float* y_data = output_tensors[2].GetTensorMutableData<float>();
    
    auto vocoder_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = vocoder_shape[0];
    int32_t n_fft_bins = vocoder_shape[1];  // 513 = (1024/2 + 1) for n_fft=1024
    int32_t vocoder_frames = vocoder_shape[2];    
    
    // Reconstruct complex STFT from magnitude and phase components
    // Follow sherpa-onnx layout: (num_frames, n_fft/2+1)
    // real = mag * x, imag = mag * y
    std::vector<float> stft_real(vocoder_frames * n_fft_bins);
    std::vector<float> stft_imag(vocoder_frames * n_fft_bins);
    
    // Vocoder output is (batch, freq, time), we need (time, freq)
    for (int32_t frame = 0; frame < vocoder_frames; ++frame) {
        for (int32_t bin = 0; bin < n_fft_bins; ++bin) {
            int32_t vocoder_idx = bin * vocoder_frames + frame;  // (freq, time) layout
            int32_t stft_idx = frame * n_fft_bins + bin;         // (time, freq) layout
            
            stft_real[stft_idx] = mag_data[vocoder_idx] * x_data[vocoder_idx];
            stft_imag[stft_idx] = mag_data[vocoder_idx] * y_data[vocoder_idx];
        }
    }
    
    // Use proper ISTFT implementation following sherpa-onnx approach
    int32_t n_fft = 1024;
    int32_t hop_length = 256;
    int32_t win_length = 1024;
    
    // Calculate proper audio length (without center padding handling for now)
    int32_t audio_length = n_fft + (vocoder_frames - 1) * hop_length;
    std::vector<float> audio(audio_length, 0.0f);
    std::vector<float> denominator(audio_length, 0.0f);
    
    // Create Hann window
    std::vector<float> window(win_length);
    for (int32_t i = 0; i < win_length; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (win_length - 1)));
    }
    
    
    // Process each frame using correct ISTFT approach
    for (int32_t frame = 0; frame < vocoder_frames; ++frame) {
        // Prepare complex data for IFFT - follow kaldi-native-fbank format
        std::vector<float> ifft_data(n_fft);
        
        // Extract real and imag for this frame
        const float *p_real = stft_real.data() + frame * n_fft_bins;
        const float *p_imag = stft_imag.data() + frame * n_fft_bins;
        
        // Pack for real FFT (hermitian symmetry format)
        for (int32_t i = 0; i < n_fft / 2; ++i) {
            if (i == 0) {
                ifft_data[0] = p_real[0];  // DC component
                ifft_data[1] = p_real[n_fft / 2];  // Nyquist component  
            } else {
                ifft_data[2 * i] = p_real[i];      // Real part
                ifft_data[2 * i + 1] = p_imag[i];  // Imaginary part
            }
        }
        
        // Perform inverse FFT using FFTW with proper alignment
        // Use aligned allocation for RISC-V compatibility
        fftwf_complex* in = nullptr;
        float* out = nullptr;
        
        // Ensure 16-byte alignment for RISC-V
        const size_t alignment = 16;
        size_t in_size = sizeof(fftwf_complex) * (n_fft / 2 + 1);
        size_t out_size = sizeof(float) * n_fft;
        
        // Use aligned allocation
        if (posix_memalign((void**)&in, alignment, in_size) != 0) {
            throw std::runtime_error("Failed to allocate aligned memory for FFT input");
        }
        if (posix_memalign((void**)&out, alignment, out_size) != 0) {
            free(in);
            throw std::runtime_error("Failed to allocate aligned memory for FFT output");
        }
        
        // Use FFTW_UNALIGNED flag for RISC-V to handle potential alignment issues
        fftwf_plan plan = fftwf_plan_dft_c2r_1d(n_fft, in, out, FFTW_ESTIMATE | FFTW_UNALIGNED);
        
        // Copy to FFTW input format with explicit bounds checking
        for (int32_t i = 0; i < n_fft_bins && i < (n_fft / 2 + 1); ++i) {
            in[i][0] = p_real[i];
            in[i][1] = p_imag[i];
        }
        
        // Execute IFFT
        fftwf_execute(plan);
        
        // Apply IFFT normalization
        float scale = 1.0f / n_fft;
        for (int32_t i = 0; i < n_fft; ++i) {
            out[i] *= scale;
        }
        
        // Apply window
        for (int32_t i = 0; i < win_length && i < n_fft; ++i) {
            out[i] *= window[i];
        }
        
        // Overlap-add with bounds checking
        int32_t start_pos = frame * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            if (start_pos + i < audio_length) {
                audio[start_pos + i] += out[i];
                denominator[start_pos + i] += window[i] * window[i];
            }
        }
        
        // Cleanup
        fftwf_destroy_plan(plan);
        free(in);  // Use free() for posix_memalign allocated memory
        free(out);
    }
    
    // Normalize by window overlap
    for (int32_t i = 0; i < audio_length; ++i) {
        if (denominator[i] > 1e-8f) {
            audio[i] /= denominator[i];
        }
    }
    
    // Apply audio normalization and dynamic range compression
    audio = normalizeAudio(audio, config);
    
    // Post-process to remove clicks and pops
    if (config.remove_clicks) {
        audio = removeClicksAndPops(audio);
    }
    
    return audio;
}

// Calculate RMS (Root Mean Square) of audio signal
float calculateRMS(const std::vector<float>& audio) {
    if (audio.empty()) return 0.0f;
    
    float sum_squares = 0.0f;
    for (float sample : audio) {
        sum_squares += sample * sample;
    }
    return std::sqrt(sum_squares / audio.size());
}

// Apply dynamic range compression
std::vector<float> applyCompression(const std::vector<float>& audio, float threshold, float ratio) {
    std::vector<float> compressed = audio;
    
    for (float& sample : compressed) {
        float abs_sample = std::abs(sample);
        if (abs_sample > threshold) {
            // Apply compression above threshold
            float over_threshold = abs_sample - threshold;
            float compressed_over = over_threshold / ratio;
            float new_abs = threshold + compressed_over;
            sample = (sample < 0) ? -new_abs : new_abs;
        }
    }
    
    return compressed;
}

// Normalize audio with RMS normalization and optional dynamic range compression
std::vector<float> normalizeAudio(const std::vector<float>& audio, const TTSConfig& config) {
    if (audio.empty()) return audio;
    
    // Use config parameters
    const float target_rms = config.target_rms;
    const float compression_ratio = config.compression_ratio;
    const float compression_threshold = config.compression_threshold;
    const bool use_rms_norm = config.use_rms_norm;
    
    // Create aligned copy for RISC-V safety
    std::vector<float> processed;
    processed.reserve(audio.size());
    processed.assign(audio.begin(), audio.end());
    
    // Step 1: Apply dynamic range compression to reduce volume variations
    processed = applyCompression(processed, compression_threshold, compression_ratio);
    
    // Step 2: Apply normalization
    if (use_rms_norm) {
        // RMS normalization for consistent perceived loudness
        float current_rms = calculateRMS(processed);
        if (current_rms > 0.0f) {
            float scale = target_rms / current_rms;
            
            // Apply soft limiting to prevent clipping
            const float max_scale = 3.0f;  // Limit amplification to prevent noise
            scale = std::min(scale, max_scale);
            
            for (float& sample : processed) {
                sample *= scale;
            }
            
            // Soft clipping to prevent harsh distortion
            for (float& sample : processed) {
                if (std::abs(sample) > 0.95f) {
                    float sign = (sample < 0) ? -1.0f : 1.0f;
                    float abs_val = std::abs(sample);
                    // Soft knee compression near clipping
                    sample = sign * (0.95f + 0.05f * std::tanh((abs_val - 0.95f) * 20.0f));
                }
            }
        }
    } else {
        // Fallback to peak normalization
        float max_amplitude = 0.0f;
        for (float sample : processed) {
            max_amplitude = std::max(max_amplitude, std::abs(sample));
        }
        
        if (max_amplitude > 0.0f) {
            float scale = 0.8f / max_amplitude;
            for (float& sample : processed) {
                sample *= scale;
            }
        }
    }
    
    return processed;
}

// Remove clicks and pops from audio by applying fade-in/out and DC offset removal
std::vector<float> removeClicksAndPops(const std::vector<float>& audio) {
    if (audio.empty()) return audio;
    
    // Create aligned copy for RISC-V safety
    std::vector<float> processed;
    processed.reserve(audio.size());
    processed.assign(audio.begin(), audio.end());
    
    // Step 1: Remove DC offset (average value should be zero)
    float dc_offset = 0.0f;
    for (float sample : processed) {
        dc_offset += sample;
    }
    dc_offset /= processed.size();
    
    // Only remove DC offset if it's significant (> 0.01)
    if (std::abs(dc_offset) > 0.01f) {
        for (float& sample : processed) {
            sample -= dc_offset;
        }
    }
    
    // Step 2: Apply very short fade-in at the beginning (2ms at 22050Hz = ~44 samples)
    // Only fade the very beginning to avoid clicks, not affect overall volume
    const int fade_in_samples = std::min(44, static_cast<int>(processed.size() / 100));
    for (int i = 0; i < fade_in_samples && i < processed.size(); ++i) {
        float fade_factor = static_cast<float>(i) / fade_in_samples;
        // Use cosine fade for smoother transition
        fade_factor = 0.5f * (1.0f - std::cos(M_PI * fade_factor));
        processed[i] *= fade_factor;
    }
    
    // Step 3: Apply short fade-out at the end (5ms = ~110 samples)
    const int fade_out_samples = std::min(110, static_cast<int>(processed.size() / 50));
    for (int i = 0; i < fade_out_samples && i < processed.size(); ++i) {
        int idx = processed.size() - 1 - i;
        float fade_factor = static_cast<float>(i) / fade_out_samples;
        // Use cosine fade for smoother transition
        fade_factor = 0.5f * (1.0f - std::cos(M_PI * fade_factor));
        processed[idx] *= fade_factor;
    }
    
    // Step 4: Simple DC blocking filter (high-pass at 20Hz)
    // This is more gentle than the previous implementation
    if (processed.size() > 1) {
        const float cutoff = 0.999f;  // Very gentle high-pass
        float prev_input = 0.0f;
        float prev_output = 0.0f;
        
        for (size_t i = 0; i < processed.size(); ++i) {
            float current_input = processed[i];
            float current_output = cutoff * (prev_output + current_input - prev_input);
            processed[i] = current_output;
            prev_input = current_input;
            prev_output = current_output;
        }
    }
    
    // Step 5: Ensure the very last sample is zero (single sample only)
    if (!processed.empty()) {
        processed.back() = 0.0f;
    }
    
    return processed;
}

} // anonymous namespace

// TTSModel implementation
class TTSModel::Impl {
public:
    explicit Impl(const TTSConfig& config) : config_(config) {}
    
    bool initialize() {
        try {
            // Initialize ONNX Runtime
            // Temporarily suppress stderr to avoid ONNX schema warnings
            int stderr_fd = dup(STDERR_FILENO);
            int devnull_fd = open("/dev/null", O_WRONLY);
            dup2(devnull_fd, STDERR_FILENO);
            
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TTSModel");
            
            // Restore stderr
            dup2(stderr_fd, STDERR_FILENO);
            close(stderr_fd);
            close(devnull_fd);
            
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(3);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // RISC-V specific: Disable memory arena to avoid alignment issues
            #if defined(__riscv) || defined(__riscv__)
            session_options.DisableMemPattern();
            session_options.DisableCpuMemArena();
            #endif
            
            // Load acoustic model
            acoustic_model_ = std::make_unique<Ort::Session>(*env_, config_.acoustic_model_path.c_str(), session_options);

            // Load vocoder model
            vocoder_model_ = std::make_unique<Ort::Session>(*env_, config_.vocoder_path.c_str(), session_options);

            // Load token to ID mapping (different format for zh-en model)
            if (config_.language == "zh-en") {
                token_to_id_ = readZhEnTokenToIdMap(config_.tokens_path);
            } else {
                token_to_id_ = readTokenToIdMap(config_.tokens_path);
            }
            
            // Load lexicon (only for Chinese or if lexicon file exists)
            if (config_.language == "zh" && !config_.lexicon_path.empty()) {
                if (std::filesystem::exists(config_.lexicon_path)) {
                    lexicon_ = readLexicon(config_.lexicon_path);
                } else {
                    std::cout << "Warning: Lexicon file not found: " << config_.lexicon_path << ". Continuing without lexicon." << std::endl;
                }
            } else if (config_.language == "en") {
                std::cout << "Info: English mode - lexicon not required." << std::endl;

                // Check if espeak-ng is available for English TTS
                if (!checkEspeakNgAvailable()) {
                    std::cerr << "Error: espeak-ng is required for English TTS but not available." << std::endl;
                    std::cerr << "Please install espeak-ng using: brew install espeak-ng (macOS) or apt-get install espeak-ng (Linux)" << std::endl;
                    throw std::runtime_error("espeak-ng not available for English TTS");
                }
                std::cout << "Info: espeak-ng found and available for English TTS." << std::endl;
            } else if (config_.language == "zh-en") {
                std::cout << "Info: zh-en bilingual mode - using cpp-pinyin for Chinese and espeak-ng for English." << std::endl;

                // Check if espeak-ng is available for English parts
                if (!checkEspeakNgAvailable()) {
                    std::cerr << "Error: espeak-ng is required for zh-en TTS but not available." << std::endl;
                    std::cerr << "Please install espeak-ng using: brew install espeak-ng (macOS) or apt-get install espeak-ng (Linux)" << std::endl;
                    throw std::runtime_error("espeak-ng not available for zh-en TTS");
                }

                // Initialize cpp-pinyin
                try {
                    initializePinyin();
                } catch (const std::exception& e) {
                    std::cerr << "Error: Failed to initialize cpp-pinyin: " << e.what() << std::endl;
                    throw std::runtime_error("cpp-pinyin initialization failed for zh-en TTS");
                }
            }
            
            // Initialize Jieba for Chinese
            if (config_.language == "zh") {
                try {
                    initializeJieba();
                } catch (const std::exception& e) {
                    std::cout << "Warning: Failed to initialize Jieba: " << e.what() << ". Continuing without Jieba." << std::endl;
                    jieba_ = nullptr;
                }
            }
            
            // Get model metadata
            extractModelMetadata();
            
            // Warm up the models to avoid slow first inference
            if (config_.enable_warmup) {
                warmUpModels();
            }
            
            initialized_ = true;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize TTS model: " << e.what() << std::endl;
            return false;
        }
    }
    
    GeneratedAudio generate(const std::string& text, int speaker_id, float speed) {
        if (!initialized_) {
            throw std::runtime_error("TTS model not initialized");
        }
        
        // Preprocess text
        std::string processed_text = preprocessText(text);
        
        // Convert text to tokens
        std::vector<int64_t> token_ids = textToTokenIds(processed_text);
        
        if (token_ids.empty()) {
            GeneratedAudio audio;
            audio.sample_rate = config_.sample_rate;
            return audio;
        }

        // Add blank tokens between phonemes (Matcha requirement)
        // Note: zh-en model does NOT use blank tokens
        std::vector<int64_t> final_tokens;
        if (config_.language == "zh-en") {
            final_tokens = token_ids;  // No blank tokens for zh-en model
        } else {
            final_tokens = addBlankTokens(token_ids);
        }

        // Run acoustic model
        std::vector<float> mel = runAcousticModel(final_tokens, speaker_id, speed);
        
        if (mel.empty()) {
            GeneratedAudio audio;
            audio.sample_rate = config_.sample_rate;
            return audio;
        }
        
        // Run vocoder
        std::vector<float> audio_samples;
        {
            // Lock for thread-safe ONNX Runtime inference on RISC-V
            std::lock_guard<std::mutex> lock(inference_mutex_);
            audio_samples = vocoderInference(*vocoder_model_, mel, mel_dim_, config_);
        }
        
        // Create result with proper initialization
        GeneratedAudio audio;
        audio.samples.reserve(audio_samples.size()); // Pre-allocate for RISC-V
        audio.samples = std::move(audio_samples);
        audio.sample_rate = config_.sample_rate;
        
        return audio;
    }
    
    bool isInitialized() const {
        return initialized_;
    }
    
    int getNumSpeakers() const {
        return num_speakers_;
    }
    
    int getSampleRate() const {
        return config_.sample_rate;
    }
    
private:
    void warmUpModels() {
        std::cout << "Warming up TTS models..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Method 1: Run a complete inference with a short text
            // This ensures all code paths are executed and optimized
            std::string warmup_text = "测试";
            
            // Process the text through the full pipeline
            std::string processed_text = preprocessText(warmup_text);
            std::vector<int64_t> token_ids = textToTokenIds(processed_text);
            
            if (!token_ids.empty()) {
                // Add blank tokens
                std::vector<int64_t> tokens_with_blanks = addBlankTokens(token_ids);
                
                // Run acoustic model
                std::vector<float> mel = runAcousticModel(tokens_with_blanks, 0, 1.0f);
                
                if (!mel.empty()) {
                    // Run vocoder
                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    std::vector<float> audio = vocoderInference(*vocoder_model_, mel, mel_dim_, config_);
                    
                    // The generated audio is discarded, we just want to warm up the models
                }
            }
            
            // Method 2: Additional warm-up with different sizes to cover more cases
            // Small input
            std::vector<int64_t> small_tokens = {1, 2, 3};
            std::vector<int64_t> small_with_blanks = addBlankTokens(small_tokens);
            runAcousticModel(small_with_blanks, 0, 1.0f);
            
            // Medium input
            std::vector<int64_t> medium_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            std::vector<int64_t> medium_with_blanks = addBlankTokens(medium_tokens);
            runAcousticModel(medium_with_blanks, 0, 1.0f);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "TTS models warmed up successfully in " << duration.count() << "ms" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: TTS warm-up failed: " << e.what() << std::endl;
            // Don't fail initialization if warm-up fails
        }
    }
    
    void initializeJieba() {
        // Initialize Jieba with dictionary
        // Use jieba_dict_dir if specified, otherwise try to find it relative to executable
        std::string jieba_dir = config_.jieba_dict_dir;
        if (jieba_dir.empty()) {
            // Try multiple possible locations
            std::vector<std::string> possible_paths = {
                "../third_party/cppjieba/dict",
                "../../third_party/cppjieba/dict",
                "../../../third_party/cppjieba/dict",
                "third_party/cppjieba/dict",
                "/home/rongmingjun/asr_llm_tts_cpp/ai/third_party/cppjieba/dict"
            };
            
            for (const auto& path : possible_paths) {
                if (fs::exists(path + "/jieba.dict.utf8")) {
                    jieba_dir = path;
                    std::cout << "Found cppjieba dictionary at: " << jieba_dir << std::endl;
                    break;
                }
            }
            
            if (jieba_dir.empty()) {
                throw std::runtime_error("Cannot find cppjieba dictionary. Please set jieba_dict_dir in config.");
            }
        }
        
        std::string dict_path = jieba_dir + "/jieba.dict.utf8";
        std::string hmm_path = jieba_dir + "/hmm_model.utf8";
        std::string user_dict = jieba_dir + "/user.dict.utf8";
        std::string idf_path = jieba_dir + "/idf.utf8";
        std::string stop_words = jieba_dir + "/stop_words.utf8";
        
        jieba_ = std::make_unique<cppjieba::Jieba>(
            dict_path, hmm_path, user_dict, idf_path, stop_words
        );

    }

    void initializePinyin() {
        // Initialize cpp-pinyin dictionary path
        // Try multiple possible locations
        std::vector<std::string> possible_paths = {
            "../third_party/cpp-pinyin/res/dict",
            "../../third_party/cpp-pinyin/res/dict",
            "../../../third_party/cpp-pinyin/res/dict",
            "third_party/cpp-pinyin/res/dict",
            "/Users/mugglepro/workspace/e2e_Voice/third_party/cpp-pinyin/res/dict"
        };

        std::string pinyin_dict_dir;
        for (const auto& path : possible_paths) {
            if (fs::exists(path + "/mandarin")) {
                pinyin_dict_dir = path;
                std::cout << "Found cpp-pinyin dictionary at: " << pinyin_dict_dir << std::endl;
                break;
            }
        }

        if (pinyin_dict_dir.empty()) {
            throw std::runtime_error("Cannot find cpp-pinyin dictionary. Please check third_party/cpp-pinyin/res/dict.");
        }

        // Set dictionary path for cpp-pinyin
        Pinyin::setDictionaryPath(fs::path(pinyin_dict_dir));

        // Create Pinyin converter
        pinyin_converter_ = std::make_unique<Pinyin::Pinyin>();

        std::cout << "cpp-pinyin initialized successfully." << std::endl;
    }
    
    void extractModelMetadata() {
        // Get model inputs/outputs info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Check number of inputs
        size_t num_inputs = acoustic_model_->GetInputCount();
        
        // Read acoustic model metadata
        try {
            Ort::ModelMetadata acoustic_meta = acoustic_model_->GetModelMetadata();
            
            // Read pad_id from metadata
            auto pad_id_value = LookupCustomModelMetaData(acoustic_meta, "pad_id", allocator);
            if (!pad_id_value.empty()) {
                pad_id_ = std::stoi(pad_id_value);
            } else {
                pad_id_ = 0;  // Default value
            }
            
        } catch (const std::exception& e) {
            pad_id_ = 0;
        }
        
        // Extract ISTFT parameters from vocoder model metadata
        try {
            Ort::ModelMetadata vocoder_meta = vocoder_model_->GetModelMetadata();
            
            // Try to read ISTFT parameters from model metadata
            auto read_meta_int = [&](const char* key, int32_t& value, int32_t default_val) {
                try {
                    auto key_alloc = vocoder_meta.LookupCustomMetadataMapAllocated(key, allocator);
                    if (key_alloc) {
                        value = std::stoi(key_alloc.get());
                        // Loaded value from metadata
                    } else {
                        value = default_val;
                    }
                } catch (...) {
                    value = default_val;
                }
            };
            
            auto read_meta_bool = [&](const char* key, bool& value, bool default_val) {
                try {
                    auto key_alloc = vocoder_meta.LookupCustomMetadataMapAllocated(key, allocator);
                    if (key_alloc) {
                        std::string str_val = key_alloc.get();
                        value = (str_val == "true" || str_val == "True" || str_val == "1");
                        // Loaded value from metadata
                    } else {
                        value = default_val;
                    }
                } catch (...) {
                    value = default_val;
                }
            };
            
            auto read_meta_string = [&](const char* key, std::string& value, const std::string& default_val) {
                try {
                    auto key_alloc = vocoder_meta.LookupCustomMetadataMapAllocated(key, allocator);
                    if (key_alloc) {
                        value = key_alloc.get();
                        // Loaded value from metadata
                    } else {
                        value = default_val;
                    }
                } catch (...) {
                    value = default_val;
                }
            };
            
            // Read ISTFT parameters
            read_meta_int("n_fft", istft_n_fft_, 1024);
            read_meta_int("hop_length", istft_hop_length_, 256);
            read_meta_int("win_length", istft_win_length_, 1024);
            read_meta_bool("center", istft_center_, true);
            read_meta_bool("normalized", istft_normalized_, false);
            read_meta_string("window_type", istft_window_type_, "hann");
            read_meta_string("pad_mode", istft_pad_mode_, "reflect");
            
        } catch (const std::exception& e) {
            // Use sensible defaults for Vocos
            istft_n_fft_ = 1024;
            istft_hop_length_ = 256;
            istft_win_length_ = 1024;
            istft_center_ = true;
            istft_normalized_ = false;
            istft_window_type_ = "hann";
            istft_pad_mode_ = "reflect";
        }
        
        // For Matcha models, mel_dim is typically 80
        mel_dim_ = 80;
        
        // TODO: Extract number of speakers from model metadata
        num_speakers_ = 1;
    }
    
    std::string preprocessText(const std::string& text) {
        // Basic text normalization
        std::string processed = text;
        
        // Convert to lowercase for English
        if (config_.language == "en") {
            std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
        }
        
        // TODO: Add more sophisticated text normalization
        // - Number to words conversion
        // - Abbreviation expansion
        // - Punctuation handling
        
        return processed;
    }
    
    
    std::string mapPhoneme(const std::string& phone) {
        // Handle common phoneme mismatches between lexicon and token vocabulary
        static std::unordered_map<std::string, std::string> phoneme_mapping = {
            // Common mappings for missing phonemes
            {"shei2", "she2"},  // who (谁) -> she sound
            {"cei2", "ce2"},    // missing variants
            {"den1", "de1"},    // missing variants
            {"den2", "de2"},
            {"den3", "de3"},
            {"den4", "de4"},
            {"kei2", "ke2"},    // missing variants
            {"kei3", "ke3"},
            {"nei1", "ne1"},    // missing variants
            {"pou1", "po1"},    // missing variants
            {"pou2", "po2"},
            {"pou3", "po3"},
            {"yo1", "yo"},      // missing tone variant
            {"m2", "m"},        // missing tone on nasal
            {"n2", "n"},        // missing tone on nasal
            {"ng2", "ng"},      // missing ng phoneme
            {"hm", "hm1"},      // add tone to hmm sound
        };
        
        auto it = phoneme_mapping.find(phone);
        if (it != phoneme_mapping.end()) {
            return it->second;
        }
        
        // If no direct mapping found, try removing or changing tone
        if (phone.length() > 1) {
            char last_char = phone.back();
            if (last_char >= '1' && last_char <= '4') {
                // Try without tone
                std::string base = phone.substr(0, phone.length() - 1);
                return base;
            } else {
                // Try adding tone 1 if no tone present
                return phone + "1";
            }
        }
        
        return phone;  // Return original if no mapping found
    }

    std::vector<int64_t> textToTokenIds(const std::string& text) {
        std::vector<int64_t> token_ids;
        
        if (config_.language == "zh" && jieba_) {
            // Follow sherpa-onnx's approach for Chinese text processing
            
            // Step 1: Replace punctuations (like sherpa-onnx)
            std::string processed_text = text;
            // Use regex to replace punctuations following sherpa-onnx pattern
            std::regex punct_re1("：|、|；");
            processed_text = std::regex_replace(processed_text, punct_re1, "，");
            std::regex punct_re2("[.]");
            processed_text = std::regex_replace(processed_text, punct_re2, "。");
            std::regex punct_re3("[?]");
            processed_text = std::regex_replace(processed_text, punct_re3, "？");
            std::regex punct_re4("[!]");
            processed_text = std::regex_replace(processed_text, punct_re4, "！");
            
            
            // Step 2: Jieba segmentation
            std::vector<std::string> words;
            jieba_->Cut(processed_text, words, true);  // Use HMM
            
            
            // Step 3: Remove redundant spaces and punctuations like sherpa-onnx
            std::vector<std::string> cleaned_words;
            for (size_t i = 0; i < words.size(); ++i) {
                if (i == 0) {
                    cleaned_words.push_back(words[i]);
                } else if (words[i] == " ") {
                    if (cleaned_words.back() == " " || isPunctuation(cleaned_words.back())) {
                        continue;  // Skip redundant spaces
                    } else {
                        cleaned_words.push_back(words[i]);
                    }
                } else if (isPunctuation(words[i])) {
                    if (cleaned_words.back() == " " || isPunctuation(cleaned_words.back())) {
                        continue;  // Skip redundant punctuations
                    } else {
                        cleaned_words.push_back(words[i]);
                    }
                } else {
                    cleaned_words.push_back(words[i]);
                }
            }
            
            
            // Step 4: Convert words to token IDs
            for (const auto& word : cleaned_words) {
                auto word_ids = convertWordToIds(word);
                if (!word_ids.empty()) {
                    token_ids.insert(token_ids.end(), word_ids.begin(), word_ids.end());
                }
            }
        } else if (config_.language == "en") {
            // English text processing
            
            // First, check if text contains Chinese characters - if so, skip it silently
            if (containsChinese(text)) {
                // Silently skip Chinese text when in English mode
                return token_ids; // Return empty token_ids
            }
            
            // English text processing using espeak-ng for phoneme generation
            std::string phonemes = processEnglishTextToPhonemes(text);
            if (phonemes.empty() && !text.empty()) {
                // espeak-ng is required for proper English TTS
                std::cerr << "Error: espeak-ng is required for English TTS but not available. Please install espeak-ng." << std::endl;
                return token_ids; // Return empty token_ids to skip this text
            }
            
            // Add start token (^) - sherpa-onnx style
            auto start_it = token_to_id_.find("^");
            if (start_it != token_to_id_.end()) {
                token_ids.push_back(start_it->second);
            }
            
            // Process phonemes character by character (IPA symbols)
            std::vector<std::string> phoneme_chars = splitUtf8(phonemes);
            bool last_was_space = false; // Track consecutive spaces to reduce excessive blanks
            
            for (const auto& phoneme_char : phoneme_chars) {
                if (phoneme_char.empty()) continue;
                
                // Filter out problematic characters
                if (phoneme_char == "\u200D" || // Zero-width joiner
                    phoneme_char == "\u200C" || // Zero-width non-joiner
                    phoneme_char == "\uFEFF" || // Byte order mark
                    phoneme_char == "\u00A0" || // Non-breaking space
                    phoneme_char.size() == 1 && std::iscntrl(static_cast<unsigned char>(phoneme_char[0]))) {
                    continue; // Skip these characters
                }
                
                // Handle spaces - limit consecutive spaces to reduce excessive blanks
                if (phoneme_char == " ") {
                    if (last_was_space) {
                        continue; // Skip consecutive spaces
                    }
                    last_was_space = true;
                } else {
                    last_was_space = false;
                }
                
                auto token_it = token_to_id_.find(phoneme_char);
                if (token_it != token_to_id_.end()) {
                    token_ids.push_back(token_it->second);
                } else if (phoneme_char != " ") { // Don't warn about spaces
                    // Only log unknown tokens that aren't common formatting characters
                    std::cerr << "Warning: Unknown phoneme token: '" << phoneme_char << "' (hex: ";
                    for (unsigned char c : phoneme_char) {
                        std::cerr << std::hex << (int)c << " ";
                    }
                    std::cerr << ")" << std::endl;
                }
            }
            
            // Add end token ($) - sherpa-onnx style
            auto end_it = token_to_id_.find("$");
            if (end_it != token_to_id_.end()) {
                token_ids.push_back(end_it->second);
            }

            return token_ids;
        } else if (config_.language == "zh-en" && pinyin_converter_) {
            // zh-en bilingual text processing
            // Process character by character, detecting Chinese vs English
            token_ids = processZhEnText(text);
            return token_ids;
        } else {
            // For other languages, skip with warning
            if (!text.empty()) {
                std::cerr << "Warning: Unsupported language '" << config_.language << "'. Text will be skipped: " << text << std::endl;
            }
            return token_ids;
        }
        
        
        return token_ids;
    }
    
    bool isPunctuation(const std::string& s) {
        static const std::unordered_set<std::string> puncts = {
            ",", ".", "!", "?", ":", "\"", "'", "，",
            "。", "！", "？", """, """, "'", "'", "；", "、",
            "—", "–", "…", "-", "(", ")", "（", "）",
            "[", "]", "【", "】", "{", "}", "《", "》"
        };
        return puncts.count(s);
    }
    
    // Helper function to check if a character is Chinese (CJK)
    bool isChinese(unsigned char ch) {
        // Simple check for Chinese characters using UTF-8 byte patterns
        // Chinese characters typically start with bytes in range 0xE4-0xE9
        return ch >= 0xE4 && ch <= 0xE9;
    }
    
    // Helper function to check if text contains Chinese characters
    bool containsChinese(const std::string& text) {
        for (size_t i = 0; i < text.length(); i++) {
            unsigned char ch = static_cast<unsigned char>(text[i]);
            if (isChinese(ch)) {
                return true;
            }
        }
        return false;
    }
    
    // Check if espeak-ng is available
    bool checkEspeakNgAvailable() {
        // Try to run espeak-ng with a simple test
        std::string command = "echo 'test' | espeak-ng -q --ipa=3 2>/dev/null";
        
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
        if (!pipe) {
            return false;
        }
        
        char buffer[128];
        std::string result;
        if (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            result += buffer;
        }
        
        int exit_status = pclose(pipe.release());
        return exit_status == 0 && !result.empty();
    }

    // Check if a UTF-8 character is a Chinese character (CJK Unified Ideographs)
    bool isChineseChar(const std::string& ch) {
        if (ch.length() != 3) return false;
        unsigned char c0 = ch[0];
        unsigned char c1 = ch[1];
        // CJK Unified Ideographs: U+4E00 to U+9FFF (3-byte UTF-8: E4 B8 80 to E9 BF BF)
        if (c0 >= 0xE4 && c0 <= 0xE9) {
            if (c0 == 0xE4 && c1 < 0xB8) return false;
            if (c0 == 0xE9 && c1 > 0xBF) return false;
            return true;
        }
        return false;
    }

    // Check if a character is an ASCII letter
    bool isEnglishLetter(const std::string& ch) {
        if (ch.length() != 1) return false;
        char c = ch[0];
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }

    // Check if a character is a Roman numeral character
    bool isRomanNumeralChar(char c) {
        c = std::toupper(c);
        return c == 'I' || c == 'V' || c == 'X' || c == 'L' || c == 'C' || c == 'D' || c == 'M';
    }

    // Check if a string is a valid Roman numeral
    bool isRomanNumeral(const std::string& str) {
        if (str.empty()) return false;
        for (char c : str) {
            if (!isRomanNumeralChar(c)) return false;
        }
        // Additional validation: check if it's a valid Roman numeral pattern
        // Simple validation: must contain at least one Roman numeral character
        return true;
    }

    // Convert Roman numeral to integer
    int romanToInt(const std::string& roman) {
        std::unordered_map<char, int> values = {
            {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50},
            {'C', 100}, {'D', 500}, {'M', 1000}
        };

        int result = 0;
        for (size_t i = 0; i < roman.length(); i++) {
            char c = std::toupper(roman[i]);
            int value = values[c];

            // Check if this is a subtractive case
            if (i + 1 < roman.length()) {
                char next = std::toupper(roman[i + 1]);
                int next_value = values[next];
                if (value < next_value) {
                    result -= value;
                } else {
                    result += value;
                }
            } else {
                result += value;
            }
        }
        return result;
    }

    // Convert integer to Chinese reading (for numbers)
    std::string intToChineseReading(int num) {
        if (num == 0) return "零";
        if (num < 0) return "负" + intToChineseReading(-num);

        static const char* digits[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
        static const char* units[] = {"", "十", "百", "千"};
        static const char* bigUnits[] = {"", "万", "亿"};

        std::string result;

        // Handle numbers up to 99999999 (亿)
        if (num >= 100000000) {
            result += intToChineseReading(num / 100000000) + "亿";
            num %= 100000000;
            if (num > 0 && num < 10000000) result += "零";
        }
        if (num >= 10000) {
            result += intToChineseReading(num / 10000) + "万";
            num %= 10000;
            if (num > 0 && num < 1000) result += "零";
        }
        if (num >= 1000) {
            result += std::string(digits[num / 1000]) + "千";
            num %= 1000;
            if (num > 0 && num < 100) result += "零";
        }
        if (num >= 100) {
            result += std::string(digits[num / 100]) + "百";
            num %= 100;
            if (num > 0 && num < 10) result += "零";
        }
        if (num >= 10) {
            if (num / 10 != 1 || result.length() > 0) {
                result += digits[num / 10];
            }
            result += "十";
            num %= 10;
        }
        if (num > 0) {
            result += digits[num];
        }

        return result;
    }

    // Process Roman numeral and convert to Chinese reading IDs
    std::vector<int64_t> processRomanNumeralToIds(const std::string& roman) {
        int value = romanToInt(roman);
        std::string chinese = intToChineseReading(value);
        return processChineseToPinyinIds(chinese);
    }

    // Process zh-en bilingual text
    std::vector<int64_t> processZhEnText(const std::string& text) {
        std::vector<int64_t> token_ids;
        std::vector<std::string> chars = splitUtf8(text);

        size_t i = 0;
        while (i < chars.size()) {
            if (isChineseChar(chars[i])) {
                // Collect consecutive Chinese characters
                std::string chinese_part;
                while (i < chars.size() && isChineseChar(chars[i])) {
                    chinese_part += chars[i];
                    i++;
                }
                // Convert Chinese to pinyin and get IDs
                auto ids = processChineseToPinyinIds(chinese_part);
                token_ids.insert(token_ids.end(), ids.begin(), ids.end());
            } else if (isEnglishLetter(chars[i])) {
                // Collect consecutive English letters
                std::string english_part;
                while (i < chars.size() && isEnglishLetter(chars[i])) {
                    english_part += chars[i];
                    i++;
                }

                // Check if it's a Roman numeral (all uppercase Roman numeral chars)
                if (isRomanNumeral(english_part)) {
                    // Convert Roman numeral to Chinese number reading
                    auto ids = processRomanNumeralToIds(english_part);
                    token_ids.insert(token_ids.end(), ids.begin(), ids.end());
                } else {
                    // Convert English to IPA and get IDs
                    auto ids = processEnglishToIds(english_part);
                    token_ids.insert(token_ids.end(), ids.begin(), ids.end());
                }
            } else {
                // Handle punctuation and other characters
                std::string ch = chars[i];
                // Map Chinese punctuation to ASCII
                if (ch == "，") ch = ",";
                else if (ch == "。") ch = ".";
                else if (ch == "！") ch = "!";
                else if (ch == "？") ch = "?";

                auto it = token_to_id_.find(ch);
                if (it != token_to_id_.end()) {
                    token_ids.push_back(it->second);
                } else {
                    // Default to 1 for unknown tokens
                    token_ids.push_back(1);
                }
                i++;
            }
        }

        return token_ids;
    }

    // Convert Chinese text to pinyin and then to token IDs
    std::vector<int64_t> processChineseToPinyinIds(const std::string& chinese_text) {
        std::vector<int64_t> ids;

        if (!pinyin_converter_) {
            return ids;
        }

        // Use cpp-pinyin to convert Chinese to pinyin with tone numbers
        Pinyin::PinyinResVector result = pinyin_converter_->hanziToPinyin(
            chinese_text,
            Pinyin::ManTone::Style::TONE3,  // Tone numbers at end (zhong1)
            Pinyin::Error::Default,
            false,  // candidates
            false,  // v_to_u
            true    // neutral_tone_with_five (轻声用5)
        );

        // Convert each pinyin to token ID
        for (const auto& res : result) {
            std::string pinyin = res.pinyin;
            auto it = token_to_id_.find(pinyin);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                // Try lowercase
                std::string lower_pinyin = pinyin;
                std::transform(lower_pinyin.begin(), lower_pinyin.end(), lower_pinyin.begin(), ::tolower);
                auto lower_it = token_to_id_.find(lower_pinyin);
                if (lower_it != token_to_id_.end()) {
                    ids.push_back(lower_it->second);
                } else {
                    // Default to 1 for unknown tokens
                    ids.push_back(1);
                }
            }
        }

        return ids;
    }

    // Convert English text to IPA and then to token IDs (for zh-en model)
    std::vector<int64_t> processEnglishToIds(const std::string& english_text) {
        std::vector<int64_t> ids;

        // Get IPA from espeak-ng
        std::string ipa = processEnglishTextToPhonemes(english_text);
        if (ipa.empty()) {
            return ids;
        }

        // Convert to gruut en-us format
        std::string gruut_ipa = convertToGruutEnUs(ipa);

        // Convert each character to token ID
        std::vector<std::string> ipa_chars = splitUtf8(gruut_ipa);
        for (const auto& ch : ipa_chars) {
            if (ch.empty()) continue;

            auto it = token_to_id_.find(ch);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                // Default to 1 for unknown tokens
                ids.push_back(1);
            }
        }

        return ids;
    }
    
    // Convert English text to IPA phonemes using espeak-ng
    std::string processEnglishTextToPhonemes(const std::string& text) {
        if (text.empty()) {
            return "";
        }
        
        // Escape single quotes in text to avoid shell command issues
        std::string escaped_text = text;
        std::string::size_type pos = 0;
        while ((pos = escaped_text.find("'", pos)) != std::string::npos) {
            escaped_text.replace(pos, 1, "'\"'\"'");
            pos += 5;
        }
        
        // Use espeak-ng to convert text to IPA phonemes
        std::string command = "echo '" + escaped_text + "' | espeak-ng -q --ipa=3";
        
        // Set espeak-ng data directory if available
        if (!config_.data_dir.empty() && std::filesystem::exists(config_.data_dir)) {
            command = "ESPEAK_DATA_PATH='" + config_.data_dir + "' " + command;
        }
        
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
        if (!pipe) {
            std::cerr << "Error: Failed to run espeak-ng command" << std::endl;
            return "";
        }
        
        char buffer[4096];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            result += buffer;
        }
        
        // Wait for the command to complete and check exit status
        int exit_status = pclose(pipe.release());
        if (exit_status != 0) {
            // Command failed, return empty to trigger fallback
            return "";
        }
        
        // Clean up the result - remove newlines and extra whitespace
        result.erase(std::remove_if(result.begin(), result.end(), 
            [](char c) { return c == '\n' || c == '\r'; }), result.end());
        
        // Replace multiple consecutive spaces with single space
        std::regex multi_space("\\s+");
        result = std::regex_replace(result, multi_space, " ");
        
        // Trim whitespace
        size_t start = result.find_first_not_of(" \t");
        if (start == std::string::npos) {
            return "";
        }
        size_t end = result.find_last_not_of(" \t");
        result = result.substr(start, end - start + 1);
        
        return result;
    }
    
    // Helper function to split UTF-8 string into individual characters
    std::vector<std::string> splitUtf8(const std::string& str) {
        std::vector<std::string> result;
        for (size_t i = 0; i < str.length();) {
            int char_len = 1;
            unsigned char c = str[i];
            
            // Determine UTF-8 character length
            if ((c & 0x80) == 0) {
                char_len = 1;  // ASCII
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;  // 2-byte UTF-8
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;  // 3-byte UTF-8
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;  // 4-byte UTF-8
            }
            
            if (i + char_len <= str.length()) {
                result.push_back(str.substr(i, char_len));
            }
            i += char_len;
        }
        return result;
    }
    
    std::vector<int64_t> convertWordToIds(const std::string& word) {
        // Convert word to lowercase for lookup (following sherpa-onnx)
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
        
        
        // Try direct word lookup in lexicon first (only if lexicon is loaded)
        if (!lexicon_.empty()) {
            auto lex_it = lexicon_.find(lower_word);
            if (lex_it != lexicon_.end()) {
                return convertPhonemesToIds(lex_it->second);
            }
        }
        
        // Try direct token lookup
        auto token_it = token_to_id_.find(word);
        if (token_it != token_to_id_.end()) {
            return {token_it->second};
        }
        
        // Handle punctuation mapping
        if (isPunctuation(word)) {
            // Map punctuations to their token equivalents
            std::string punct_token = mapPunctuation(word);
            if (!punct_token.empty()) {
                auto punct_it = token_to_id_.find(punct_token);
                if (punct_it != token_to_id_.end()) {
                    return {punct_it->second};
                }
            }
        }
        
        // Character-level fallback for OOV words
        std::vector<int64_t> result;
        result.reserve(word.length() * 2); // Pre-allocate for RISC-V alignment
        std::vector<std::string> chars = splitUtf8(word);
        
        for (const auto& char_str : chars) {
            if (!lexicon_.empty()) {
                auto char_lex_it = lexicon_.find(char_str);
                if (char_lex_it != lexicon_.end()) {
                    auto char_ids = convertPhonemesToIds(char_lex_it->second);
                    result.insert(result.end(), char_ids.begin(), char_ids.end());
                } else {
                    // Last resort: try direct token lookup for the character
                    auto char_token_it = token_to_id_.find(char_str);
                    if (char_token_it != token_to_id_.end()) {
                        result.push_back(char_token_it->second);
                    } else {
                        std::cerr << "Warning: No mapping for character: '" << char_str << "'" << std::endl;
                    }
                }
            } else {
                // No lexicon available, try direct token lookup
                auto char_token_it = token_to_id_.find(char_str);
                if (char_token_it != token_to_id_.end()) {
                    result.push_back(char_token_it->second);
                } else {
                    std::cerr << "Warning: No mapping for character: '" << char_str << "'" << std::endl;
                }
            }
        }
        
        return result;
    }
    
    std::vector<int64_t> convertPhonemesToIds(const std::string& phonemes) {
        std::vector<int64_t> ids;
        std::istringstream iss(phonemes);
        std::string phone;
        
        while (iss >> phone) {
            auto token_it = token_to_id_.find(phone);
            if (token_it != token_to_id_.end()) {
                ids.push_back(token_it->second);
            } else {
                // Try fallback mappings
                std::string mapped_phone = mapPhoneme(phone);
                if (mapped_phone != phone) {
                    auto mapped_it = token_to_id_.find(mapped_phone);
                    if (mapped_it != token_to_id_.end()) {
                        ids.push_back(mapped_it->second);
                    }
                }
            }
        }
        
        return ids;
    }
    
    
    std::string mapPunctuation(const std::string& punct) {
        // Try to find the punctuation directly in tokens first
        auto direct_it = token_to_id_.find(punct);
        if (direct_it != token_to_id_.end()) {
            return punct;
        }
        
        // Simple ASCII punctuation mappings
        if (punct == "！") return "!";
        if (punct == "？") return "?";
        if (punct == "，") return ",";
        if (punct == "。") return ".";
        if (punct == "：") return ":";
        if (punct == "；") return ";";
        if (punct == "、") return ",";
        // Skip quotes for now to avoid compilation issues
        if (punct == "'") return "'";
        if (punct == "'") return "'";
        if (punct == "—") return "-";  // em-dash to hyphen
        if (punct == "–") return "-";  // en-dash to hyphen
        if (punct == "…") return "..."; // ellipsis
        
        // Try to find common pause tokens for major punctuations
        if (punct == "。" || punct == "！" || punct == "？") {
            if (token_to_id_.count("sil")) return "sil";
            if (token_to_id_.count("sp")) return "sp";
            if (token_to_id_.count("<eps>")) return "<eps>";
        }
        
        return "";  // No mapping found
    }
    
    std::vector<int64_t> addBlankTokens(const std::vector<int64_t>& tokens) {
        // Matcha models expect blank tokens between phonemes
        // Use pad_id from model metadata (following sherpa-onnx approach)
        std::vector<int64_t> result(tokens.size() * 2 + 1, pad_id_);
        
        int32_t i = 1;
        for (auto token : tokens) {
            result[i] = token;
            i += 2;
        }
        
        return result;
    }
    
    std::vector<float> runAcousticModel(const std::vector<int64_t>& tokens, int speaker_id, float speed) {
        // Prepare inputs according to actual model signature
        std::vector<int64_t> token_shape = {1, static_cast<int64_t>(tokens.size())};
        std::vector<int64_t> length_data = {static_cast<int64_t>(tokens.size())};
        std::vector<int64_t> length_shape = {1};
        std::vector<float> noise_scale_data = {config_.noise_scale};
        std::vector<int64_t> noise_scale_shape = {1};
        std::vector<float> length_scale_data = {speed * config_.length_scale};
        std::vector<int64_t> length_scale_shape = {1};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Create input tensors according to model signature
        std::vector<Ort::Value> input_tensors;
        
        // Input 0: x (tokens)
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, const_cast<int64_t*>(tokens.data()), tokens.size(),
            token_shape.data(), token_shape.size()
        ));
        
        // Input 1: x_length
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), 1,
            length_shape.data(), length_shape.size()
        ));
        
        // Input 2: noise_scale
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, noise_scale_data.data(), 1,
            noise_scale_shape.data(), noise_scale_shape.size()
        ));
        
        // Input 3: length_scale
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, length_scale_data.data(), 1,
            length_scale_shape.data(), length_scale_shape.size()
        ));
        
        // Get actual input names from the model
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        
        
        // Use correct input names based on model signature
        const char* model_input_names[] = {"x", "x_length", "noise_scale", "length_scale"};
        const char* model_output_names[] = {"mel"};
        size_t num_inputs = 4;
        
        // Lock for thread-safe ONNX Runtime inference on RISC-V
        std::lock_guard<std::mutex> lock(inference_mutex_);
        auto output_tensors = acoustic_model_->Run(
            Ort::RunOptions{nullptr},
            model_input_names, input_tensors.data(), num_inputs,
            model_output_names, 1
        );
        
        // Extract mel spectrogram
        float* mel_data = output_tensors[0].GetTensorMutableData<float>();
        auto mel_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t mel_size = std::accumulate(mel_shape.begin(), mel_shape.end(),
                                         1, std::multiplies<size_t>());
        
        return std::vector<float>(mel_data, mel_data + mel_size);
    }
    
private:
    TTSConfig config_;
    bool initialized_ = false;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> acoustic_model_;
    std::unique_ptr<Ort::Session> vocoder_model_;

    // Text processing
    std::unique_ptr<cppjieba::Jieba> jieba_;
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::unordered_map<std::string, std::string> lexicon_;

    // cpp-pinyin for zh-en bilingual model
    std::unique_ptr<Pinyin::Pinyin> pinyin_converter_;
    
    // Model info
    int mel_dim_ = 80;
    int num_speakers_ = 1;
    int64_t pad_id_ = 0;
    
    // Thread safety for ONNX Runtime on RISC-V
    mutable std::mutex inference_mutex_;
    
    // ISTFT parameters from vocoder metadata
    int32_t istft_n_fft_ = 1024;
    int32_t istft_hop_length_ = 256;
    int32_t istft_win_length_ = 1024;
    bool istft_center_ = true;
    bool istft_normalized_ = false;
    std::string istft_window_type_ = "hann";
    std::string istft_pad_mode_ = "reflect";
};

// TTSModel public interface
TTSModel::TTSModel(const TTSConfig& config) 
    : pImpl(std::make_unique<Impl>(config)) {
}

TTSModel::~TTSModel() = default;

bool TTSModel::initialize() {
    return pImpl->initialize();
}

GeneratedAudio TTSModel::generate(const std::string& text) {
    return pImpl->generate(text, 0, 1.0f);  // Default speaker ID 0
}

GeneratedAudio TTSModel::generate(const std::string& text, int speaker_id) {
    return pImpl->generate(text, speaker_id, 1.0f);
}

GeneratedAudio TTSModel::generate(const std::string& text, int speaker_id, float speed) {
    return pImpl->generate(text, speaker_id, speed);
}

bool TTSModel::isInitialized() const {
    return pImpl->isInitialized();
}

int TTSModel::getNumSpeakers() const {
    return pImpl->getNumSpeakers();
}

int TTSModel::getSampleRate() const {
    return pImpl->getSampleRate();
}

} // namespace tts