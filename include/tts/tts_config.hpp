#ifndef TTS_CONFIG_HPP
#define TTS_CONFIG_HPP

#include <string>

namespace tts {

struct TTSConfig {
    // Model paths
    std::string acoustic_model_path;  // Matcha acoustic model
    std::string vocoder_path;          // Vocoder model (HiFiGAN/Vocos)
    std::string lexicon_path;          // Lexicon file for pronunciation
    std::string tokens_path;           // Token vocabulary
    std::string dict_dir;              // Model dictionary directory (for lexicon)
    std::string jieba_dict_dir;        // Jieba dictionary directory (for Chinese segmentation)
    std::string data_dir;              // espeak-ng data directory (for English)
    
    // Model parameters
    float noise_scale = 1.0f;          // Controls variation (0.5-2.0)
    float length_scale = 1.0f;         // Speech speed (>1.0 = slower)
    int speaker_id = 0;                // For multi-speaker models
    
    // Runtime parameters
    int sample_rate = 22050;           // Output sample rate
    int max_num_sentences = 5;         // Max sentences per batch
    std::string language = "zh";       // Language: "zh" only (English not supported)
    
    // Audio normalization parameters
    float target_rms = 0.15f;          // Target RMS level (0.15 = -16.5dB, louder)
    float compression_ratio = 2.0f;    // Dynamic range compression ratio (gentler)
    float compression_threshold = 0.7f; // Threshold for compression (0-1)
    bool use_rms_norm = true;          // Use RMS normalization instead of peak
    
    // Model type
    std::string model_type = "matcha"; // Currently only supporting matcha
    
    // Performance optimization
    bool enable_warmup = true;         // Warm up models during initialization
    
    // Audio quality
    bool remove_clicks = true;         // Apply click/pop removal post-processing
    
    TTSConfig() = default;
};

} // namespace tts

#endif // TTS_CONFIG_HPP