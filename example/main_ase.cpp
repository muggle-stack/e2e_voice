#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <sndfile.h>

#include "asr_model.hpp"
#include "model_downloader.hpp"

class ASEDemo {
public:
    ASEDemo() = default;
    ~ASEDemo() = default;

    bool initialize() {
        std::cout << "Initializing ASE (Audio Speech Engine) Demo..." << std::endl;
        
        // Download models if needed
        ModelDownloader downloader;
        if (!downloader.ensureModelsExist()) {
            std::cerr << "Failed to ensure models exist" << std::endl;
            return false;
        }
        
        // Initialize ASR model
        ASRModel::Config asr_config;
        asr_config.model_path = downloader.getModelPath(ModelDownloader::ASR_MODEL_QUANT_NAME);
        asr_config.config_path = downloader.getModelPath(ModelDownloader::CONFIG_NAME);
        asr_config.vocab_path = downloader.getModelPath(ModelDownloader::VOCAB_NAME);
        asr_config.decoder_path = downloader.getModelPath(ModelDownloader::DECODER_NAME);
        asr_config.sample_rate = 16000;
        asr_config.language = "zh";
        asr_config.use_itn = true;
        asr_config.quantized = true;
        
        asr_model_ = std::make_unique<ASRModel>(asr_config);
        if (!asr_model_->initialize()) {
            std::cerr << "Failed to initialize ASR model" << std::endl;
            return false;
        }
        
        std::cout << "ASE Demo initialized successfully!" << std::endl;
        return true;
    }
    
    std::vector<float> loadAudioFile(const std::string& filename) {
        SF_INFO sf_info;
        memset(&sf_info, 0, sizeof(sf_info));
        
        SNDFILE* sf = sf_open(filename.c_str(), SFM_READ, &sf_info);
        if (!sf) {
            std::cerr << "Error opening audio file: " << filename << std::endl;
            std::cerr << "Error: " << sf_strerror(sf) << std::endl;
            return {};
        }
        
        std::cout << "Audio file info:" << std::endl;
        std::cout << "  Sample rate: " << sf_info.samplerate << " Hz" << std::endl;
        std::cout << "  Channels: " << sf_info.channels << std::endl;
        std::cout << "  Frames: " << sf_info.frames << std::endl;
        std::cout << "  Duration: " << std::fixed << std::setprecision(2) 
                  << (double)sf_info.frames / sf_info.samplerate << " seconds" << std::endl;

        // Read audio data
        std::vector<float> audio_data(sf_info.frames * sf_info.channels);
        sf_count_t frames_read = sf_readf_float(sf, audio_data.data(), sf_info.frames);
        if (frames_read != sf_info.frames) {
            std::cerr << "Warning: Only read " << frames_read << " frames out of " << sf_info.frames << std::endl;
        }
        
        sf_close(sf);
        
        // Convert to mono if stereo
        if (sf_info.channels > 1) {
            std::vector<float> mono_audio(sf_info.frames);
            for (int i = 0; i < sf_info.frames; i++) {
                float sum = 0.0f;
                for (int ch = 0; ch < sf_info.channels; ch++) {
                    sum += audio_data[i * sf_info.channels + ch];
                }
                mono_audio[i] = sum / sf_info.channels;
            }
            audio_data = std::move(mono_audio);
        }
        
        // Resample to 16kHz if needed
        if (sf_info.samplerate != 16000) {
            std::cout << "Resampling from " << sf_info.samplerate << " Hz to 16000 Hz..." << std::endl;
            audio_data = resampleAudio(audio_data, sf_info.samplerate, 16000);
        }
        
        return audio_data;
    }
    
    std::vector<float> resampleAudio(const std::vector<float>& input, int input_rate, int output_rate) {
        // Simple linear interpolation resampling
        if (input_rate == output_rate) {
            return input;
        }
        
        double ratio = (double)input_rate / output_rate;
        size_t output_length = (size_t)(input.size() / ratio);
        std::vector<float> output(output_length);
        
        for (size_t i = 0; i < output_length; i++) {
            double src_index = i * ratio;
            size_t index = (size_t)src_index;
            double frac = src_index - index;
            
            if (index + 1 < input.size()) {
                output[i] = input[index] * (1.0 - frac) + input[index + 1] * frac;
            } else {
                output[i] = input[index];
            }
        }
        
        return output;
    }
    
    void processAudioFile(const std::string& filename) {
        std::cout << "\n=== Processing Audio File: " << filename << " ===" << std::endl;
        
        // Load audio file
        auto audio_data = loadAudioFile(filename);
        if (audio_data.empty()) {
            std::cerr << "Failed to load audio file: " << filename << std::endl;
            return;
        }
        
        // Perform ASR
        std::cout << "\nPerforming speech recognition..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::string result = asr_model_->recognize(audio_data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Display results
        std::cout << "\n=== ASR Results ===" << std::endl;
        std::cout << "Recognition Result: " << result << std::endl;
        std::cout << "Processing Time: " << duration.count() << " ms" << std::endl;
        
        // Calculate real-time factor
        double audio_duration = (double)audio_data.size() / 16000.0;
        double processing_time = duration.count() / 1000.0;
        double rtf = processing_time / audio_duration;
        std::cout << "Real-time Factor: " << std::fixed << std::setprecision(3) << rtf << std::endl;
        std::cout << "===================" << std::endl;
    }

private:
    std::unique_ptr<ASRModel> asr_model_;
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] <audio_file>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help           Show this help message" << std::endl;
    std::cout << "  -v, --version        Show version information" << std::endl;
    std::cout << "\nSupported audio formats:" << std::endl;
    std::cout << "  WAV, FLAC, OGG, MP3 (and other formats supported by libsndfile)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " audio.wav" << std::endl;
    std::cout << "  " << program_name << " recording.flac" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string audio_file;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            std::cout << "ASE Demo v1.0.0" << std::endl;
            std::cout << "Audio Speech Engine for file-based ASR" << std::endl;
            return 0;
        } else if (arg.length() > 0 && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        } else {
            audio_file = arg;
        }
    }
    
    if (audio_file.empty()) {
        std::cerr << "Error: No audio file specified" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        ASEDemo demo;
        if (!demo.initialize()) {
            std::cerr << "Failed to initialize ASE Demo" << std::endl;
            return 1;
        }
        
        demo.processAudioFile(audio_file);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}