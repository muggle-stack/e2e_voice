#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <functional>

#include "audio_recorder.hpp"
#include "vad_detector.hpp"
#include "asr_model.hpp"
#include "model_downloader.hpp"
#include "ollama.hpp"

class ASRLLMDemo {
public:
    struct RecorderParams {
        int sample_rate;
        int channels;
        int device_index;
        double silence_duration;
        double max_record_time;
        double trigger_threshold;
        double stop_threshold;
        std::string vad_type;
        std::string llm_model;
        
        RecorderParams() :
            sample_rate(16000),
            channels(1),
            device_index(6),
            silence_duration(1.0),
            max_record_time(5.0),
            trigger_threshold(0.6),
            stop_threshold(0.35),
            vad_type("energy"),
            llm_model("qwen2.5:0.5b") {}
    };

    ASRLLMDemo(const RecorderParams& params = RecorderParams()) : recorder_params_(params) {}
    ~ASRLLMDemo() = default;

    bool initialize() {
        std::cout << "Initializing ASR-LLM Demo..." << std::endl;
        
        // Check if Ollama server is running
        if (!ollama::is_running()) {
            std::cerr << "Error: Ollama server is not running. Please start ollama service first." << std::endl;
            std::cerr << "Run: sudo systemctl start ollama" << std::endl;
            return false;
        }
        std::cout << "Ollama server is running (version: " << ollama::get_version() << ")" << std::endl;
        
        // Check if the specified model is available
        std::vector<std::string> models = ollama::list_models();
        bool model_found = false;
        for (const auto& model : models) {
            if (model == recorder_params_.llm_model) {
                model_found = true;
                break;
            }
        }
        
        if (!model_found) {
            std::cout << "Model '" << recorder_params_.llm_model << "' not found locally. Attempting to pull..." << std::endl;
            if (!ollama::pull_model(recorder_params_.llm_model)) {
                std::cerr << "Failed to pull model: " << recorder_params_.llm_model << std::endl;
                std::cerr << "Please ensure the model name is correct or pull it manually:" << std::endl;
                std::cerr << "ollama pull " << recorder_params_.llm_model << std::endl;
                return false;
            }
            std::cout << "Model pulled successfully!" << std::endl;
        } else {
            std::cout << "Using existing model: " << recorder_params_.llm_model << std::endl;
        }
        
        // Download ASR models if needed
        ModelDownloader downloader;
        if (!downloader.ensureModelsExist()) {
            std::cerr << "Failed to ensure ASR models exist" << std::endl;
            return false;
        }
        
        // Initialize VAD detector if using Silero VAD
        if (recorder_params_.vad_type == "silero") {
            VADDetector::Config vad_config;
            vad_config.model_path = downloader.getModelPath(ModelDownloader::VAD_MODEL_NAME);
            vad_config.sample_rate = 16000;
            vad_config.window_size = 512;  // Silero VAD expects 512 samples (32ms at 16kHz)
            vad_config.context_size = 64;   // Additional context for model
            
            vad_detector_ = std::make_unique<VADDetector>(vad_config);
            if (!vad_detector_->initialize()) {
                std::cerr << "Failed to initialize Silero VAD detector" << std::endl;
                return false;
            }
            std::cout << "Using Silero VAD for voice activity detection" << std::endl;
        } else {
            std::cout << "Using energy-based VAD for voice activity detection" << std::endl;
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
        
        // Initialize audio recorder
        AudioRecorder::Config recorder_config;
        recorder_config.sample_rate = recorder_params_.sample_rate;
        recorder_config.channels = recorder_params_.channels;
        recorder_config.frames_per_buffer = 512;
        recorder_config.device_index = recorder_params_.device_index;
        recorder_config.silence_duration = recorder_params_.silence_duration;
        recorder_config.max_record_time = recorder_params_.max_record_time;
        recorder_config.trigger_threshold = recorder_params_.trigger_threshold;
        recorder_config.stop_threshold = recorder_params_.stop_threshold;
        recorder_config.vad_type = recorder_params_.vad_type;
        
        audio_recorder_ = std::make_unique<AudioRecorder>(recorder_config);
        if (!audio_recorder_->initialize()) {
            std::cerr << "Failed to initialize audio recorder" << std::endl;
            return false;
        }
        
        // Set up VAD callback if using Silero VAD
        if (recorder_params_.vad_type == "silero" && vad_detector_) {
            audio_recorder_->setVADDetector(vad_detector_.get());
        }
        
        std::cout << "ASR-LLM Demo initialized successfully!" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "\n=== ASR-LLM Demo Started ===" << std::endl;
        std::cout << "This demo will:" << std::endl;
        std::cout << "1. Record your speech using ASR" << std::endl;
        std::cout << "2. Convert speech to text" << std::endl;
        std::cout << "3. Send text to LLM (" << recorder_params_.llm_model << ") for processing" << std::endl;
        std::cout << "4. Display the LLM response" << std::endl;
        std::cout << "\nPress Enter to start recording, or 'q' to quit" << std::endl;
        
        std::string input;
        while (true) {
            std::cout << "\nPress Enter to record (or 'q' to quit): ";
            std::getline(std::cin, input);
            
            if (input == "q" || input == "quit" || input == "exit") {
                break;
            }
            
            recordRecognizeAndGenerate();
        }
        
        std::cout << "Demo finished." << std::endl;
    }

private:
    void recordRecognizeAndGenerate() {
        std::cout << "\n=== Recording Phase ===" << std::endl;
        std::cout << "Starting recording..." << std::endl;
        std::cout << "Speak now! (max " << recorder_params_.max_record_time 
                  << " seconds, or silence for " << recorder_params_.silence_duration 
                  << " second to stop)" << std::endl;
        
        // Record audio
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> audio = audio_recorder_->recordAudio();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (audio.empty()) {
            std::cout << "No audio recorded or recording failed" << std::endl;
            return;
        }
        
        auto recording_duration = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Recording completed (" << recording_duration << "s, " 
                  << audio.size() << " samples at " << recorder_params_.sample_rate << "Hz)" << std::endl;
        
        // Convert stereo to mono if necessary
        std::vector<float> mono_audio;
        if (recorder_params_.channels == 2) {
            mono_audio.reserve(audio.size() / 2);
            for (size_t i = 0; i < audio.size(); i += 2) {
                // Average left and right channels
                float left = audio[i];
                float right = audio[i + 1];
                mono_audio.push_back((left + right) / 2.0f);
            }
        } else {
            mono_audio = audio;
        }
        
        // Resample if necessary
        std::vector<float> resampled_audio;
        if (recorder_params_.sample_rate != 16000) {
            std::cout << "Resampling from " << recorder_params_.sample_rate << "Hz to 16000Hz..." << std::endl;
            auto resample_start = std::chrono::high_resolution_clock::now();
            resampled_audio = resampleAudio(mono_audio, recorder_params_.sample_rate, 16000);
            auto resample_end = std::chrono::high_resolution_clock::now();
            auto resample_time = std::chrono::duration<double>(resample_end - resample_start).count();
            std::cout << "Resampled to " << resampled_audio.size() << " samples in " 
                      << resample_time << "s" << std::endl;
        } else {
            resampled_audio = mono_audio;
        }
        
        // Recognize speech
        std::cout << "\n=== ASR Phase ===" << std::endl;
        std::cout << "Processing audio for speech recognition..." << std::endl;
        
        start_time = std::chrono::high_resolution_clock::now();
        std::string asr_result = asr_model_->recognize(resampled_audio);
        end_time = std::chrono::high_resolution_clock::now();
        
        auto asr_duration = std::chrono::duration<double>(end_time - start_time).count();
        
        if (asr_result.empty()) {
            std::cout << "No speech recognized" << std::endl;
            return;
        }
        
        std::cout << "ASR Result: " << asr_result << std::endl;
        std::cout << "ASR Processing time: " << asr_duration << "s" << std::endl;
        
        double audio_duration = static_cast<double>(resampled_audio.size()) / 16000.0;
        double rtf = asr_duration / audio_duration;
        std::cout << "ASR Real-time factor: " << rtf << std::endl;
        
        // Generate LLM response with streaming
        std::cout << "\n=== LLM Phase ===" << std::endl;
        std::cout << "Sending to LLM (" << recorder_params_.llm_model << ")..." << std::endl;
        
        try {
            start_time = std::chrono::high_resolution_clock::now();
            
            // Set reasonable options for the LLM
            ollama::options options;
            options["temperature"] = 0.7;
            options["max_tokens"] = 200;
            
            std::cout << "\n=== Results ===" << std::endl;
            std::cout << "Your speech: " << asr_result << std::endl;
            std::cout << "LLM Response: " << std::flush;
            
            // Create streaming callback function
            bool stream_finished = false;
            auto stream_callback = [&stream_finished](const ollama::response& response) -> bool {
                // Print each token as it arrives
                std::cout << response.as_simple_string() << std::flush;
                
                // Check if this is the final response
                if (response.as_json()["done"] == true) {
                    stream_finished = true;
                    std::cout << std::endl; // Add newline at the end
                }
                
                // Return true to continue streaming, false to stop
                return !stream_finished;
            };
            
            // Use streaming generation
            ollama::generate(recorder_params_.llm_model, asr_result, stream_callback, options);
            end_time = std::chrono::high_resolution_clock::now();
            
            auto llm_duration = std::chrono::duration<double>(end_time - start_time).count();
            std::cout << "LLM Processing time: " << llm_duration << "s" << std::endl;
            
        } catch (const ollama::exception& e) {
            std::cerr << "LLM Error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during LLM processing: " << e.what() << std::endl;
        }
    }
    
    // Fast decimation resampling (for common ratios like 48k->16k)
    std::vector<float> resampleAudio(const std::vector<float>& input, int from_rate, int to_rate) {
        if (from_rate == to_rate) {
            return input;
        }
        
        // For 48kHz to 16kHz, we can simply take every 3rd sample (48/16 = 3)
        if (from_rate == 48000 && to_rate == 16000) {
            std::vector<float> output;
            output.reserve(input.size() / 3);
            for (size_t i = 0; i < input.size(); i += 3) {
                output.push_back(input[i]);
            }
            return output;
        }
        
        // For other ratios, use simple decimation
        double ratio = static_cast<double>(from_rate) / to_rate;
        size_t output_size = static_cast<size_t>(input.size() / ratio);
        std::vector<float> output;
        output.reserve(output_size);
        
        for (size_t i = 0; i < output_size; ++i) {
            size_t src_idx = static_cast<size_t>(i * ratio);
            if (src_idx < input.size()) {
                output.push_back(input[src_idx]);
            }
        }
        
        return output;
    }
    
    std::unique_ptr<AudioRecorder> audio_recorder_;
    std::unique_ptr<VADDetector> vad_detector_;
    std::unique_ptr<ASRModel> asr_model_;
    RecorderParams recorder_params_;
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --sample_rate <value>       Audio sample rate (default: 16000)" << std::endl;
    std::cout << "  --channels <value>          Number of audio channels (default: 1)" << std::endl;
    std::cout << "  --device_index <value>      Audio device index (default: 6)" << std::endl;
    std::cout << "  --silence_duration <value>  Silence duration to stop recording in seconds (default: 1.0)" << std::endl;
    std::cout << "  --max_record_time <value>   Maximum recording time in seconds (default: 5.0)" << std::endl;
    std::cout << "  --trigger_threshold <value> VAD trigger threshold (default: 0.6)" << std::endl;
    std::cout << "  --stop_threshold <value>    VAD stop threshold (default: 0.35)" << std::endl;
    std::cout << "  --vad_type <type>           VAD type: 'energy' or 'silero' (default: energy)" << std::endl;
    std::cout << "  --model <model_name>        LLM model name (default: qwen2.5:0.5b)" << std::endl;
    std::cout << "  --help                      Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "ASR-LLM C++ Demo Application" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Parse command line arguments
    ASRLLMDemo::RecorderParams params;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--sample_rate" && i + 1 < argc) {
            params.sample_rate = std::atoi(argv[++i]);
        }
        else if (arg == "--channels" && i + 1 < argc) {
            params.channels = std::atoi(argv[++i]);
        }
        else if (arg == "--device_index" && i + 1 < argc) {
            params.device_index = std::atoi(argv[++i]);
        }
        else if (arg == "--silence_duration" && i + 1 < argc) {
            params.silence_duration = std::atof(argv[++i]);
        }
        else if (arg == "--max_record_time" && i + 1 < argc) {
            params.max_record_time = std::atof(argv[++i]);
        }
        else if (arg == "--trigger_threshold" && i + 1 < argc) {
            params.trigger_threshold = std::atof(argv[++i]);
        }
        else if (arg == "--stop_threshold" && i + 1 < argc) {
            params.stop_threshold = std::atof(argv[++i]);
        }
        else if (arg == "--vad_type" && i + 1 < argc) {
            params.vad_type = argv[++i];
            if (params.vad_type != "energy" && params.vad_type != "silero") {
                std::cerr << "Invalid VAD type: " << params.vad_type << ". Must be 'energy' or 'silero'" << std::endl;
                return 1;
            }
        }
        else if (arg == "--model" && i + 1 < argc) {
            params.llm_model = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Print configuration
    std::cout << "\nRecorder Configuration:" << std::endl;
    std::cout << "  Sample rate: " << params.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << params.channels << std::endl;
    std::cout << "  Device index: " << params.device_index << std::endl;
    std::cout << "  Silence duration: " << std::fixed << std::setprecision(1) 
              << params.silence_duration << " seconds" << std::endl;
    std::cout << "  Max record time: " << params.max_record_time << " seconds" << std::endl;
    std::cout << "  Trigger threshold: " << std::setprecision(2) 
              << params.trigger_threshold << std::endl;
    std::cout << "  Stop threshold: " << params.stop_threshold << std::endl;
    std::cout << "  VAD type: " << params.vad_type << std::endl;
    std::cout << "  LLM model: " << params.llm_model << std::endl;
    std::cout << std::endl;
    
    try {
        ASRLLMDemo demo(params);
        if (!demo.initialize()) {
            std::cerr << "Failed to initialize demo" << std::endl;
            return 1;
        }
        
        demo.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}