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
#include <portaudio.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "audio_recorder.hpp"
#include "vad_detector.hpp"
#include "asr_model.hpp"
#include "model_downloader.hpp"
#include "api_comm.hpp"
#include "tts/tts_model.hpp"
#include "tts/tts_model_downloader.hpp"
#include "text_buffer.hpp"
#include "ordered_audio_queue.hpp"

class ASRLLMTTSApiDemo {
public:
    struct Params {
        // Audio recording params
        int sample_rate;
        int channels;
        int device_index;
        double silence_duration;
        double max_record_time;
        double trigger_threshold;
        double stop_threshold;
        std::string vad_type;
        
        // LLM params
        std::string llm_model;
        int max_tokens;
        
        // TTS params
        float tts_speed;
        int tts_speaker_id;
        float target_rms;
        float compression_ratio;
        float compression_threshold;
        bool use_rms_norm;
        
        // API params
        std::string api_key;
        std::string api_url;
        std::string env_file;
        
        Params() :
            sample_rate(16000),
            channels(1),
            device_index(6),
            silence_duration(1.0),
            max_record_time(5.0),
            trigger_threshold(0.6),
            stop_threshold(0.35),
            vad_type("energy"),
            llm_model("deepseek-chat"),
            max_tokens(500),
            tts_speed(1.0f),
            tts_speaker_id(0),
            target_rms(0.15f),
            compression_ratio(2.0f),
            compression_threshold(0.7f),
            use_rms_norm(true),
            env_file(".env") {}
    };

    ASRLLMTTSApiDemo(const Params& params = Params()) : params_(params) {}
    ~ASRLLMTTSApiDemo() {
        // Stop TTS worker thread
        tts_stop_flag_ = true;
        tts_queue_cv_.notify_all();
        if (tts_worker_thread_.joinable()) {
            tts_worker_thread_.join();
        }
    }

    bool initialize() {
        std::cout << "Initializing ASR-LLM-TTS API Demo..." << std::endl;
        
        // Configure API
        if (!params_.api_key.empty()) {
            api_comm::setApiKey(params_.api_key);
            std::cout << "API key set from command line" << std::endl;
        }
        
        if (!params_.api_url.empty()) {
            api_comm::setApiUrl(params_.api_url);
            std::cout << "API URL set from command line: " << params_.api_url << std::endl;
        }
        
        // Try to load from .env file
        if (api_comm::loadConfigFromEnv(params_.env_file)) {
            std::cout << "API configuration loaded from " << params_.env_file << std::endl;
        }
        
        // Try environment variables as fallback
        if (!api_comm::getClient().isConfigured()) {
            const char* env_key = std::getenv("API_KEY");
            if (!env_key) env_key = std::getenv("OPENAI_API_KEY");
            if (!env_key) env_key = std::getenv("DEEPSEEK_API_KEY");
            
            if (env_key) {
                api_comm::setApiKey(env_key);
                std::cout << "API key loaded from environment variable" << std::endl;
            }
            
            const char* env_url = std::getenv("API_URL");
            if (env_url) {
                api_comm::setApiUrl(env_url);
                std::cout << "API URL loaded from environment variable" << std::endl;
            }
        }
        
        if (!api_comm::getClient().isConfigured()) {
            std::cerr << "Error: API not configured. Please set API key and URL via:" << std::endl;
            std::cerr << "  1. Command line: --api_key YOUR_KEY --api_url YOUR_URL" << std::endl;
            std::cerr << "  2. Environment variables: export API_KEY=YOUR_KEY API_URL=YOUR_URL" << std::endl;
            std::cerr << "  3. .env file: API_KEY=YOUR_KEY and API_URL=YOUR_URL" << std::endl;
            std::cerr << "\nSupported APIs (use corresponding URL):" << std::endl;
            std::cerr << "  - DeepSeek: https://api.deepseek.com/chat/completions" << std::endl;
            std::cerr << "  - OpenAI: https://api.openai.com/v1/chat/completions" << std::endl;
            std::cerr << "  - Local: http://localhost:8000/v1/chat/completions" << std::endl;
            return false;
        }
        
        std::cout << "API configured successfully (" << api_comm::getClient().getApiProvider() << ")" << std::endl;
        std::cout << "Using model: " << params_.llm_model << std::endl;
        
        // Download ASR models if needed
        ModelDownloader downloader;
        if (!downloader.ensureModelsExist()) {
            std::cerr << "Failed to ensure ASR models exist" << std::endl;
            return false;
        }
        
        // Download TTS models if needed
        tts::TTSModelDownloader tts_downloader;
        if (!tts_downloader.ensureModelsExist()) {
            std::cerr << "Failed to ensure TTS models exist" << std::endl;
            return false;
        }
        
        // Initialize VAD detector if using Silero VAD
        if (params_.vad_type == "silero") {
            VADDetector::Config vad_config;
            vad_config.model_path = downloader.getModelPath(ModelDownloader::VAD_MODEL_NAME);
            vad_config.sample_rate = 16000;
            vad_config.window_size = 512;
            vad_config.context_size = 64;
            
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
        
        // Initialize TTS model
        tts::TTSConfig tts_config;
        tts_config.acoustic_model_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_MODEL);
        tts_config.vocoder_path = tts_downloader.getModelPath(tts::TTSModelDownloader::VOCOS_VOCODER);
        tts_config.lexicon_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_LEXICON);
        tts_config.tokens_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_TOKENS);
        tts_config.dict_dir = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_DICT_DIR);
        tts_config.jieba_dict_dir = "";  // Will auto-detect cppjieba location
        tts_config.language = "zh";
        tts_config.sample_rate = 22050;
        tts_config.noise_scale = 1.0f;
        tts_config.length_scale = params_.tts_speed;
        tts_config.target_rms = params_.target_rms;
        tts_config.compression_ratio = params_.compression_ratio;
        tts_config.compression_threshold = params_.compression_threshold;
        tts_config.use_rms_norm = params_.use_rms_norm;
        
        tts_model_ = std::make_unique<tts::TTSModel>(tts_config);
        if (!tts_model_->initialize()) {
            std::cerr << "Failed to initialize TTS model" << std::endl;
            return false;
        }
        std::cout << "TTS model initialized (sample rate: " << tts_model_->getSampleRate() << "Hz)" << std::endl;
        
        // Initialize audio recorder
        AudioRecorder::Config recorder_config;
        recorder_config.sample_rate = params_.sample_rate;
        recorder_config.channels = params_.channels;
        recorder_config.frames_per_buffer = 512;
        recorder_config.device_index = params_.device_index;
        recorder_config.silence_duration = params_.silence_duration;
        recorder_config.max_record_time = params_.max_record_time;
        recorder_config.trigger_threshold = params_.trigger_threshold;
        recorder_config.stop_threshold = params_.stop_threshold;
        recorder_config.vad_type = params_.vad_type;
        
        audio_recorder_ = std::make_unique<AudioRecorder>(recorder_config);
        if (!audio_recorder_->initialize()) {
            std::cerr << "Failed to initialize audio recorder" << std::endl;
            return false;
        }
        
        // Set up VAD callback if using Silero VAD
        if (params_.vad_type == "silero" && vad_detector_) {
            audio_recorder_->setVADDetector(vad_detector_.get());
        }
        
        // Initialize PortAudio for playback
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        
        // Initialize ordered audio queue for sequential TTS playback
        audio_queue_ = std::make_unique<OrderedAudioQueue>();
        audio_queue_->start();
        
        // Start TTS worker thread for RISC-V
        #if defined(__riscv) || defined(__riscv__)
        std::cout << "Starting TTS worker thread for RISC-V..." << std::endl;
        tts_worker_thread_ = std::thread(&ASRLLMTTSApiDemo::ttsWorker, this);
        #endif
        
        std::cout << "ASR-LLM-TTS API Demo initialized successfully!" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "\n=== ASR-LLM-TTS API Demo Started ===" << std::endl;
        std::cout << "This demo will:" << std::endl;
        std::cout << "1. Record your speech using ASR" << std::endl;
        std::cout << "2. Convert speech to text" << std::endl;
        std::cout << "3. Send text to " << api_comm::getClient().getApiProvider() 
                  << " (" << params_.llm_model << ") for processing" << std::endl;
        std::cout << "4. Convert API response to speech using TTS" << std::endl;
        std::cout << "5. Play the generated speech" << std::endl;
        std::cout << "\nPress Enter to start recording, or 'q' to quit" << std::endl;
        
        std::string input;
        while (true) {
            std::cout << "\nPress Enter to record (or 'q' to quit): ";
            std::getline(std::cin, input);
            
            if (input == "q" || input == "quit" || input == "exit") {
                break;
            }
            
            processVoiceInteraction();
        }
        
        // Stop audio queue and cleanup PortAudio
        if (audio_queue_) {
            audio_queue_->stop();
        }
        Pa_Terminate();
        
        std::cout << "Demo finished." << std::endl;
    }

private:
    void processVoiceInteraction() {
        // 1. Record audio
        std::cout << "\n=== Recording Phase ===" << std::endl;
        std::cout << "Starting recording..." << std::endl;
        std::cout << "Speak now! (max " << params_.max_record_time 
                  << " seconds, or silence for " << params_.silence_duration 
                  << " second to stop)" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> audio = audio_recorder_->recordAudio();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (audio.empty()) {
            std::cout << "No audio recorded or recording failed" << std::endl;
            return;
        }
        
        auto recording_duration = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Recording completed (" << recording_duration << "s, " 
                  << audio.size() << " samples at " << params_.sample_rate << "Hz)" << std::endl;
        
        // 2. Resample if necessary
        std::vector<float> resampled_audio;
        if (params_.sample_rate != 16000) {
            std::cout << "Resampling from " << params_.sample_rate << "Hz to 16000Hz..." << std::endl;
            auto resample_start = std::chrono::high_resolution_clock::now();
            resampled_audio = resampleAudio(audio, params_.sample_rate, 16000);
            auto resample_end = std::chrono::high_resolution_clock::now();
            auto resample_time = std::chrono::duration<double>(resample_end - resample_start).count();
            std::cout << "Resampled to " << resampled_audio.size() << " samples in " 
                      << resample_time << "s" << std::endl;
        } else {
            resampled_audio = audio;
        }
        
        // 3. Recognize speech
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
        
        // 4. Generate API response
        std::cout << "\n=== LLM Phase ===" << std::endl;
        std::cout << "Sending to " << api_comm::getClient().getApiProvider() 
                  << " (" << params_.llm_model << ")..." << std::endl;
        
        // Reset audio queue order for new conversation
        audio_queue_->resetOrder();
        
        std::string llm_response;
        try {
            start_time = std::chrono::high_resolution_clock::now();
            
            // Set reasonable options for the LLM
            api_comm::Options options;
            options.temperature = 0.7f;
            options.max_tokens = params_.max_tokens;
            options.stream = true;
            
            // Use streaming generation with text buffer
            TextBuffer text_buffer;
            std::vector<std::string> processed_sentences;
            size_t sentence_order = 0;  // Sentence order counter
            
            std::cout << "API Response: " << std::flush;
            
            // Create streaming callback function
            bool stream_finished = false;
            auto stream_callback = [&](const api_comm::Response& response) -> bool {
                // Check if this is the final response
                if (response.raw_json.contains("done") && response.raw_json["done"] == true) {
                    stream_finished = true;
                    std::cout << std::endl;
                    
                    // Process any remaining text in buffer as final sentence
                    text_buffer.addText("。"); // Add period to force final sentence
                    while (text_buffer.hasSentence()) {
                        std::string sentence = text_buffer.getNextSentence();
                        if (!sentence.empty()) {
                            processed_sentences.push_back(sentence);
                            size_t current_order = sentence_order++;
                            
                            #if defined(__riscv) || defined(__riscv__)
                            // On RISC-V, enqueue to worker thread
                            enqueueTTSTask(sentence, current_order);
                            #else
                            std::thread([this, sentence, current_order]() {
                                generateAndEnqueueOrderedTTS(sentence, current_order);
                            }).detach();
                            #endif
                        }
                    }
                    return false;
                }
                
                // Add text chunk to buffer
                std::string chunk = response.content;
                std::cout << chunk << std::flush;
                text_buffer.addText(chunk);
                
                // Process any complete sentences immediately
                while (text_buffer.hasSentence()) {
                    std::string sentence = text_buffer.getNextSentence();
                    if (!sentence.empty()) {
                        processed_sentences.push_back(sentence);
                        
                        // Assign order number to each sentence for sequential playback
                        size_t current_order = sentence_order++;
                        
                        // Limit concurrent threads on RISC-V to prevent overload
                        #if defined(__riscv) || defined(__riscv__)
                        // On RISC-V, enqueue to worker thread to avoid blocking LLM
                        enqueueTTSTask(sentence, current_order);
                        #else
                        // On other platforms, use threads for parallel processing
                        std::thread([this, sentence, current_order]() {
                            generateAndEnqueueOrderedTTS(sentence, current_order);
                        }).detach();
                        #endif
                    }
                }
                
                return !stream_finished;
            };
            
            // Use streaming generation
            bool success = api_comm::generate(params_.llm_model, asr_result, stream_callback, options);
            
            if (!success) {
                std::cerr << "\nFailed to generate response from API" << std::endl;
                return;
            }
            
            // Combine all sentences for final display
            llm_response = "";
            for (const auto& sentence : processed_sentences) {
                llm_response += sentence;
            }
            
            end_time = std::chrono::high_resolution_clock::now();
            auto llm_duration = std::chrono::duration<double>(end_time - start_time).count();
            
            std::cout << "LLM Processing time: " << llm_duration << "s" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error during LLM processing: " << e.what() << std::endl;
            return;
        }
        
        // TTS processing was done per sentence during streaming
        // Wait a bit for any remaining TTS generation to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Your question: " << asr_result << std::endl;
        std::cout << "AI response: " << llm_response << std::endl;
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
    
private:
    std::unique_ptr<AudioRecorder> audio_recorder_;
    std::unique_ptr<VADDetector> vad_detector_;
    std::unique_ptr<ASRModel> asr_model_;
    std::unique_ptr<tts::TTSModel> tts_model_;
    std::unique_ptr<OrderedAudioQueue> audio_queue_;
    Params params_;
    
    // TTS task queue for RISC-V to prevent blocking LLM streaming
    struct TTSTask {
        std::string sentence;
        size_t order;
    };
    std::queue<TTSTask> tts_queue_;
    std::mutex tts_queue_mutex_;
    std::condition_variable tts_queue_cv_;
    std::thread tts_worker_thread_;
    std::atomic<bool> tts_stop_flag_{false};
    
    void generateAndEnqueueOrderedTTS(const std::string& sentence, size_t order) {
        auto start_time = std::chrono::high_resolution_clock::now();
        tts::GeneratedAudio generated_audio = tts_model_->generate(sentence, params_.tts_speaker_id, params_.tts_speed);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto tts_duration = std::chrono::duration<double>(end_time - start_time).count();
        
        if (generated_audio.samples.empty()) {
            std::cout << "\n[TTS] Failed to generate speech #" << order << ": " << sentence << std::endl;
            return;
        }
        
        std::cout << "\n[TTS] Generated #" << order << " (" << generated_audio.duration() << "s audio in " << tts_duration << "s)" << std::endl;
        
        // Add to ordered playback queue
        OrderedAudioData audio_data(std::move(generated_audio.samples), generated_audio.sample_rate, sentence, order);
        audio_queue_->enqueue(audio_data);
    }
    
    void ttsWorker() {
        while (!tts_stop_flag_) {
            TTSTask task;
            bool has_task = false;
            
            // Get task from queue
            {
                std::unique_lock<std::mutex> lock(tts_queue_mutex_);
                tts_queue_cv_.wait(lock, [this] { 
                    return !tts_queue_.empty() || tts_stop_flag_; 
                });
                
                if (tts_stop_flag_) break;
                
                if (!tts_queue_.empty()) {
                    task = tts_queue_.front();
                    tts_queue_.pop();
                    has_task = true;
                }
            }
            
            // Process task outside of lock
            if (has_task) {
                generateAndEnqueueOrderedTTS(task.sentence, task.order);
            }
        }
    }
    
    void enqueueTTSTask(const std::string& sentence, size_t order) {
        {
            std::lock_guard<std::mutex> lock(tts_queue_mutex_);
            tts_queue_.push({sentence, order});
        }
        tts_queue_cv_.notify_one();
    }
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
    std::cout << "  --model <model_name>        LLM model name (default: deepseek-chat)" << std::endl;
    std::cout << "  --max_tokens <value>        Maximum tokens for LLM response (default: 500)" << std::endl;
    std::cout << "  --api_key <key>             API key for the service" << std::endl;
    std::cout << "  --api_url <url>             API endpoint URL" << std::endl;
    std::cout << "  --env_file <path>           Path to .env file (default: .env)" << std::endl;
    std::cout << "  --tts_speed <value>         TTS speech speed (default: 1.0, >1.0 = slower)" << std::endl;
    std::cout << "  --tts_speaker <value>       TTS speaker ID for multi-speaker models (default: 0)" << std::endl;
    std::cout << "  --target_rms <value>        Target RMS level for volume normalization (default: 0.15)" << std::endl;
    std::cout << "  --compression_ratio <value> Dynamic range compression ratio (default: 2.0)" << std::endl;
    std::cout << "  --use_peak_norm             Use peak normalization instead of RMS" << std::endl;
    std::cout << "  --help                      Show this help message" << std::endl;
    std::cout << "\nSupported APIs:" << std::endl;
    std::cout << "  - DeepSeek: --api_url https://api.deepseek.com/chat/completions" << std::endl;
    std::cout << "  - OpenAI:   --api_url https://api.openai.com/v1/chat/completions" << std::endl;
    std::cout << "  - Local:    --api_url http://localhost:8000/v1/chat/completions" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "ASR-LLM-TTS C++ Demo Application (Universal API)" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Parse command line arguments
    ASRLLMTTSApiDemo::Params params;
    
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
        else if (arg == "--max_tokens" && i + 1 < argc) {
            params.max_tokens = std::atoi(argv[++i]);
        }
        else if (arg == "--api_key" && i + 1 < argc) {
            params.api_key = argv[++i];
        }
        else if (arg == "--api_url" && i + 1 < argc) {
            params.api_url = argv[++i];
        }
        else if (arg == "--env_file" && i + 1 < argc) {
            params.env_file = argv[++i];
        }
        else if (arg == "--tts_speed" && i + 1 < argc) {
            params.tts_speed = std::atof(argv[++i]);
        }
        else if (arg == "--tts_speaker" && i + 1 < argc) {
            params.tts_speaker_id = std::atoi(argv[++i]);
        }
        else if (arg == "--target_rms" && i + 1 < argc) {
            params.target_rms = std::atof(argv[++i]);
        }
        else if (arg == "--compression_ratio" && i + 1 < argc) {
            params.compression_ratio = std::atof(argv[++i]);
        }
        else if (arg == "--use_peak_norm") {
            params.use_rms_norm = false;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Print configuration
    std::cout << "\nConfiguration:" << std::endl;
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
    std::cout << "  Max tokens: " << params.max_tokens << std::endl;
    std::cout << "  TTS speed: " << params.tts_speed << std::endl;
    std::cout << "  TTS speaker ID: " << params.tts_speaker_id << std::endl;
    std::cout << std::endl;
    
    try {
        ASRLLMTTSApiDemo demo(params);
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