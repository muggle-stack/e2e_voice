/**
 * ASR-LLM-TTS Demo with WebRTC AEC (Acoustic Echo Cancellation)
 *
 * This version supports:
 * - Full-duplex audio with echo cancellation
 * - TRUE Barge-in (interrupt TTS playback by speaking)
 * - Continuous listening mode (no need to press Enter)
 */

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
#include <atomic>
#include <unistd.h>
#include <fcntl.h>

#include "aec_audio_processor.hpp"
#include "vad_detector.hpp"
#ifdef USE_CLOUD_ASR
    #include "asr_realtime_api.hpp"
    using ASREngine = ASRRealtimeClient;
#else
    #include "asr_model.hpp"
    using ASREngine = ASRModel;
#endif
#include "model_downloader.hpp"
#ifdef USE_CLOUD_LLM
    #include "api_comm.hpp"
#else
    #include "ollama.hpp"
#endif
#include "tts/tts_model.hpp"
#include "tts/tts_model_downloader.hpp"
#include "text_buffer.hpp"

class ASRLLMTTSAECDemo {
public:
    struct Params {
        // Audio settings (48kHz for AEC)
        int sample_rate = 48000;
        int channels = 1;
        int input_device_index = -1;
        int output_device_index = -1;

        // Channel selection
        int input_channels = 1;       // Number of channels to open
        int selected_channel = 0;     // Which channel to use (0-based), -1 for mix

        // VAD settings
        std::string vad_type = "silero";
        double trigger_threshold = 0.6;
        double stop_threshold = 0.35;
        double silence_duration = 1.0;
        double max_record_time = 10.0;

        // LLM params
        std::string llm_model;
        int max_tokens;
        std::string api_key;
        std::string api_url;
        std::string env_file;

        // TTS params
        float tts_speed = 1.0f;
        int tts_speaker_id = 0;
        float target_rms = 0.15f;
        float compression_ratio = 2.0f;
        float compression_threshold = 0.7f;
        bool use_rms_norm = true;
        std::string tts_type = "zh";
        int output_sample_rate = 0;

        // Speaker recognition params
        bool enable_speaker_recognition = false;
        float speaker_threshold = 0.6f;
        std::string speaker_database = "speakers.db";

        // AEC/Barge-in settings
        bool aec_enabled = true;
        bool barge_in_enabled = true;
        double barge_in_threshold = 0.5;

        Params() {
#ifdef USE_CLOUD_LLM
            llm_model = "deepseek-chat";
            max_tokens = 500;
            env_file = ".env";
#else
            llm_model = "qwen2.5:0.5b";
            max_tokens = 100;
#endif
        }
    };

    ASRLLMTTSAECDemo(const Params& params = Params()) : params_(params) {}

    ~ASRLLMTTSAECDemo() {
        stop();
    }

    bool initialize() {
        std::cout << "Initializing ASR-LLM-TTS Demo with AEC..." << std::endl;

        // Initialize LLM (same as original)
#ifdef USE_CLOUD_LLM
        if (!params_.api_key.empty()) {
            api_comm::setApiKey(params_.api_key);
        }
        if (!params_.api_url.empty()) {
            api_comm::setApiUrl(params_.api_url);
        }
        if (api_comm::loadConfigFromEnv(params_.env_file)) {
            std::cout << "API configuration loaded from " << params_.env_file << std::endl;
        }
        if (!api_comm::getClient().isConfigured()) {
            const char* env_key = std::getenv("API_KEY");
            if (!env_key) env_key = std::getenv("OPENAI_API_KEY");
            if (!env_key) env_key = std::getenv("DEEPSEEK_API_KEY");
            if (env_key) api_comm::setApiKey(env_key);

            const char* env_url = std::getenv("API_URL");
            if (env_url) api_comm::setApiUrl(env_url);
        }
        if (!api_comm::getClient().isConfigured()) {
            std::cerr << "Error: API not configured" << std::endl;
            return false;
        }
        std::cout << "Using cloud LLM: " << api_comm::getClient().getApiProvider() << std::endl;
#else
        if (!ollama::is_running()) {
            std::cerr << "Error: Ollama server is not running" << std::endl;
            return false;
        }
        std::cout << "Ollama server running (version: " << ollama::get_version() << ")" << std::endl;

        std::vector<std::string> models = ollama::list_models();
        bool found = false;
        for (const auto& m : models) {
            if (m == params_.llm_model) { found = true; break; }
        }
        if (!found) {
            std::cout << "Pulling model: " << params_.llm_model << std::endl;
            if (!ollama::pull_model(params_.llm_model)) {
                std::cerr << "Failed to pull model" << std::endl;
                return false;
            }
        }
#endif

        // Download models
        ModelDownloader downloader;
        if (!downloader.ensureModelsExist()) {
            std::cerr << "Failed to download ASR models" << std::endl;
            return false;
        }

        tts::TTSModelDownloader tts_downloader;
        if (!tts_downloader.ensureModelsExist(params_.tts_type)) {
            std::cerr << "Failed to download TTS models" << std::endl;
            return false;
        }

        // Initialize VAD detector
        if (params_.vad_type == "silero") {
            VADDetector::Config vad_config;
            vad_config.model_path = downloader.getModelPath(ModelDownloader::VAD_MODEL_NAME);
            vad_config.sample_rate = 16000;  // VAD works at 16kHz
            vad_config.window_size = 512;
            vad_config.context_size = 64;

            vad_detector_ = std::make_unique<VADDetector>(vad_config);
            if (!vad_detector_->initialize()) {
                std::cerr << "Failed to initialize Silero VAD" << std::endl;
                return false;
            }
            std::cout << "Using Silero VAD" << std::endl;
        }

        // Initialize ASR model
#ifdef USE_CLOUD_ASR
        ASRRealtimeClient::Config asr_config;
        asr_config.sample_rate = 16000;
        asr_model_ = std::make_unique<ASRRealtimeClient>(asr_config);
#else
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
#endif
        if (!asr_model_->initialize()) {
            std::cerr << "Failed to initialize ASR model" << std::endl;
            return false;
        }

        // Initialize TTS model
        tts::TTSConfig tts_config;
        if (params_.tts_type == "en") {
            tts_config.acoustic_model_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_EN_MODEL);
            tts_config.vocoder_path = tts_downloader.getModelPath(tts::TTSModelDownloader::VOCOS_VOCODER);
            tts_config.tokens_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_EN_TOKENS);
            tts_config.data_dir = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_EN_DATA_DIR);
            tts_config.language = "en";
            tts_config.sample_rate = 22050;
        } else if (params_.tts_type == "zh-en") {
            tts_config.acoustic_model_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_EN_MODEL);
            tts_config.vocoder_path = tts_downloader.getModelPath(tts::TTSModelDownloader::VOCOS_VOCODER_16K);
            tts_config.tokens_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_EN_TOKENS);
            tts_config.language = "zh-en";
            tts_config.sample_rate = 16000;
        } else {
            tts_config.acoustic_model_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_MODEL);
            tts_config.vocoder_path = tts_downloader.getModelPath(tts::TTSModelDownloader::VOCOS_VOCODER);
            tts_config.lexicon_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_LEXICON);
            tts_config.tokens_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_TOKENS);
            tts_config.dict_dir = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_DICT_DIR);
            tts_config.language = "zh";
            tts_config.sample_rate = 22050;
        }
        tts_config.noise_scale = 1.0f;
        tts_config.length_scale = params_.tts_speed;
        tts_config.target_rms = params_.target_rms;
        tts_config.use_rms_norm = params_.use_rms_norm;

        tts_model_ = std::make_unique<tts::TTSModel>(tts_config);
        if (!tts_model_->initialize()) {
            std::cerr << "Failed to initialize TTS model" << std::endl;
            return false;
        }
        tts_sample_rate_ = tts_model_->getSampleRate();
        std::cout << "TTS initialized (sample rate: " << tts_sample_rate_ << "Hz)" << std::endl;

        // Initialize AEC Audio Processor
        AECAudioProcessor::Config aec_config;
        aec_config.sample_rate = params_.sample_rate;
        aec_config.channels = params_.channels;
        aec_config.frames_per_buffer = 480;  // 10ms at 48kHz
        aec_config.input_device_index = params_.input_device_index;
        aec_config.output_device_index = params_.output_device_index;
        aec_config.input_channels = params_.input_channels;
        aec_config.selected_channel = params_.selected_channel;
        aec_config.vad_type = params_.vad_type;
        aec_config.vad_trigger_threshold = params_.trigger_threshold;
        aec_config.vad_stop_threshold = params_.stop_threshold;
        aec_config.silence_duration = params_.silence_duration;
        aec_config.max_record_time = params_.max_record_time;
        aec_config.aec_enabled = params_.aec_enabled;
        aec_config.agc_enabled = true;
        aec_config.ns_enabled = true;
        aec_config.barge_in_enabled = params_.barge_in_enabled;
        aec_config.barge_in_threshold = params_.barge_in_threshold;

        aec_processor_ = std::make_unique<AECAudioProcessor>(aec_config);

        // Set VAD detector
        if (vad_detector_) {
            aec_processor_->setVADDetector(vad_detector_.get());
        }

        // Set barge-in callback - this is the key for true interruption!
        aec_processor_->setBargeInCallback([this]() {
            handleBargeIn();
        });

        if (!aec_processor_->initialize()) {
            std::cerr << "Failed to initialize AEC Audio Processor" << std::endl;
            return false;
        }

        std::cout << "AEC Audio Processor initialized" << std::endl;
        std::cout << "  Echo Cancellation: " << (params_.aec_enabled ? "ON" : "OFF") << std::endl;
        std::cout << "  Barge-in: " << (params_.barge_in_enabled ? "ON" : "OFF") << std::endl;

        return true;
    }

    void run() {
        std::cout << "\n=== ASR-LLM-TTS Demo with TRUE Barge-in ===" << std::endl;
        std::cout << "Features:" << std::endl;
        std::cout << "  - Continuous listening (always on)" << std::endl;
        std::cout << "  - Echo cancellation (TTS won't trigger ASR)" << std::endl;
        std::cout << "  - TRUE Barge-in (speak anytime to interrupt)" << std::endl;
        std::cout << "\nPress Ctrl+C to quit\n" << std::endl;

        // Start audio processing - this runs continuously!
        if (!aec_processor_->start()) {
            std::cerr << "Failed to start audio processor" << std::endl;
            return;
        }

        running_ = true;

        // Start the LLM/TTS processing thread
        llm_thread_ = std::thread(&ASRLLMTTSAECDemo::llmProcessingLoop, this);

        // Main loop - continuously collect speech
        while (running_) {
            collectSpeech();
        }

        // Cleanup
        if (llm_thread_.joinable()) {
            llm_thread_.join();
        }

        aec_processor_->stop();
        std::cout << "Demo finished." << std::endl;
    }

    void stop() {
        running_ = false;
        interrupted_ = true;
        processing_cv_.notify_all();
        if (aec_processor_) {
            aec_processor_->clearPlaybackQueue();
            aec_processor_->stop();
        }
    }

private:
    // Continuously collect speech and queue for processing
    void collectSpeech() {
        std::cout << "\n[Listening...] Speak anytime (even during TTS)" << std::endl;

        // Reset state
        aec_processor_->resetBargeInFlag();

        // Collect audio with VAD - but this doesn't block playback!
        // The AEC processor handles playback in the audio callback
        std::vector<float> audio_48k = aec_processor_->recordWithVAD();

        if (audio_48k.empty()) {
            // No speech detected, just continue listening
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return;
        }

        auto audio_duration = static_cast<float>(audio_48k.size()) / params_.sample_rate;
        std::cout << "[Captured] " << audio_duration << "s of speech" << std::endl;

        // Downsample from 48kHz to 16kHz for ASR
        std::vector<float> audio_16k = AECAudioProcessor::resample(audio_48k, 48000, 16000);

        // Queue for processing
        {
            std::lock_guard<std::mutex> lock(processing_mutex_);
            pending_audio_ = std::move(audio_16k);
            has_pending_audio_ = true;
        }
        processing_cv_.notify_one();
    }

    // Separate thread for LLM/TTS processing
    void llmProcessingLoop() {
        while (running_) {
            std::vector<float> audio_to_process;

            // Wait for audio to process
            {
                std::unique_lock<std::mutex> lock(processing_mutex_);
                processing_cv_.wait(lock, [this] {
                    return has_pending_audio_ || !running_;
                });

                if (!running_) break;

                audio_to_process = std::move(pending_audio_);
                has_pending_audio_ = false;
            }

            if (audio_to_process.empty()) continue;

            // Process: ASR -> LLM -> TTS
            processAudioToResponse(audio_to_process);
        }
    }

    void processAudioToResponse(const std::vector<float>& audio_16k) {
        // Reset interrupted flag for this turn
        interrupted_ = false;

        // ASR
        std::cout << "\n[ASR] Processing..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        std::string asr_result = asr_model_->recognize(audio_16k);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();

        if (asr_result.empty()) {
            std::cout << "[ASR] No speech recognized" << std::endl;
            return;
        }

        std::cout << "[ASR] Result: " << asr_result << " (" << duration << "s)" << std::endl;

        // Check for exit commands
        if (asr_result.find("退出") != std::string::npos ||
            asr_result.find("再见") != std::string::npos ||
            asr_result.find("拜拜") != std::string::npos) {
            std::cout << "\n[Goodbye!]" << std::endl;
            running_ = false;
            return;
        }

        // LLM + TTS (streaming)
        std::cout << "\n[LLM] Generating response..." << std::endl;
        processLLMAndTTS(asr_result);
    }

    void processLLMAndTTS(const std::string& prompt) {
        try {
#ifdef USE_CLOUD_LLM
            api_comm::Options options;
            options.temperature = 0.7f;
            options.max_tokens = params_.max_tokens;
            options.stream = true;
#else
            ollama::options options;
            options["temperature"] = 0.7;
            options["max_tokens"] = params_.max_tokens;
#endif

            TextBuffer text_buffer;
            std::string full_response;

            std::cout << "[LLM] Response: " << std::flush;

#ifdef USE_CLOUD_LLM
            auto stream_callback = [&](const api_comm::Response& response) -> bool {
                if (interrupted_) {
                    std::cout << "\n[INTERRUPTED by user]" << std::endl;
                    return false;
                }

                if (response.raw_json.contains("done") && response.raw_json["done"] == true) {
                    text_buffer.addText("。");
                    while (text_buffer.hasSentence()) {
                        std::string sentence = text_buffer.getNextSentence();
                        if (!sentence.empty() && !interrupted_) {
                            generateAndPlayTTS(sentence);
                        }
                    }
                    std::cout << std::endl;
                    return false;
                }

                std::string chunk = response.content;
                std::cout << chunk << std::flush;
                full_response += chunk;
                text_buffer.addText(chunk);

                while (text_buffer.hasSentence() && !interrupted_) {
                    std::string sentence = text_buffer.getNextSentence();
                    if (!sentence.empty()) {
                        generateAndPlayTTS(sentence);
                    }
                }

                return !interrupted_;
            };

            api_comm::generate(params_.llm_model, prompt, stream_callback, options);
#else
            auto stream_callback = [&](const ollama::response& response) -> bool {
                if (interrupted_) {
                    std::cout << "\n[INTERRUPTED by user]" << std::endl;
                    return false;
                }

                std::string chunk = response.as_simple_string();
                std::cout << chunk << std::flush;
                full_response += chunk;
                text_buffer.addText(chunk);

                while (text_buffer.hasSentence() && !interrupted_) {
                    std::string sentence = text_buffer.getNextSentence();
                    if (!sentence.empty()) {
                        generateAndPlayTTS(sentence);
                    }
                }

                if (response.as_json()["done"] == true) {
                    text_buffer.addText("。");
                    while (text_buffer.hasSentence() && !interrupted_) {
                        std::string sentence = text_buffer.getNextSentence();
                        if (!sentence.empty()) {
                            generateAndPlayTTS(sentence);
                        }
                    }
                    std::cout << std::endl;
                }

                return !interrupted_;
            };

            ollama::generate(params_.llm_model, prompt, stream_callback, options);
#endif

            // Note: We DON'T wait for playback to finish here!
            // The audio continues to play while we go back to listening
            // If user speaks, barge-in will interrupt

        } catch (const std::exception& e) {
            std::cerr << "\n[LLM Error] " << e.what() << std::endl;
        }
    }

    void generateAndPlayTTS(const std::string& text) {
        if (interrupted_ || text.empty()) return;

        auto start = std::chrono::high_resolution_clock::now();
        tts::GeneratedAudio audio = tts_model_->generate(text, params_.tts_speaker_id, params_.tts_speed);
        auto end = std::chrono::high_resolution_clock::now();

        if (audio.samples.empty()) {
            std::cerr << "\n[TTS] Failed to generate: " << text << std::endl;
            return;
        }

        auto gen_time = std::chrono::duration<double>(end - start).count();
        std::cout << "\n[TTS] Generated " << audio.duration() << "s in " << gen_time << "s" << std::endl;

        // Enqueue for playback - this returns immediately!
        // Audio plays in background while we continue processing
        if (!interrupted_) {
            aec_processor_->enqueuePlayback(audio.samples, audio.sample_rate);
        }
    }

    void handleBargeIn() {
        std::cout << "\n[BARGE-IN] User speaking! Stopping TTS..." << std::endl;
        interrupted_ = true;
        // The AEC processor already cleared the playback queue
    }

    Params params_;
    std::unique_ptr<AECAudioProcessor> aec_processor_;
    std::unique_ptr<VADDetector> vad_detector_;
    std::unique_ptr<ASREngine> asr_model_;
    std::unique_ptr<tts::TTSModel> tts_model_;
    int tts_sample_rate_ = 22050;

    std::atomic<bool> running_{false};
    std::atomic<bool> interrupted_{false};

    // Processing thread
    std::thread llm_thread_;
    std::mutex processing_mutex_;
    std::condition_variable processing_cv_;
    std::vector<float> pending_audio_;
    bool has_pending_audio_ = false;
};

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input_device <idx>    Input device index (-1 for default)" << std::endl;
    std::cout << "  --output_device <idx>   Output device index (-1 for default)" << std::endl;
    std::cout << "  --input_channels <n>    Number of input channels to open (default: 1)" << std::endl;
    std::cout << "  --selected_channel <n>  Channel to use (0-based, -1 for mix all, default: 0)" << std::endl;
    std::cout << "  --vad_type <type>       VAD type: 'energy' or 'silero' (default: silero)" << std::endl;
    std::cout << "  --trigger_threshold <v> VAD trigger threshold (default: 0.6)" << std::endl;
    std::cout << "  --stop_threshold <v>    VAD stop threshold (default: 0.35)" << std::endl;
    std::cout << "  --silence_duration <s>  Silence to stop recording (default: 1.0)" << std::endl;
    std::cout << "  --max_record_time <s>   Max recording time (default: 10.0)" << std::endl;
    std::cout << "  --model <name>          LLM model name" << std::endl;
    std::cout << "  --max_tokens <n>        Max LLM tokens" << std::endl;
    std::cout << "  --tts_speed <v>         TTS speed (default: 1.0)" << std::endl;
    std::cout << "  --tts_type <type>       TTS type: 'zh', 'en', or 'zh-en'" << std::endl;
    std::cout << "  --no_aec                Disable echo cancellation" << std::endl;
    std::cout << "  --no_barge_in           Disable barge-in" << std::endl;
    std::cout << "  --barge_in_threshold <v> Barge-in VAD threshold (default: 0.5)" << std::endl;
    std::cout << "  --help                  Show this help" << std::endl;
}

int main(int argc, char** argv) {
    // Suppress ONNX warnings
    int stderr_fd = dup(STDERR_FILENO);
    int devnull_fd = open("/dev/null", O_WRONLY);
    dup2(devnull_fd, STDERR_FILENO);

    std::cout << "ASR-LLM-TTS with WebRTC AEC (TRUE Barge-in)" << std::endl;
    std::cout << "============================================" << std::endl;

    ASRLLMTTSAECDemo::Params params;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--input_device" && i + 1 < argc) {
            params.input_device_index = std::atoi(argv[++i]);
        }
        else if (arg == "--output_device" && i + 1 < argc) {
            params.output_device_index = std::atoi(argv[++i]);
        }
        else if (arg == "--input_channels" && i + 1 < argc) {
            params.input_channels = std::atoi(argv[++i]);
        }
        else if (arg == "--selected_channel" && i + 1 < argc) {
            params.selected_channel = std::atoi(argv[++i]);
        }
        else if (arg == "--vad_type" && i + 1 < argc) {
            params.vad_type = argv[++i];
        }
        else if (arg == "--trigger_threshold" && i + 1 < argc) {
            params.trigger_threshold = std::atof(argv[++i]);
        }
        else if (arg == "--stop_threshold" && i + 1 < argc) {
            params.stop_threshold = std::atof(argv[++i]);
        }
        else if (arg == "--silence_duration" && i + 1 < argc) {
            params.silence_duration = std::atof(argv[++i]);
        }
        else if (arg == "--max_record_time" && i + 1 < argc) {
            params.max_record_time = std::atof(argv[++i]);
        }
        else if (arg == "--model" && i + 1 < argc) {
            params.llm_model = argv[++i];
        }
        else if (arg == "--max_tokens" && i + 1 < argc) {
            params.max_tokens = std::atoi(argv[++i]);
        }
        else if (arg == "--tts_speed" && i + 1 < argc) {
            params.tts_speed = std::atof(argv[++i]);
        }
        else if (arg == "--tts_type" && i + 1 < argc) {
            params.tts_type = argv[++i];
        }
        else if (arg == "--no_aec") {
            params.aec_enabled = false;
        }
        else if (arg == "--no_barge_in") {
            params.barge_in_enabled = false;
        }
        else if (arg == "--barge_in_threshold" && i + 1 < argc) {
            params.barge_in_threshold = std::atof(argv[++i]);
        }
    }

    // Print config
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Sample rate: " << params.sample_rate << " Hz (48kHz for AEC)" << std::endl;
    std::cout << "  VAD type: " << params.vad_type << std::endl;
    std::cout << "  LLM model: " << params.llm_model << std::endl;
    std::cout << "  TTS type: " << params.tts_type << std::endl;
    std::cout << "  AEC: " << (params.aec_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "  Barge-in: " << (params.barge_in_enabled ? "ON" : "OFF") << std::endl;
    std::cout << std::endl;

    try {
        ASRLLMTTSAECDemo demo(params);

        if (!demo.initialize()) {
            dup2(stderr_fd, STDERR_FILENO);
            close(stderr_fd);
            close(devnull_fd);
            std::cerr << "Failed to initialize" << std::endl;
            return 1;
        }

        // Restore stderr
        dup2(stderr_fd, STDERR_FILENO);
        close(stderr_fd);
        close(devnull_fd);

        demo.run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
