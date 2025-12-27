/**
 * @file main_asr_llm_tts_mcp.cpp
 * @brief ASR-LLM-TTS with MCP integration
 *
 * Full-featured voice assistant with MCP tool calling support.
 * Based on main_asr_llm_tts.cpp with MCP client integration.
 * Supports both local (Ollama/SenseVoice) and cloud (API/Aliyun) backends.
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
#include <portaudio.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <regex>

#include "audio_recorder.hpp"
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
    // Include ollama.hpp first - it embeds httplib 0.15.3
    #include "ollama.hpp"
#endif
#include "tts/tts_model.hpp"
#include "tts/tts_model_downloader.hpp"
#include "text_buffer.hpp"
#include "ordered_audio_queue.hpp"

// MCP includes - uses httplib from ollama.hpp due to include guard
#include "mcp_sse_client.h"

class ASRLLMTTSMCPDemo {
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

        // API params (for cloud LLM)
        std::string api_key;
        std::string api_url;
        std::string env_file;

        // TTS params
        float tts_speed;
        int tts_speaker_id;
        float target_rms;
        float compression_ratio;
        float compression_threshold;
        bool use_rms_norm;
        std::string tts_type;
        int output_sample_rate;

        // Speaker recognition params
        bool enable_speaker_recognition;
        float speaker_threshold;
        std::string speaker_database;

        // MCP params
        bool enable_mcp;
        std::string mcp_server_url;
        int mcp_timeout;

        Params() :
            sample_rate(16000),
            channels(1),
            device_index(-1),
            silence_duration(1.0),
            max_record_time(5.0),
            trigger_threshold(0.6),
            stop_threshold(0.35),
            vad_type("silero"),
#ifdef USE_CLOUD_LLM
            llm_model("deepseek-chat"),
            max_tokens(500),
            env_file(".env"),
#else
            llm_model("qwen2.5:0.5b"),
            max_tokens(100),
#endif
            tts_speed(1.0f),
            tts_speaker_id(0),
            target_rms(0.15f),
            compression_ratio(2.0f),
            compression_threshold(0.7f),
            use_rms_norm(true),
            tts_type("zh"),
            output_sample_rate(0),
            enable_speaker_recognition(false),
            speaker_threshold(0.6f),
            speaker_database("speakers.db"),
            enable_mcp(true),
            mcp_server_url("http://localhost:8888"),
            mcp_timeout(10) {}
    };

    ASRLLMTTSMCPDemo(const Params& params) : params_(params) {}

    ~ASRLLMTTSMCPDemo() {
        tts_stop_flag_ = true;
        tts_queue_cv_.notify_all();
        if (tts_worker_thread_.joinable()) {
            tts_worker_thread_.join();
        }
    }

    bool initialize() {
        std::cout << "Initializing ASR-LLM-TTS-MCP Demo..." << std::endl;

#ifdef USE_CLOUD_LLM
        // Configure cloud LLM API
        std::cout << "[DEBUG] Step 1: Configuring Cloud LLM API..." << std::flush;
        if (!params_.api_key.empty()) {
            api_comm::setApiKey(params_.api_key);
            std::cout << " API key set from command line." << std::endl;
        }

        if (!params_.api_url.empty()) {
            api_comm::setApiUrl(params_.api_url);
            std::cout << " API URL set: " << params_.api_url << std::endl;
        }

        // Try to load from .env file
        if (api_comm::loadConfigFromEnv(params_.env_file)) {
            std::cout << " API configuration loaded from " << params_.env_file << std::endl;
        }

        // Try environment variables as fallback
        if (!api_comm::getClient().isConfigured()) {
            const char* env_key = std::getenv("API_KEY");
            if (!env_key) env_key = std::getenv("OPENAI_API_KEY");
            if (!env_key) env_key = std::getenv("DEEPSEEK_API_KEY");

            if (env_key) {
                api_comm::setApiKey(env_key);
                std::cout << " API key loaded from environment variable" << std::endl;
            }

            const char* env_url = std::getenv("API_URL");
            if (env_url) {
                api_comm::setApiUrl(env_url);
                std::cout << " API URL loaded from environment variable" << std::endl;
            }
        }

        if (!api_comm::getClient().isConfigured()) {
            std::cerr << "Error: API not configured. Please set API key and URL via:" << std::endl;
            std::cerr << "  1. Command line: --api_key YOUR_KEY --api_url YOUR_URL" << std::endl;
            std::cerr << "  2. Environment variables: export API_KEY=YOUR_KEY API_URL=YOUR_URL" << std::endl;
            std::cerr << "  3. .env file: API_KEY=YOUR_KEY and API_URL=YOUR_URL" << std::endl;
            return false;
        }

        std::cout << "API configured successfully (" << api_comm::getClient().getApiProvider() << ")" << std::endl;
        std::cout << "Using model: " << params_.llm_model << std::endl;
#else
        // Check if Ollama server is running
        std::cout << "[DEBUG] Step 1: Checking Ollama server..." << std::flush;
        try {
            bool is_running = ollama::is_running();
            std::cout << " checked." << std::endl;
            if (!is_running) {
                std::cerr << "Error: Ollama server is not running. Please start ollama service first." << std::endl;
                return false;
            }
            std::cout << "Ollama server is running (version: " << ollama::get_version() << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error checking Ollama: " << e.what() << std::endl;
            return false;
        }

        // Check if the specified model is available
        std::cout << "[DEBUG] Step 2: Checking LLM model..." << std::flush;
        try {
            std::vector<std::string> models = ollama::list_models();
            std::cout << " listed " << models.size() << " models." << std::endl;
            bool model_found = false;
            for (const auto& model : models) {
                if (model == params_.llm_model) {
                    model_found = true;
                    break;
                }
            }

            if (!model_found) {
                std::cout << "Model '" << params_.llm_model << "' not found locally. Attempting to pull..." << std::endl;
                if (!ollama::pull_model(params_.llm_model)) {
                    std::cerr << "Failed to pull model: " << params_.llm_model << std::endl;
                    return false;
                }
                std::cout << "Model pulled successfully!" << std::endl;
            } else {
                std::cout << "Using existing model: " << params_.llm_model << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error with LLM model: " << e.what() << std::endl;
            return false;
        }
#endif

        // Download ASR models if needed
        std::cout << "[DEBUG] Step 3: Downloading ASR models..." << std::flush;
        ModelDownloader downloader;
        if (!downloader.ensureModelsExist()) {
            std::cerr << "Failed to ensure ASR models exist" << std::endl;
            return false;
        }
        std::cout << " done." << std::endl;

        // Download TTS models if needed
        std::cout << "[DEBUG] Step 4: Downloading TTS models for type: " << params_.tts_type << "..." << std::flush;
        tts::TTSModelDownloader tts_downloader;
        if (!tts_downloader.ensureModelsExist(params_.tts_type)) {
            std::cerr << "Failed to ensure TTS models exist" << std::endl;
            return false;
        }
        std::cout << " done." << std::endl;

        // Initialize VAD detector if using Silero VAD
        std::cout << "[DEBUG] Step 5: Initializing VAD..." << std::flush;
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
            std::cout << " done (Silero)." << std::endl;
        } else {
            std::cout << " done (energy)." << std::endl;
        }

        // Initialize ASR model
        std::cout << "[DEBUG] Step 6: Initializing ASR model..." << std::flush;
#ifdef USE_CLOUD_ASR
        // 云端 ASR 配置
        ASRRealtimeClient::Config asr_config;
        // 从环境变量自动加载配置（TOKEN 或 AccessKey）
        asr_config.sample_rate = 16000;

        asr_model_ = std::make_unique<ASRRealtimeClient>(asr_config);
#else
        // 本地 ASR 配置
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
        std::cout << " done." << std::endl;

        // Initialize TTS model
        std::cout << "[DEBUG] Step 7: Initializing TTS model..." << std::flush;
        tts::TTSConfig tts_config;

        if (params_.tts_type == "en") {
            tts_config.acoustic_model_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_EN_MODEL);
            tts_config.vocoder_path = tts_downloader.getModelPath(tts::TTSModelDownloader::VOCOS_VOCODER);
            tts_config.tokens_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_EN_TOKENS);
            tts_config.data_dir = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_EN_DATA_DIR);
            tts_config.language = "en";
            tts_config.lexicon_path = "";
            tts_config.dict_dir = "";
            tts_config.jieba_dict_dir = "";
        } else if (params_.tts_type == "zh-en") {
            tts_config.acoustic_model_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_EN_MODEL);
            tts_config.vocoder_path = tts_downloader.getModelPath(tts::TTSModelDownloader::VOCOS_VOCODER_16K);
            tts_config.tokens_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_EN_TOKENS);
            tts_config.language = "zh-en";
            tts_config.sample_rate = 16000;
            tts_config.lexicon_path = "";
            tts_config.dict_dir = "";
            tts_config.jieba_dict_dir = "";
            tts_config.data_dir = "";
        } else {
            tts_config.acoustic_model_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_MODEL);
            tts_config.vocoder_path = tts_downloader.getModelPath(tts::TTSModelDownloader::VOCOS_VOCODER);
            tts_config.lexicon_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_LEXICON);
            tts_config.tokens_path = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_TOKENS);
            tts_config.dict_dir = tts_downloader.getModelPath(tts::TTSModelDownloader::MATCHA_ZH_DICT_DIR);
            tts_config.jieba_dict_dir = "";
            tts_config.language = "zh";
        }

        if (params_.tts_type != "zh-en") {
            tts_config.sample_rate = 22050;
        }
        tts_config.noise_scale = 1.0f;
        tts_config.length_scale = params_.tts_speed;
        tts_config.target_rms = params_.target_rms;
        tts_config.compression_ratio = params_.compression_ratio;
        tts_config.compression_threshold = params_.compression_threshold;
        tts_config.use_rms_norm = params_.use_rms_norm;
        tts_config.output_sample_rate = params_.output_sample_rate;

        tts_model_ = std::make_unique<tts::TTSModel>(tts_config);
        if (!tts_model_->initialize()) {
            std::cerr << "Failed to initialize TTS model" << std::endl;
            return false;
        }
        std::cout << " done (type: " << params_.tts_type
                  << ", sample rate: " << tts_model_->getSampleRate() << "Hz)." << std::endl;

        // Initialize audio recorder
        std::cout << "[DEBUG] Step 8: Initializing audio recorder..." << std::flush;
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
        recorder_config.enable_speaker_recognition = params_.enable_speaker_recognition;
        recorder_config.speaker_threshold = params_.speaker_threshold;
        recorder_config.speaker_database = params_.speaker_database;

        audio_recorder_ = std::make_unique<AudioRecorder>(recorder_config);
        if (!audio_recorder_->initialize()) {
            std::cerr << "Failed to initialize audio recorder" << std::endl;
            return false;
        }
        std::cout << " done." << std::endl;

        if (params_.vad_type == "silero" && vad_detector_) {
            audio_recorder_->setVADDetector(vad_detector_.get());
        }

        // Initialize PortAudio for playback
        std::cout << "[DEBUG] Step 9: Initializing PortAudio..." << std::flush;
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        std::cout << " done." << std::endl;

        // Initialize ordered audio queue
        std::cout << "[DEBUG] Step 10: Initializing audio queue..." << std::flush;
        audio_queue_ = std::make_unique<OrderedAudioQueue>();
        audio_queue_->start();
        std::cout << " done." << std::endl;

        // Start TTS worker thread for RISC-V
        #if defined(__riscv) || defined(__riscv__)
        std::cout << "[DEBUG] Step 11: Starting TTS worker thread for RISC-V..." << std::flush;
        tts_worker_thread_ = std::thread(&ASRLLMTTSMCPDemo::ttsWorker, this);
        std::cout << " done." << std::endl;
        #endif

        // Initialize MCP client if enabled
        if (params_.enable_mcp) {
            std::cout << "[DEBUG] Step 12: Connecting to MCP server at " << params_.mcp_server_url << "..." << std::flush;
            mcp_client_ = std::make_unique<mcp::sse_client>(params_.mcp_server_url);
            mcp_client_->set_timeout(params_.mcp_timeout);

            try {
                std::cout << std::endl;
                std::cout << "[DEBUG] Step 12a: Creating MCP client..." << std::flush;
                if (!mcp_client_->initialize("VoiceClient", mcp::MCP_VERSION)) {
                    std::cerr << "Failed to connect to MCP server, MCP features disabled" << std::endl;
                    mcp_client_.reset();
                } else {
                    std::cout << " OK" << std::endl;
                    std::cout << "[DEBUG] Step 12b: Pinging MCP server..." << std::flush;
                    if (!mcp_client_->ping()) {
                        std::cerr << "MCP server not responding, MCP features disabled" << std::endl;
                        mcp_client_.reset();
                    } else {
                        std::cout << " OK" << std::endl;
                        auto tools = mcp_client_->get_tools();
                        std::cout << "MCP connected! Available tools:" << std::endl;
                        for (const auto& tool : tools) {
                            std::cout << "  - " << tool.name << ": " << tool.description << std::endl;
                            mcp_tools_.push_back(tool.name);
                        }
                        // Build Ollama-compatible tools JSON for function calling
                        buildOllamaTools();
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "MCP connection error: " << e.what() << ", MCP features disabled" << std::endl;
                mcp_client_.reset();
            }
        }

        std::cout << "ASR-LLM-TTS-MCP Demo initialized successfully!" << std::endl;
        return true;
    }

    void run() {
        std::cout << "\n=== ASR-LLM-TTS-MCP Demo Started ===" << std::endl;
        std::cout << "This demo will:" << std::endl;
        std::cout << "1. Record your speech using ASR" << std::endl;
        std::cout << "2. Convert speech to text" << std::endl;
        std::cout << "3. Send to LLM (" << params_.llm_model << ")" << std::endl;
        if (mcp_client_) {
            std::cout << "   - LLM can use MCP tools: " << mcp_tools_.size() << " available" << std::endl;
        }
        std::cout << "4. Convert response to speech using TTS" << std::endl;
        std::cout << "5. Play the generated speech" << std::endl;

        if (mcp_client_) {
            std::cout << "\nMCP Tools available for LLM:" << std::endl;
            for (const auto& tool : mcp_tools_) {
                std::cout << "  - " << tool << std::endl;
            }
        }

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

        if (audio_queue_) {
            audio_queue_->stop();
        }
        Pa_Terminate();

        std::cout << "Demo finished." << std::endl;
    }

private:
    // Store MCP tools in Ollama/OpenAI function calling format
    mcp::json ollama_tools_;

    // Build tools JSON for Ollama from MCP tools
    void buildOllamaTools() {
        if (!mcp_client_) return;

        ollama_tools_ = mcp::json::array();
        auto mcp_tools = mcp_client_->get_tools();

        for (const auto& tool : mcp_tools) {
            mcp::json ollama_tool = {
                {"type", "function"},
                {"function", {
                    {"name", tool.name},
                    {"description", tool.description},
                    {"parameters", tool.parameters_schema}
                }}
            };
            ollama_tools_.push_back(ollama_tool);
        }

        std::cout << "Built " << ollama_tools_.size() << " tools for LLM function calling" << std::endl;
    }

    // Execute a tool call from LLM response
    std::string executeMCPToolCall(const std::string& tool_name, const mcp::json& arguments) {
        if (!mcp_client_) return "MCP client not available";

        try {
            std::cout << "[MCP] Calling tool: " << tool_name << std::endl;
            std::cout << "[MCP] Arguments: " << arguments.dump() << std::endl;

            auto result = mcp_client_->call_tool(tool_name, arguments);

            if (result.contains("content") && result["content"].is_array() && !result["content"].empty()) {
                std::string text_result = result["content"][0]["text"].get<std::string>();
                std::cout << "[MCP] Result: " << text_result << std::endl;
                return text_result;
            }

            return "操作成功";
        } catch (const mcp::mcp_exception& e) {
            return std::string("MCP错误: ") + e.what();
        } catch (const std::exception& e) {
            return std::string("错误: ") + e.what();
        }
    }

    // Convert math symbols to Chinese for TTS
    std::string convertMathToChineseTTS(const std::string& text) {
        std::string result = text;

        // Remove trailing zeros after decimal point (100.00 -> 100)
        std::regex decimal_pattern(R"((\d+)\.0+\b)");
        result = std::regex_replace(result, decimal_pattern, "$1");

        // Convert operators to Chinese
        // Note: order matters - do longer patterns first
        std::vector<std::pair<std::string, std::string>> replacements = {
            {" + ", " 加 "},
            {" - ", " 减 "},
            {" * ", " 乘以 "},
            {" × ", " 乘以 "},
            {" / ", " 除以 "},
            {" ÷ ", " 除以 "},
            {" = ", " 等于 "},
            {"+", "加"},
            {"-", "减"},
            {"*", "乘以"},
            {"×", "乘以"},
            {"/", "除以"},
            {"÷", "除以"},
            {"=", "等于"},
        };

        for (const auto& [from, to] : replacements) {
            size_t pos = 0;
            while ((pos = result.find(from, pos)) != std::string::npos) {
                result.replace(pos, from.length(), to);
                pos += to.length();
            }
        }

        return result;
    }

    void processVoiceInteraction() {
        // 1. Record audio
        std::cout << "\n=== Recording Phase ===" << std::endl;
        std::cout << "Speak now! (max " << params_.max_record_time
                  << "s, or silence for " << params_.silence_duration << "s to stop)" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> audio = audio_recorder_->recordAudio();
        auto end_time = std::chrono::high_resolution_clock::now();

        if (audio.empty()) {
            std::cout << "No audio recorded" << std::endl;
            return;
        }

        auto recording_duration = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Recording completed (" << recording_duration << "s)" << std::endl;

        // 2. Convert stereo to mono if necessary
        std::vector<float> mono_audio;
        if (params_.channels == 2) {
            mono_audio.reserve(audio.size() / 2);
            for (size_t i = 0; i < audio.size(); i += 2) {
                mono_audio.push_back((audio[i] + audio[i + 1]) / 2.0f);
            }
        } else {
            mono_audio = audio;
        }

        // 3. Resample if necessary
        std::vector<float> resampled_audio;
        if (params_.sample_rate != 16000) {
            resampled_audio = resampleAudio(mono_audio, params_.sample_rate, 16000);
        } else {
            resampled_audio = mono_audio;
        }

        // 4. Recognize speech
        std::cout << "\n=== ASR Phase ===" << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        std::string asr_result = asr_model_->recognize(resampled_audio);
        end_time = std::chrono::high_resolution_clock::now();

        auto asr_duration = std::chrono::duration<double>(end_time - start_time).count();

        if (asr_result.empty()) {
            std::cout << "No speech recognized" << std::endl;
            return;
        }

        std::cout << "ASR Result: " << asr_result << std::endl;
        std::cout << "ASR time: " << asr_duration << "s" << std::endl;

        // Check speaker recognition
        if (params_.enable_speaker_recognition) {
            if (!audio_recorder_->isSpeakerRegistered()) {
                std::cout << "[Speaker] Not recognized. Skipping." << std::endl;
                return;
            }
            std::cout << "[Speaker] Verified: " << audio_recorder_->getLastIdentifiedSpeaker() << std::endl;
        }

        std::string response;

        // Send to LLM (with MCP tools if available)
        std::cout << "\n=== LLM Phase ===" << std::endl;
#ifdef USE_CLOUD_LLM
        std::cout << "Sending to " << api_comm::getClient().getApiProvider()
                  << " (" << params_.llm_model << ")..." << std::endl;
        // TODO: Implement cloud LLM function calling
        // For now, fall back to simple generation
        audio_queue_->resetOrder();
        // ... existing cloud LLM code ...
#else
        std::cout << "Sending to LLM (" << params_.llm_model << ") with "
                  << (mcp_client_ ? std::to_string(ollama_tools_.size()) + " tools" : "no tools")
                  << "..." << std::endl;

        try {
            start_time = std::chrono::high_resolution_clock::now();

            // Build messages for chat
            ollama::messages messages;
            messages.push_back(ollama::message("user", asr_result));

            // Create chat request with tools
            ollama::request request(params_.llm_model, messages);
            request["stream"] = false;  // Non-streaming for tool calling

            if (mcp_client_ && !ollama_tools_.empty()) {
                request["tools"] = ollama_tools_;
            }

            // First LLM call
            ollama::response llm_response = ollama::chat(request);
            auto response_json = llm_response.as_json();

            std::cout << "LLM Response received" << std::endl;

            // Check if LLM wants to call a tool
            if (response_json.contains("message") &&
                response_json["message"].contains("tool_calls") &&
                response_json["message"]["tool_calls"].is_array() &&
                !response_json["message"]["tool_calls"].empty()) {

                std::cout << "\n=== MCP Tool Calling ===" << std::endl;

                // Add assistant's response to messages
                ollama::message assistant_msg("assistant", "");
                assistant_msg["tool_calls"] = response_json["message"]["tool_calls"];
                messages.push_back(assistant_msg);

                // Process each tool call
                for (const auto& tool_call : response_json["message"]["tool_calls"]) {
                    std::string tool_name = tool_call["function"]["name"].get<std::string>();
                    mcp::json arguments = tool_call["function"]["arguments"];

                    // If arguments is a string, parse it
                    if (arguments.is_string()) {
                        arguments = mcp::json::parse(arguments.get<std::string>());
                    }

                    // Execute MCP tool
                    std::string tool_result = executeMCPToolCall(tool_name, arguments);

                    // Add tool result to messages
                    ollama::message tool_msg("tool", tool_result);
                    messages.push_back(tool_msg);
                }

                // Second LLM call with tool results
                std::cout << "Sending tool results back to LLM..." << std::endl;
                ollama::request follow_up_request(params_.llm_model, messages);
                follow_up_request["stream"] = false;

                llm_response = ollama::chat(follow_up_request);
                response_json = llm_response.as_json();
            }

            // Extract final response
            if (response_json.contains("message") && response_json["message"].contains("content")) {
                response = response_json["message"]["content"].get<std::string>();
            } else {
                response = llm_response.as_simple_string();
            }

            end_time = std::chrono::high_resolution_clock::now();
            auto llm_duration = std::chrono::duration<double>(end_time - start_time).count();

            std::cout << "LLM Response: " << response << std::endl;
            std::cout << "LLM time: " << llm_duration << "s" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "LLM Error: " << e.what() << std::endl;
            return;
        }
#endif

        // Generate TTS for response
        if (!response.empty()) {
            std::cout << "\n=== TTS Phase ===" << std::endl;
            // Convert math symbols to Chinese for TTS
            std::string tts_text = convertMathToChineseTTS(response);
            std::cout << "TTS text: " << tts_text << std::endl;
            generateAndPlayTTS(tts_text);
        }

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Your input: " << asr_result << std::endl;
        std::cout << "Response: " << response << std::endl;
        std::cout << std::flush;
    }

    void generateAndPlayTTS(const std::string& text) {
        auto start_time = std::chrono::high_resolution_clock::now();
        tts::GeneratedAudio gen_audio = tts_model_->generate(text, params_.tts_speaker_id, params_.tts_speed);
        auto end_time = std::chrono::high_resolution_clock::now();

        if (gen_audio.samples.empty()) {
            std::cout << "TTS generation failed" << std::endl;
            return;
        }

        auto tts_duration = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "TTS generated (" << gen_audio.duration() << "s audio in " << tts_duration << "s)" << std::endl;

        std::cout << "[DEBUG] Starting audio playback..." << std::endl;
        playAudio(gen_audio.samples, gen_audio.sample_rate);
        std::cout << "[DEBUG] Audio playback finished." << std::endl;
    }

    std::vector<float> resampleAudio(const std::vector<float>& input, int from_rate, int to_rate) {
        if (from_rate == to_rate) return input;

        if (from_rate == 48000 && to_rate == 16000) {
            std::vector<float> output;
            output.reserve(input.size() / 3);
            for (size_t i = 0; i < input.size(); i += 3) {
                output.push_back(input[i]);
            }
            return output;
        }

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

    void playAudio(const std::vector<float>& samples, int sample_rate) {
        std::cout << "[DEBUG] playAudio: samples=" << samples.size() << ", rate=" << sample_rate << std::endl;

        PaStream* stream;
        PaError err = Pa_OpenDefaultStream(&stream, 0, 1, paFloat32, sample_rate, 256, nullptr, nullptr);

        if (err != paNoError) {
            std::cerr << "Failed to open audio stream: " << Pa_GetErrorText(err) << std::endl;
            return;
        }

        err = Pa_StartStream(stream);
        if (err != paNoError) {
            Pa_CloseStream(stream);
            return;
        }

        const size_t chunk_size = 1024;
        for (size_t i = 0; i < samples.size(); i += chunk_size) {
            size_t frames_to_write = std::min(chunk_size, samples.size() - i);
            err = Pa_WriteStream(stream, &samples[i], frames_to_write);
            if (err != paNoError && err != paOutputUnderflowed) break;
        }

        // Pa_WriteStream is blocking, so audio is already playing/played
        // Just need a small delay for the last buffer to finish
        Pa_Sleep(100);

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        std::cout << "[DEBUG] playAudio: done" << std::endl;
    }

    void generateAndEnqueueOrderedTTS(const std::string& sentence, size_t order) {
        auto start_time = std::chrono::high_resolution_clock::now();
        tts::GeneratedAudio generated_audio = tts_model_->generate(sentence, params_.tts_speaker_id, params_.tts_speed);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto tts_duration = std::chrono::duration<double>(end_time - start_time).count();

        if (generated_audio.samples.empty()) {
            std::cout << "\n[TTS] Failed #" << order << ": " << sentence << std::endl;
            return;
        }

        std::cout << "\n[TTS] Generated #" << order << " (" << generated_audio.duration() << "s in " << tts_duration << "s)" << std::endl;

        OrderedAudioData audio_data(std::move(generated_audio.samples), generated_audio.sample_rate, sentence, order, params_.tts_type);
        audio_queue_->enqueue(audio_data);
    }

    struct TTSTask {
        std::string sentence;
        size_t order;
    };

    void ttsWorker() {
        while (!tts_stop_flag_) {
            TTSTask task;
            bool has_task = false;

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

    Params params_;
    std::unique_ptr<AudioRecorder> audio_recorder_;
    std::unique_ptr<VADDetector> vad_detector_;
    std::unique_ptr<ASREngine> asr_model_;
    std::unique_ptr<tts::TTSModel> tts_model_;
    std::unique_ptr<OrderedAudioQueue> audio_queue_;
    std::unique_ptr<mcp::sse_client> mcp_client_;
    std::vector<std::string> mcp_tools_;

    std::queue<TTSTask> tts_queue_;
    std::mutex tts_queue_mutex_;
    std::condition_variable tts_queue_cv_;
    std::thread tts_worker_thread_;
    std::atomic<bool> tts_stop_flag_{false};
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "\nAudio Options:" << std::endl;
    std::cout << "  --sample_rate <value>       Audio sample rate (default: 16000)" << std::endl;
    std::cout << "  --channels <value>          Number of audio channels (default: 1)" << std::endl;
    std::cout << "  --device_index <value>      Audio device index (default: -1)" << std::endl;
    std::cout << "  --silence_duration <value>  Silence to stop recording (default: 1.0s)" << std::endl;
    std::cout << "  --max_record_time <value>   Max recording time (default: 5.0s)" << std::endl;
    std::cout << "  --trigger_threshold <value> VAD trigger threshold (default: 0.6)" << std::endl;
    std::cout << "  --stop_threshold <value>    VAD stop threshold (default: 0.35)" << std::endl;
    std::cout << "  --vad_type <type>           VAD type: energy or silero (default: silero)" << std::endl;
    std::cout << "\nLLM Options:" << std::endl;
#ifdef USE_CLOUD_LLM
    std::cout << "  --model <model_name>        LLM model name (default: deepseek-chat)" << std::endl;
    std::cout << "  --max_tokens <value>        Max tokens for LLM response (default: 500)" << std::endl;
    std::cout << "  --api_key <key>             API key for cloud LLM" << std::endl;
    std::cout << "  --api_url <url>             API URL for cloud LLM" << std::endl;
    std::cout << "  --env_file <path>           Path to .env file (default: .env)" << std::endl;
#else
    std::cout << "  --model <model_name>        LLM model name (default: qwen2.5:0.5b)" << std::endl;
    std::cout << "  --max_tokens <value>        Max tokens for LLM response (default: 100)" << std::endl;
#endif
    std::cout << "\nTTS Options:" << std::endl;
    std::cout << "  --tts_speed <value>         TTS speech speed (default: 1.0)" << std::endl;
    std::cout << "  --tts_speaker <value>       TTS speaker ID (default: 0)" << std::endl;
    std::cout << "  --tts_type <value>          TTS type: zh, en, zh-en (default: zh)" << std::endl;
    std::cout << "  --output_sample_rate <value> Output sample rate (default: 0 = no resampling)" << std::endl;
    std::cout << "  --target_rms <value>        Target RMS for normalization (default: 0.15)" << std::endl;
    std::cout << "  --compression_ratio <value> Dynamic range compression (default: 2.0)" << std::endl;
    std::cout << "  --use_peak_norm             Use peak normalization instead of RMS" << std::endl;
    std::cout << "\nSpeaker Recognition:" << std::endl;
    std::cout << "  --enable_speaker            Enable speaker recognition" << std::endl;
    std::cout << "  --speaker_threshold <value> Speaker threshold (default: 0.6)" << std::endl;
    std::cout << "  --speaker_database <file>   Speaker database (default: speakers.db)" << std::endl;
    std::cout << "\nMCP Options:" << std::endl;
    std::cout << "  --mcp_url <url>             MCP server URL (default: http://localhost:8888)" << std::endl;
    std::cout << "  --mcp_timeout <seconds>     MCP request timeout (default: 10)" << std::endl;
    std::cout << "  --no_mcp                    Disable MCP integration" << std::endl;
    std::cout << "\n  --help                      Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "[MAIN] Starting program..." << std::flush;
    std::cout << " OK" << std::endl;

    int stderr_fd = dup(STDERR_FILENO);
    int devnull_fd = open("/dev/null", O_WRONLY);
    dup2(devnull_fd, STDERR_FILENO);

    std::cout << "ASR-LLM-TTS-MCP Demo" << std::endl;
    std::cout << "====================" << std::endl;

    ASRLLMTTSMCPDemo::Params params;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--sample_rate" && i + 1 < argc) params.sample_rate = std::atoi(argv[++i]);
        else if (arg == "--channels" && i + 1 < argc) params.channels = std::atoi(argv[++i]);
        else if (arg == "--device_index" && i + 1 < argc) params.device_index = std::atoi(argv[++i]);
        else if (arg == "--silence_duration" && i + 1 < argc) params.silence_duration = std::atof(argv[++i]);
        else if (arg == "--max_record_time" && i + 1 < argc) params.max_record_time = std::atof(argv[++i]);
        else if (arg == "--trigger_threshold" && i + 1 < argc) params.trigger_threshold = std::atof(argv[++i]);
        else if (arg == "--stop_threshold" && i + 1 < argc) params.stop_threshold = std::atof(argv[++i]);
        else if (arg == "--vad_type" && i + 1 < argc) params.vad_type = argv[++i];
        else if (arg == "--model" && i + 1 < argc) params.llm_model = argv[++i];
        else if (arg == "--max_tokens" && i + 1 < argc) params.max_tokens = std::atoi(argv[++i]);
#ifdef USE_CLOUD_LLM
        else if (arg == "--api_key" && i + 1 < argc) params.api_key = argv[++i];
        else if (arg == "--api_url" && i + 1 < argc) params.api_url = argv[++i];
        else if (arg == "--env_file" && i + 1 < argc) params.env_file = argv[++i];
#endif
        else if (arg == "--tts_speed" && i + 1 < argc) params.tts_speed = std::atof(argv[++i]);
        else if (arg == "--tts_speaker" && i + 1 < argc) params.tts_speaker_id = std::atoi(argv[++i]);
        else if (arg == "--tts_type" && i + 1 < argc) params.tts_type = argv[++i];
        else if (arg == "--output_sample_rate" && i + 1 < argc) params.output_sample_rate = std::atoi(argv[++i]);
        else if (arg == "--target_rms" && i + 1 < argc) params.target_rms = std::atof(argv[++i]);
        else if (arg == "--compression_ratio" && i + 1 < argc) params.compression_ratio = std::atof(argv[++i]);
        else if (arg == "--use_peak_norm") params.use_rms_norm = false;
        else if (arg == "--enable_speaker") params.enable_speaker_recognition = true;
        else if (arg == "--speaker_threshold" && i + 1 < argc) params.speaker_threshold = std::atof(argv[++i]);
        else if (arg == "--speaker_database" && i + 1 < argc) params.speaker_database = argv[++i];
        else if (arg == "--mcp_url" && i + 1 < argc) params.mcp_server_url = argv[++i];
        else if (arg == "--mcp_timeout" && i + 1 < argc) params.mcp_timeout = std::atoi(argv[++i]);
        else if (arg == "--no_mcp") params.enable_mcp = false;
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Sample rate: " << params.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << params.channels << std::endl;
    std::cout << "  Device index: " << params.device_index << std::endl;
    std::cout << "  VAD type: " << params.vad_type << std::endl;
#ifdef USE_CLOUD_LLM
    std::cout << "  LLM: Cloud API (" << params.llm_model << ")" << std::endl;
#else
    std::cout << "  LLM: Ollama (" << params.llm_model << ")" << std::endl;
#endif
#ifdef USE_CLOUD_ASR
    std::cout << "  ASR: Cloud (Aliyun)" << std::endl;
#else
    std::cout << "  ASR: Local (SenseVoice)" << std::endl;
#endif
    std::cout << "  TTS type: " << params.tts_type << std::endl;
    std::cout << "  TTS speed: " << params.tts_speed << std::endl;
    if (params.output_sample_rate > 0) {
        std::cout << "  Output sample rate: " << params.output_sample_rate << " Hz" << std::endl;
    }
    if (params.enable_mcp) {
        std::cout << "  MCP server: " << params.mcp_server_url << std::endl;
    } else {
        std::cout << "  MCP: disabled" << std::endl;
    }
    std::cout << std::endl;

    try {
        std::cout << "[MAIN] Creating demo object..." << std::flush;
        ASRLLMTTSMCPDemo demo(params);
        std::cout << " OK" << std::endl;

        std::cout << "[MAIN] Calling initialize()..." << std::flush;
        std::cout << std::endl;
        if (!demo.initialize()) {
            dup2(stderr_fd, STDERR_FILENO);
            close(stderr_fd);
            close(devnull_fd);
            std::cerr << "Failed to initialize demo" << std::endl;
            return 1;
        }

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
