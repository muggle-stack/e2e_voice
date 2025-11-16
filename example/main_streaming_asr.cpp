#include "streaming_audio_recorder.hpp"
#include "asr_thread_pool.hpp"
#include "asr_model.hpp"
#include "model_downloader.hpp"
#include "vad_detector.hpp"
#include <iostream>
#include <signal.h>
#include <atomic>
#include <iomanip>
#include <map>

std::atomic<bool> g_running(true);

void signalHandler(int signum) {
    std::cout << "\r" << std::string(60, ' ') << "\r";  // Clear audio level display
    std::cout << "\nInterrupt signal received. Stopping..." << std::endl;
    g_running = false;
    
    // Restore default signal handler to allow force quit
    signal(signum, SIG_DFL);
}

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --device_index <index>    Audio device index (default: system default)" << std::endl;
    std::cout << "  --sample_rate <rate>      Sample rate (default: 16000)" << std::endl;
    std::cout << "  --max_duration <seconds>  Maximum recording duration (default: 60)" << std::endl;
    std::cout << "  --silence_threshold <sec> Silence duration to segment speech (default: 0.5)" << std::endl;
    std::cout << "  --pre_speech_buffer <sec> Pre-speech buffer duration (default: 0.25)" << std::endl;
    std::cout << "  --num_threads <n>         Number of ASR threads (default: 2)" << std::endl;
    std::cout << "  --vad_threshold <val>     Energy threshold for VAD (default: 0.005)" << std::endl;
    std::cout << "  --vad_type <type>         VAD type: energy or silero (default: silero)" << std::endl;
    std::cout << "  --vad_trigger <val>       Silero trigger threshold (default: 0.5)" << std::endl;
    std::cout << "  --vad_stop <val>          Silero stop threshold (default: 0.35)" << std::endl;
    std::cout << "  --help                    Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int device_index = -1;
    int sample_rate = 16000;
    double max_duration = 60.0;
    double silence_threshold = 0.5;
    double pre_speech_buffer = 0.25;
    size_t num_threads = 2;
    float vad_threshold = 0.005f;  // Lower threshold for better sensitivity
    std::string vad_type = "silero";  // "energy" or "silero"
    float vad_trigger_threshold = 0.5f;
    float vad_stop_threshold = 0.35f;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--device_index" && i + 1 < argc) {
            device_index = std::stoi(argv[++i]);
        } else if (arg == "--sample_rate" && i + 1 < argc) {
            sample_rate = std::stoi(argv[++i]);
        } else if (arg == "--max_duration" && i + 1 < argc) {
            max_duration = std::stod(argv[++i]);
        } else if (arg == "--silence_threshold" && i + 1 < argc) {
            silence_threshold = std::stod(argv[++i]);
        } else if (arg == "--pre_speech_buffer" && i + 1 < argc) {
            pre_speech_buffer = std::stod(argv[++i]);
        } else if (arg == "--num_threads" && i + 1 < argc) {
            num_threads = std::stoul(argv[++i]);
        } else if (arg == "--vad_threshold" && i + 1 < argc) {
            vad_threshold = std::stof(argv[++i]);
        } else if (arg == "--vad_type" && i + 1 < argc) {
            vad_type = argv[++i];
        } else if (arg == "--vad_trigger" && i + 1 < argc) {
            vad_trigger_threshold = std::stof(argv[++i]);
        } else if (arg == "--vad_stop" && i + 1 < argc) {
            vad_stop_threshold = std::stof(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    // Set up signal handlers
    signal(SIGINT, signalHandler);   // Ctrl+C
    signal(SIGTERM, signalHandler);  // Termination
    signal(SIGTSTP, SIG_IGN);        // Ignore Ctrl+Z (suspend)
    
    try {
        // Initialize PortAudio
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(err) << std::endl;
            return 1;
        }
        
        // Print device info
        if (device_index == -1) {
            device_index = Pa_GetDefaultInputDevice();
        }
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(device_index);
        std::cout << "Using audio device: " << deviceInfo->name << std::endl;
        std::cout << "Sample rate: " << sample_rate << " Hz" << std::endl;
        
        // Download models if needed
        ModelDownloader downloader;
        if (!downloader.ensureModelsExist()) {
            std::cerr << "Failed to download models" << std::endl;
            return 1;
        }
        
        // Initialize ASR model
        ASRModel::Config asr_config;
        asr_config.model_path = downloader.getModelPath(ModelDownloader::ASR_MODEL_QUANT_NAME);
        asr_config.config_path = downloader.getModelPath(ModelDownloader::CONFIG_NAME);
        asr_config.vocab_path = downloader.getModelPath(ModelDownloader::VOCAB_NAME);
        
        auto asrModel = std::make_shared<ASRModel>(asr_config);
        std::cout << "Loading ASR model..." << std::endl;
        if (!asrModel->initialize()) {
            std::cerr << "Failed to initialize ASR model" << std::endl;
            return 1;
        }
        
        // Initialize streaming recorder
        StreamingAudioRecorder recorder(device_index, sample_rate);
        
        // Set up VAD
        if (vad_type == "silero") {
            std::cout << "Using Silero VAD (trigger: " << vad_trigger_threshold 
                      << ", stop: " << vad_stop_threshold << ")" << std::endl;
            
            // Initialize Silero VAD
            VADDetector::Config vad_config;
            vad_config.model_path = downloader.getModelPath(ModelDownloader::VAD_MODEL_NAME);
            vad_config.sample_rate = 16000;  // Silero expects 16kHz
            
            auto vadDetector = std::make_shared<VADDetector>(vad_config);
            if (!vadDetector->initialize()) {
                std::cerr << "Failed to initialize Silero VAD" << std::endl;
                return 1;
            }
            
            recorder.setUseEnergyVAD(false);
            recorder.setVADDetector(vadDetector);
            recorder.setVADTriggerThreshold(vad_trigger_threshold);
            recorder.setVADStopThreshold(vad_stop_threshold);
        } else {
            std::cout << "Using Energy-based VAD (threshold: " << vad_threshold << ")" << std::endl;
            recorder.setVADThreshold(vad_threshold);
            recorder.setUseEnergyVAD(true);
        }
        
        // Initialize ASR thread pool
        ASRThreadPool asrPool(num_threads);
        asrPool.initialize(asrModel);
        
        // Track results
        std::map<size_t, ASRResult> results;
        std::mutex resultsMutex;
        size_t lastPrintedId = 0;
        
        // Set result callback
        asrPool.setResultCallback([&](const ASRResult& result) {
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                results[result.segment_id] = result;
            }
            
            // Print results in order
            std::lock_guard<std::mutex> lock(resultsMutex);
            while (results.find(lastPrintedId) != results.end()) {
                const auto& res = results[lastPrintedId];
                
                std::cout << "\n[" << std::fixed << std::setprecision(2) 
                          << res.timestamp_start << "s - " 
                          << res.timestamp_end << "s] "
                          << res.text << std::endl;
                std::cout << "(Processing time: " << res.processing_time << "s)" << std::endl;
                
                results.erase(lastPrintedId);
                lastPrintedId++;
            }
            
            std::cout << "Listening..." << std::flush;
        });
        
        // Start ASR thread pool
        asrPool.start();
        
        // Start streaming session
        std::cout << "\n=== Streaming ASR Session ===" << std::endl;
        std::cout << "Max duration: " << max_duration << "s" << std::endl;
        std::cout << "Silence threshold: " << silence_threshold << "s" << std::endl;
        std::cout << "Pre-speech buffer: " << pre_speech_buffer << "s" << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        std::cout << "=============================\n" << std::endl;
        
        recorder.startStreamingSession(
            [&](const AudioSegment& segment) {
                asrPool.processSegment(segment);
            },
            max_duration,
            silence_threshold,
            pre_speech_buffer
        );
        
        // Wait for completion or interruption
        while (g_running) {
            recorder.waitForCompletion();
            if (!g_running) {
                recorder.stopStreamingSession();
            }
            break;
        }
        
        // Stop components
        asrPool.stop();
        
        // Print final statistics
        std::cout << "\n\nSession Statistics:" << std::endl;
        std::cout << "Total segments processed: " << lastPrintedId << std::endl;
        
        // Cleanup PortAudio
        Pa_Terminate();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        Pa_Terminate();
        return 1;
    }
    
    return 0;
}