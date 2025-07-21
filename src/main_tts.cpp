#include <iostream>
#include <string>
#include <cstdlib>
#include "tts_demo.hpp"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --text <value>              Text to convert to speech" << std::endl;
    std::cout << "  --save_audio_path <value>   Path to save generated audio file" << std::endl;
    std::cout << "  --tts_speed <value>         TTS speech speed (default: 1.0, >1.0 = slower)" << std::endl;
    std::cout << "  --tts_speaker_id <value>    TTS speaker ID for multi-speaker models (default: 0)" << std::endl;
    std::cout << "  --target_rms <value>        Target RMS level for volume normalization (default: 0.1, range: 0.05-0.3)" << std::endl;
    std::cout << "  --compression_ratio <value> Dynamic range compression ratio (default: 3.0, range: 1.0-10.0)" << std::endl;
    std::cout << "  --use_peak_norm             Use peak normalization instead of RMS (not recommended)" << std::endl;
    std::cout << "  --help                      Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "TTS Demo Application" << std::endl;
    std::cout << "=================================" << std::endl;

    TTSDemo::Params params;
    std::string text;
    std::string save_audio_path;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--text" && i + 1 < argc) {
            text = argv[++i];
        }
        else if (arg == "--save_audio_path" && i + 1 < argc) {
            save_audio_path = argv[++i];
        }
        else if (arg == "--tts_speed" && i + 1 < argc) {
            params.tts_speed = std::atof(argv[++i]);
        }
        else if (arg == "--tts_speaker_id" && i + 1 < argc) {
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
            std::cerr << "Invalid argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (text.empty()) {
        text = "你好，这是一个语音合成测试。";
        std::cout << "No text provided, using default: " << text << std::endl;
    }

    try {
        TTSDemo demo(params);
        if (!demo.initialize()) {
            std::cerr << "Failed to initialize demo" << std::endl;
            return 1;
        }

        demo.run(text, save_audio_path);
        std::cout << "Demo finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}