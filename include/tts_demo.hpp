#pragma once

#include <string>
#include <memory>
#include "tts/tts_model.hpp"

class TTSDemo {
public:
    struct Params {
        float tts_speed;
        int tts_speaker_id;
        float target_rms;
        float compression_ratio;
        float compression_threshold;
        bool use_rms_norm;
        Params() :
            tts_speed(1.0f),
            tts_speaker_id(0),
            target_rms(0.15f),
            compression_ratio(2.0f),
            compression_threshold(0.7f),
            use_rms_norm(true) {}
    };

    TTSDemo(const Params& params = Params());
    ~TTSDemo() = default;

    bool initialize();
    
    void run(const std::string& text, const std::string& save_audio_path = "");

private:
    Params params_;
    std::unique_ptr<tts::TTSModel> tts_model_;
}; 