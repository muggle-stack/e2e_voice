#include "tts_demo.hpp"
#include "tts/tts_model_downloader.hpp"
#include <iostream>
#include <fstream>
#include <sndfile.h>

TTSDemo::TTSDemo(const Params& params) : params_(params) {}

bool TTSDemo::initialize(){
    tts::TTSModelDownloader tts_downloader;
    if (!tts_downloader.ensureModelsExist()) {
        std::cerr << "Failed to ensure TTS models exist" << std::endl;
        return false;
    }
    
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
    return true;
}

void TTSDemo::run(const std::string& text, const std::string& save_audio_path) {
    std::cout << "Generating TTS for text: " << text << std::endl;
    tts::GeneratedAudio generated_audio = tts_model_->generate(text, params_.tts_speaker_id, params_.tts_speed);

    if (generated_audio.samples.empty()) {
        std::cerr << "Failed to generate TTS for text: " << text << std::endl;
        return;
    }

    if (!save_audio_path.empty()) {
        std::cout << "Saving audio to: " << save_audio_path << std::endl;
        
        // 设置WAV文件格式信息
        SF_INFO sf_info;
        sf_info.samplerate = tts_model_->getSampleRate();
        sf_info.channels = 1;  // 单声道
        sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
        
        SNDFILE* file = sf_open(save_audio_path.c_str(), SFM_WRITE, &sf_info);
        if (!file) {
            std::cerr << "Failed to open file for writing: " << sf_strerror(NULL) << std::endl;
            return;
        }
        
        sf_count_t written = sf_write_float(file, generated_audio.samples.data(), generated_audio.samples.size());
        sf_close(file);
        
        std::cout << "Successfully saved " << written << " samples to " << save_audio_path << std::endl;
    }
} 