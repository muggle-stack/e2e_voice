#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sndfile.h>
#include <portaudio.h>
#include "asr_realtime_api.hpp"
#include "audio_recorder.hpp"
#include "vad_detector.hpp"
#include "model_downloader.hpp"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --access_key_id <id>      阿里云AccessKey ID (或设置环境变量 ALIYUN_ACCESS_KEY_ID)" << std::endl;
    std::cout << "  --access_key_secret <sec> 阿里云AccessKey Secret (或设置环境变量 ALIYUN_ACCESS_KEY_SECRET)" << std::endl;
    std::cout << "  --appkey <key>            语音服务AppKey (或设置环境变量 ALIYUN_ASR_APPKEY)" << std::endl;
    std::cout << "  --token <token>           直接使用Token (或设置环境变量 ALIYUN_NLS_TOKEN)" << std::endl;
    std::cout << "  --audio_file <path>       音频文件路径" << std::endl;
    std::cout << "  --sample_rate <value>     采样率 (默认: 16000)" << std::endl;
    std::cout << "  --channels <value>        音频通道数 (默认: 1)" << std::endl;
    std::cout << "  --device_index <value>    音频设备索引 (默认: -1 自动选择)" << std::endl;
    std::cout << "  --silence_duration <sec>  静音停止时长 (默认: 1.0)" << std::endl;
    std::cout << "  --max_record_time <sec>   最大录音时长 (默认: 60.0)" << std::endl;
    std::cout << "  --vad_type <type>         VAD类型: energy 或 silero (默认: energy)" << std::endl;
    std::cout << "  --help                    显示帮助信息" << std::endl;
    std::cout << "\n环境变量:" << std::endl;
    std::cout << "  ALIYUN_ACCESS_KEY_ID      阿里云AccessKey ID" << std::endl;
    std::cout << "  ALIYUN_ACCESS_KEY_SECRET  阿里云AccessKey Secret" << std::endl;
    std::cout << "  ALIYUN_ASR_APPKEY         语音服务AppKey" << std::endl;
    std::cout << "  ALIYUN_NLS_TOKEN          直接使用Token（24小时有效，用于快速测试）" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  # 使用环境变量配置（快速测试模式）" << std::endl;
    std::cout << "  export ALIYUN_NLS_TOKEN=你的Token" << std::endl;
    std::cout << "  export ALIYUN_ASR_APPKEY=tt43P2u****" << std::endl;
    std::cout << "  " << program_name << "  # 交互模式（按Enter录音）" << std::endl;
    std::cout << "\n  # 或测试音频文件" << std::endl;
    std::cout << "  " << program_name << " --audio_file test.wav" << std::endl;
    std::cout << "\n  # 使用双声道录音" << std::endl;
    std::cout << "  " << program_name << " --channels 2 --sample_rate 48000" << std::endl;
    std::cout << "\n注意:" << std::endl;
    std::cout << "  - 音频时长需 ≤ 60秒" << std::endl;
    std::cout << "  - 支持采样率: 8000Hz 或 16000Hz" << std::endl;
    std::cout << "  - 获取密钥: https://ram.console.aliyun.com/manage/ak" << std::endl;
    std::cout << "  - 获取AppKey: https://nls-portal.console.aliyun.com/" << std::endl;
    std::cout << "  - 获取测试Token: 控制台->总览->获取临时AccessToken" << std::endl;
}

std::vector<float> loadAudioFile(const std::string& filepath, int& sample_rate) {
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open(filepath.c_str(), SFM_READ, &sfinfo);
    if (!infile) {
        std::cerr << "Error: Failed to open audio file: " << filepath << std::endl;
        return {};
    }

    std::vector<float> audio(sfinfo.frames * sfinfo.channels);
    sf_count_t num_frames = sf_readf_float(infile, audio.data(), sfinfo.frames);
    sf_close(infile);

    if (num_frames != sfinfo.frames) {
        std::cerr << "Warning: Only read " << num_frames << " frames out of " << sfinfo.frames << std::endl;
    }

    sample_rate = sfinfo.samplerate;

    // 如果是多声道，转为单声道
    if (sfinfo.channels > 1) {
        std::vector<float> mono(sfinfo.frames);
        for (sf_count_t i = 0; i < sfinfo.frames; ++i) {
            float sum = 0;
            for (int ch = 0; ch < sfinfo.channels; ++ch) {
                sum += audio[i * sfinfo.channels + ch];
            }
            mono[i] = sum / sfinfo.channels;
        }
        audio = std::move(mono);
    }

    std::cout << "Loaded audio: " << audio.size() << " samples, " << sample_rate << " Hz, "
              << (float)audio.size() / sample_rate << " seconds" << std::endl;

    return audio;
}

int main(int argc, char* argv[]) {
    ASRRealtimeClient::Config config;
    AudioRecorder::Config recorder_config;
    std::string audio_file;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--access_key_id" && i + 1 < argc) {
            config.access_key_id = argv[++i];
        } else if (arg == "--access_key_secret" && i + 1 < argc) {
            config.access_key_secret = argv[++i];
        } else if (arg == "--token" && i + 1 < argc) {
            config.token = argv[++i];
        } else if (arg == "--appkey" && i + 1 < argc) {
            config.appkey = argv[++i];
        } else if (arg == "--audio_file" && i + 1 < argc) {
            audio_file = argv[++i];
        } else if (arg == "--sample_rate" && i + 1 < argc) {
            recorder_config.sample_rate = std::atoi(argv[++i]);
        } else if (arg == "--channels" && i + 1 < argc) {
            recorder_config.channels = std::atoi(argv[++i]);
        } else if (arg == "--device_index" && i + 1 < argc) {
            recorder_config.device_index = std::atoi(argv[++i]);
        } else if (arg == "--silence_duration" && i + 1 < argc) {
            recorder_config.silence_duration = std::atof(argv[++i]);
        } else if (arg == "--max_record_time" && i + 1 < argc) {
            recorder_config.max_record_time = std::atof(argv[++i]);
        } else if (arg == "--vad_type" && i + 1 < argc) {
            recorder_config.vad_type = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== 阿里云实时ASR测试（一句话识别）===" << std::endl;

    // 创建ASR客户端
    auto asr_client = std::make_unique<ASRRealtimeClient>(config);

    // 初始化
    if (!asr_client->initialize()) {
        std::cerr << "Failed to initialize ASRRealtimeClient" << std::endl;
        std::cerr << "Error: " << asr_client->getLastError() << std::endl;
        return 1;
    }

    if (!audio_file.empty()) {
        // 文件模式
        int sample_rate;
        std::vector<float> audio = loadAudioFile(audio_file, sample_rate);
        if (audio.empty()) {
            return 1;
        }

        std::cout << "\n开始识别..." << std::endl;
        std::string result = asr_client->recognize(audio, sample_rate);

        if (result.empty()) {
            std::cerr << "\n识别失败!" << std::endl;
            std::cerr << "Error: " << asr_client->getLastError() << std::endl;
            return 1;
        }

        std::cout << "\n=== 识别结果 ===" << std::endl;
        std::cout << result << std::endl;
        std::cout << "================" << std::endl;
    } else {
        // 交互模式（实时录音）
        std::cout << "\n进入交互模式（实时对话）" << std::endl;
        std::cout << "初始化音频录制..." << std::endl;

        // 初始化PortAudio
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(err) << std::endl;
            return 1;
        }

        // 如果使用Silero VAD，需要初始化VAD检测器
        std::unique_ptr<VADDetector> vad_detector;
        if (recorder_config.vad_type == "silero") {
            ModelDownloader downloader;
            if (!downloader.ensureModelsExist()) {
                std::cerr << "Failed to download required models" << std::endl;
                Pa_Terminate();
                return 1;
            }

            VADDetector::Config vad_config;
            vad_config.model_path = downloader.getModelPath(ModelDownloader::VAD_MODEL_NAME);
            vad_config.sample_rate = 16000;
            vad_config.window_size = 512;
            vad_config.context_size = 64;

            vad_detector = std::make_unique<VADDetector>(vad_config);
            if (!vad_detector->initialize()) {
                std::cerr << "Failed to initialize Silero VAD detector" << std::endl;
                Pa_Terminate();
                return 1;
            }
            std::cout << "Using Silero VAD for voice activity detection" << std::endl;
        } else {
            std::cout << "Using energy-based VAD for voice activity detection" << std::endl;
        }

        // 创建AudioRecorder
        auto audio_recorder = std::make_unique<AudioRecorder>(recorder_config);
        if (!audio_recorder->initialize()) {
            std::cerr << "Failed to initialize AudioRecorder" << std::endl;
            Pa_Terminate();
            return 1;
        }

        // 设置VAD检测器（如果使用Silero）
        if (vad_detector) {
            audio_recorder->setVADDetector(vad_detector.get());
        }

        std::cout << "音频录制初始化完成!" << std::endl;
        std::cout << "\n配置:" << std::endl;
        std::cout << "  采样率: " << recorder_config.sample_rate << " Hz" << std::endl;
        std::cout << "  声道数: " << recorder_config.channels << std::endl;
        std::cout << "  最大录音时长: " << recorder_config.max_record_time << " 秒" << std::endl;
        std::cout << "  静音停止时长: " << recorder_config.silence_duration << " 秒" << std::endl;
        std::cout << "  VAD类型: " << recorder_config.vad_type << std::endl;
        std::cout << "\n按Enter键开始录音，说话后保持" << recorder_config.silence_duration
                  << "秒静音自动停止" << std::endl;
        std::cout << "输入 'quit' 退出\n" << std::endl;

        std::string line;
        int turn = 0;
        while (true) {
            std::cout << "\n[轮次 " << (++turn) << "] 按Enter开始录音（或输入quit退出）: ";
            std::getline(std::cin, line);

            if (line == "quit" || line == "q") {
                std::cout << "退出程序" << std::endl;
                break;
            }

            // 录音
            std::cout << "开始录音... (说话后保持静音 " << recorder_config.silence_duration
                      << " 秒自动停止，或最多 " << recorder_config.max_record_time << " 秒)" << std::endl;

            auto record_start = std::chrono::high_resolution_clock::now();
            std::vector<float> audio = audio_recorder->recordAudio();
            auto record_end = std::chrono::high_resolution_clock::now();

            auto record_duration = std::chrono::duration<double>(record_end - record_start).count();

            if (audio.empty()) {
                std::cout << "未检测到语音或录音失败，请重试" << std::endl;
                continue;
            }

            std::cout << "录音完成 (" << record_duration << "秒, "
                      << audio.size() << " 采样点, "
                      << (float)audio.size() / recorder_config.sample_rate << "秒音频)" << std::endl;

            // 转换双声道为单声道（如果需要）
            std::vector<float> mono_audio;
            if (recorder_config.channels == 2) {
                std::cout << "转换双声道到单声道..." << std::endl;
                mono_audio.reserve(audio.size() / 2);
                for (size_t i = 0; i < audio.size(); i += 2) {
                    // 平均左右声道
                    float left = audio[i];
                    float right = audio[i + 1];
                    mono_audio.push_back((left + right) / 2.0f);
                }
                std::cout << "转换完成: " << mono_audio.size() << " 采样点" << std::endl;
            } else {
                mono_audio = audio;
            }

            // 识别（重采样在 ASRRealtimeClient 内部自动完成）
            std::cout << "\n========== ASR 识别阶段 ==========" << std::endl;
            auto asr_start = std::chrono::high_resolution_clock::now();
            std::string result = asr_client->recognize(mono_audio, recorder_config.sample_rate);
            auto asr_end = std::chrono::high_resolution_clock::now();

            auto asr_duration = std::chrono::duration<double>(asr_end - asr_start).count();
            std::cout << "=================================" << std::endl;

            if (result.empty()) {
                std::cerr << "识别失败: " << asr_client->getLastError() << std::endl;
                continue;
            }

            float audio_duration = (float)mono_audio.size() / recorder_config.sample_rate;
            std::cout << "\n========== 识别结果 ==========" << std::endl;
            std::cout << "识别文本: " << result << std::endl;
            std::cout << "总耗时: " << std::fixed << std::setprecision(3) << asr_duration << " 秒 ("
                      << (int)(asr_duration * 1000) << " ms)" << std::endl;
            std::cout << "音频时长: " << audio_duration << " 秒" << std::endl;
            std::cout << "实时率: " << std::setprecision(2)
                      << (asr_duration / audio_duration) << "x" << std::endl;
            std::cout << "=============================" << std::endl;
            std::cout << "\n💡 提示: 在 main_asr_llm_tts.cpp 中，识别结果会发送给LLM并转为语音回复" << std::endl;
        }

        // 清理
        audio_recorder->cleanup();
        Pa_Terminate();
    }

    return 0;
}
