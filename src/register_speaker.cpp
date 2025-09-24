#include "speaker_recognition.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include "speaker_model_downloader.hpp"
#include <chrono>
#include <thread>
#include <portaudio.h>
#include <cmath>

using namespace speaker_recognition;

static const int kRecordingSampleRate = 16000;
static const float kRecordingDurationSeconds = 4.0f;
static const int kRecordingRepeats = 3;
static const int kRecordingChannels = 1;

static void WaitForEnter(const std::string& prompt) {
    std::cout << prompt << std::flush;
    std::string dummy;
    std::getline(std::cin, dummy);
}

// Speaker recognition audio recorder for registration
class SRAudioRecorder {
public:
    SRAudioRecorder() : stream_(nullptr) {}

    ~SRAudioRecorder() {
        if (stream_) {
            Pa_CloseStream(stream_);
        }
    }

    bool Initialize(int sample_rate, int channels) {
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            last_error_ = Pa_GetErrorText(err);
            return false;
        }

        sample_rate_ = sample_rate;
        channels_ = channels;

        PaStreamParameters inputParameters;
        inputParameters.device = Pa_GetDefaultInputDevice();
        if (inputParameters.device == paNoDevice) {
            last_error_ = "No default input device available";
            return false;
        }

        inputParameters.channelCount = channels;
        inputParameters.sampleFormat = paFloat32;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;

        err = Pa_OpenStream(&stream_,
                            &inputParameters,
                            nullptr,
                            sample_rate,
                            512,
                            paClipOff,
                            nullptr,
                            nullptr);

        if (err != paNoError) {
            last_error_ = Pa_GetErrorText(err);
            return false;
        }

        return true;
    }

    bool Record(float duration, std::vector<float>& samples) {
        samples.clear();

        PaError err = Pa_StartStream(stream_);
        if (err != paNoError) {
            last_error_ = Pa_GetErrorText(err);
            return false;
        }

        int total_frames = static_cast<int>(sample_rate_ * duration);
        int frames_per_buffer = 512;
        std::vector<float> buffer(frames_per_buffer * channels_);

        samples.reserve(total_frames * channels_);

        while (samples.size() < total_frames * channels_) {
            err = Pa_ReadStream(stream_, buffer.data(), frames_per_buffer);
            if (err != paNoError && err != paInputOverflowed) {
                last_error_ = Pa_GetErrorText(err);
                Pa_StopStream(stream_);
                return false;
            }

            samples.insert(samples.end(), buffer.begin(), buffer.end());
        }

        Pa_StopStream(stream_);

        // Trim to exact size
        if (samples.size() > total_frames * channels_) {
            samples.resize(total_frames * channels_);
        }

        return true;
    }

    std::string GetLastError() const { return last_error_; }

private:
    PaStream* stream_;
    int sample_rate_;
    int channels_;
    std::string last_error_;
};

static bool RecordSamples(SRAudioRecorder& recorder,
                          std::vector<float>& samples,
                          int index) {
    WaitForEnter("按 Enter 键开始录音...");
    std::cout << "开始录音 " << (index + 1) << " / " << kRecordingRepeats
              << " (" << kRecordingDurationSeconds << " 秒)..." << std::endl;

    if (!recorder.Record(kRecordingDurationSeconds, samples)) {
        std::cerr << "录音失败: " << recorder.GetLastError() << std::endl;
        return false;
    }

    std::cout << "录音完成，样本数: " << samples.size() << "\n";
    return true;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS] -n NAME audio1.wav [audio2.wav ...]\n";
    std::cout << "\nRegister a speaker with one or more audio samples.\n\n";
    std::cout << "Options:\n";
    std::cout << "  -n, --name NAME       Speaker name (required)\n";
    std::cout << "  -m, --model FILE      Model file path (default: ~/.cache/speaker-recognition/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx)\n";
    std::cout << "  -d, --database FILE   Database file (default: speakers.db)\n";
    std::cout << "  -t, --threads NUM     Number of threads (default: 1)\n";
    std::cout << "  -f, --force          Force overwrite if speaker exists\n";
    std::cout << "  -D, --debug          Enable debug output (timing and embeddings)\n";
    std::cout << "  -h, --help           Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " -n john john_sample.wav\n";
    std::cout << "  " << prog << " -n alice alice1.wav alice2.wav alice3.wav\n";
    std::cout << "  " << prog << " -d company.db -n employee001 sample.wav\n";
}

int main(int argc, char* argv[]) {
    // Suppress ONNX Runtime Schema errors
    setenv("ORT_DISABLE_ALL_LOGS", "1", 1);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Initialize model downloader to resolve default paths
    SRModelDownloader downloader;

    // Default values
    std::string model_path = downloader.getModelPath(SRModelDownloader::AR_MODEL_NAME);
    std::string database_file = "speakers.db";
    std::string speaker_name;
    int32_t num_threads = 1;
    bool force_overwrite = false;
    bool debug = false;
    std::vector<std::string> audio_files;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--name") == 0) {
            if (i + 1 < argc) {
                speaker_name = argv[++i];
            } else {
                std::cerr << "Error: -n requires a speaker name\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                std::cerr << "Error: -m requires a model file path\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--database") == 0) {
            if (i + 1 < argc) {
                database_file = argv[++i];
            } else {
                std::cerr << "Error: -d requires a database file path\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: -t requires a number\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--force") == 0) {
            force_overwrite = true;
        } else if (strcmp(argv[i], "-D") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (argv[i][0] != '-') {
            // Audio file
            audio_files.push_back(argv[i]);
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            return 1;
        }
    }

    // Validate arguments
    if (speaker_name.empty()) {
        std::cerr << "Error: Speaker name is required (-n NAME)\n";
        print_usage(argv[0]);
        return 1;
    }

    bool use_recording = audio_files.empty();
    if (!use_recording && audio_files.size() < kRecordingRepeats) {
        std::cout << "提示: 建议至少提供 " << kRecordingRepeats << " 个音频样本以获得稳定效果\n";
    }

    if (!downloader.ensureModelsExist()) {
        std::cerr << "Failed to ensure speaker recognition models exist" << std::endl;
        return 1;
    }

    // Create embedder
    EmbedderConfig config;
    config.model_path = model_path;
    config.num_threads = num_threads;
    config.debug = debug;
    config.provider = "cpu";

    std::cout << "Loading model from: " << model_path << "\n";
    auto embedder = SpeakerEmbedder::Create(config);
    if (!embedder) {
        std::cerr << "Failed to create speaker embedder\n";
        return 1;
    }

    // Create manager
    auto manager = SpeakerManager::Create(embedder->GetEmbeddingDimension());
    if (!manager) {
        std::cerr << "Failed to create speaker manager\n";
        return 1;
    }

    // Load existing database
    manager->LoadDatabase(database_file);

    // Check if speaker already exists
    if (manager->ContainsSpeaker(speaker_name) && !force_overwrite) {
        std::cerr << "Error: Speaker '" << speaker_name << "' already exists in database.\n";
        std::cerr << "Use -f to force overwrite.\n";
        return 1;
    }

    // Process audio files
    std::vector<float> recorded_samples;
    std::vector<std::vector<float>> embeddings;

    if (use_recording) {
        SRAudioRecorder recorder;
        if (!recorder.Initialize(kRecordingSampleRate, kRecordingChannels)) {
            std::cerr << "录音初始化失败: " << recorder.GetLastError() << "\n";
            return 1;
        }

        std::cout << "进入录音模式，将录制 " << kRecordingRepeats
                  << " 次，每次 " << kRecordingDurationSeconds << " 秒。\n";
        std::cout << "请确保麦克风可用。若准备好，请按 Enter 进入录音流程。\n";
        WaitForEnter("按 Enter 开始...");

        for (int i = 0; i < kRecordingRepeats; ++i) {
            if (!RecordSamples(recorder, recorded_samples, i)) {
                return 1;
            }

            auto stream = embedder->CreateStream();
            stream->AcceptWaveform(kRecordingSampleRate, recorded_samples);
            stream->InputFinished();
            if (!embedder->IsStreamReady(stream.get())) {
                std::cerr << "录音样本太短，请重试。\n";
                return 1;
            }

            auto embedding = embedder->ComputeEmbedding(stream.get());
            if (embedding.empty()) {
                std::cerr << "生成嵌入失败，请重试。\n";
                return 1;
            }
            embeddings.push_back(std::move(embedding));
        }

        Pa_Terminate();
    } else {
        std::cout << "Processing " << audio_files.size() << " audio file(s) for speaker '"
                  << speaker_name << "'...\n";

        for (const auto& file : audio_files) {
            std::cout << "  Processing: " << file << "...\n";
            auto embedding = embedder->ComputeEmbeddingFromFile(file);
            if (embedding.empty()) {
                std::cerr << "    Failed to compute embedding for: " << file << "\n";
                continue;
            }
            embeddings.push_back(embedding);
            std::cout << "    Done\n";
        }
    }

    if (embeddings.empty()) {
        std::cerr << "Error: No valid embeddings computed\n";
        return 1;
    }

    // Register speaker
    if (force_overwrite && manager->ContainsSpeaker(speaker_name)) {
        manager->RemoveSpeaker(speaker_name);
    }

    bool success = manager->RegisterSpeaker(speaker_name, embeddings);
    if (!success) {
        std::cerr << "Failed to register speaker\n";
        return 1;
    }

    // Save database
    if (!manager->SaveDatabase(database_file)) {
        std::cerr << "Failed to save database\n";
        return 1;
    }

    std::cout << "Successfully registered speaker '" << speaker_name << "' with "
              << embeddings.size() << " sample(s)\n";
    std::cout << "Database saved to: " << database_file << "\n";
    std::cout << "Total speakers in database: " << manager->GetSpeakerCount() << "\n";

    return 0;
}