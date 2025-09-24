#include "../include/speaker_recognition.hpp"
#include "onnx_utils.hpp"
#include "mel_filterbank.hpp"
#include <chrono>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace speaker_recognition {

// Stream implementation
class StreamImpl : public Stream {
public:
    StreamImpl() : input_finished_(false) {
        // Pre-allocate for typical audio length (5 seconds at 16kHz)
        samples_.reserve(5 * 16000);
    }

    void AcceptWaveform(int32_t sample_rate,
                       const float* samples,
                       int32_t num_samples) override {
        // Append samples to buffer
        samples_.insert(samples_.end(), samples, samples + num_samples);
        sample_rate_ = sample_rate;
    }

    void AcceptWaveform(int32_t sample_rate,
                       std::vector<float>&& samples) override {
        // Move semantics for efficiency when we own the data
        if (samples_.empty()) {
            samples_ = std::move(samples);
        } else {
            samples_.insert(samples_.end(),
                          std::make_move_iterator(samples.begin()),
                          std::make_move_iterator(samples.end()));
        }
        sample_rate_ = sample_rate;
    }

    void InputFinished() override {
        input_finished_ = true;
    }

    const std::vector<float>& GetSamples() const { return samples_; }
    int32_t GetSampleRate() const { return sample_rate_; }
    bool IsInputFinished() const { return input_finished_; }

private:
    std::vector<float> samples_;
    int32_t sample_rate_ = 16000;
    bool input_finished_;
};

// Speaker Embedder implementation
class SpeakerEmbedderImpl : public SpeakerEmbedder {
public:
    SpeakerEmbedderImpl(const EmbedderConfig& config)
        : config_(config),
          model_(config.model_path, config.num_threads, config.provider) {
        embedding_dim_ = model_.GetEmbeddingDim();
    }

    int32_t GetEmbeddingDimension() const override {
        return embedding_dim_;
    }

    std::unique_ptr<Stream> CreateStream() const override {
        return std::make_unique<StreamImpl>();
    }

    bool IsStreamReady(const Stream* stream) const override {
        auto* stream_impl = dynamic_cast<const StreamImpl*>(stream);
        if (!stream_impl) return false;

        const auto& samples = stream_impl->GetSamples();
        // Minimum 0.3 seconds of audio at 16kHz
        return samples.size() >= 0.3 * stream_impl->GetSampleRate();
    }

    std::vector<float> ComputeEmbedding(const Stream* stream) const override {
        float rtf;
        return ComputeEmbedding(stream, rtf);
    }

    std::vector<float> ComputeEmbedding(const Stream* stream, float& rtf_out) const override {
        auto start = std::chrono::high_resolution_clock::now();

        auto* stream_impl = dynamic_cast<const StreamImpl*>(stream);
        if (!stream_impl || !IsStreamReady(stream)) {
            return std::vector<float>();
        }

        // Get samples
        const auto& samples = stream_impl->GetSamples();
        int32_t sample_rate = stream_impl->GetSampleRate();

        // Compute mel spectrogram
        auto mel_start = std::chrono::high_resolution_clock::now();
        std::vector<int64_t> shape;
        auto fbank = MelFilterbank::ComputeFbank(samples, sample_rate, shape);
        auto mel_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> mel_duration = mel_end - mel_start;

        // Run inference
        auto infer_start = std::chrono::high_resolution_clock::now();
        auto embedding = model_.RunInference(fbank, shape);
        auto infer_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> infer_duration = infer_end - infer_start;

        // Normalize embedding
        NormalizeEmbedding(embedding);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        float audio_duration = static_cast<float>(samples.size()) / sample_rate;
        rtf_out = duration.count() / audio_duration;

        // Print timing information
        if (config_.debug) {
            std::cout << "Timing Information:" << std::endl;
            std::cout << "  Audio duration: " << audio_duration << " s" << std::endl;
            std::cout << "  Mel spectrogram: " << mel_duration.count() * 1000 << " ms" << std::endl;
            std::cout << "  Model inference: " << infer_duration.count() * 1000 << " ms" << std::endl;
            std::cout << "  Total processing: " << duration.count() * 1000 << " ms" << std::endl;
            std::cout << "  RTF (Real-Time Factor): " << rtf_out << std::endl;
        }

        // Print embedding information
        if (config_.debug && !embedding.empty()) {
            std::cout << "Embedding Information:" << std::endl;
            std::cout << "  Dimension: " << embedding.size() << std::endl;

            // Calculate norm
            float norm = 0.0f;
            for (float val : embedding) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            std::cout << "  L2 Norm: " << norm << std::endl;

            // Print first few values
            std::cout << "  First 10 values: ";
            for (size_t i = 0; i < std::min(size_t(10), embedding.size()); ++i) {
                std::cout << embedding[i] << " ";
            }
            std::cout << std::endl;
        }

        return embedding;
    }

    std::vector<float> ComputeEmbeddingFromFile(const std::string& filename) const override {
        auto wave = WaveReader::ReadFile(filename);
        if (!wave) {
            return std::vector<float>();
        }

        auto stream = CreateStream();
        // Use move semantics since wave is about to be destroyed
        stream->AcceptWaveform(wave->sample_rate, std::move(wave->samples));
        stream->InputFinished();

        if (!IsStreamReady(stream.get())) {
            return std::vector<float>();
        }

        return ComputeEmbedding(stream.get());
    }

private:
    void NormalizeEmbedding(std::vector<float>& embedding) const {
        // L2 normalization
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);

        if (norm > 0) {
            for (float& val : embedding) {
                val /= norm;
            }
        }
    }

    EmbedderConfig config_;
    OnnxModel model_;
    int32_t embedding_dim_;
};

// Factory function
std::unique_ptr<SpeakerEmbedder> SpeakerEmbedder::Create(const EmbedderConfig& config) {
    return std::make_unique<SpeakerEmbedderImpl>(config);
}

} // namespace speaker_recognition