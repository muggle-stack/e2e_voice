#ifndef SPEAKER_RECOGNITION_HPP_
#define SPEAKER_RECOGNITION_HPP_

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace speaker_recognition {

// Wave data structure
struct Wave {
    std::vector<float> samples;
    int32_t sample_rate;

    Wave() : sample_rate(0) {}
    Wave(std::vector<float> samples, int32_t rate)
        : samples(std::move(samples)), sample_rate(rate) {}
};

// Configuration for speaker embedder
struct EmbedderConfig {
    std::string model_path;
    int32_t num_threads = 1;
    bool debug = false;
    std::string provider = "cpu";
};

// Speaker match result
struct SpeakerMatch {
    float score;
    std::string name;

    SpeakerMatch() : score(0.0f) {}
    SpeakerMatch(float s, const std::string& n) : score(s), name(n) {}
};

// Forward declarations
class Stream;
class SpeakerEmbedder;
class SpeakerManager;

// Wave file reader
class WaveReader {
public:
    static std::unique_ptr<Wave> ReadFile(const std::string& filename);
};

// Stream for processing audio
class Stream {
public:
    virtual ~Stream() = default;

    virtual void AcceptWaveform(int32_t sample_rate,
                                const float* samples,
                                int32_t num_samples) = 0;
    virtual void AcceptWaveform(int32_t sample_rate,
                                const std::vector<float>& samples) {
        AcceptWaveform(sample_rate, samples.data(), samples.size());
    }
    virtual void AcceptWaveform(int32_t sample_rate,
                                std::vector<float>&& samples) {
        AcceptWaveform(sample_rate, samples.data(), samples.size());
    }
    virtual void InputFinished() = 0;
};

// Speaker embedder for computing embeddings from audio
class SpeakerEmbedder {
public:
    virtual ~SpeakerEmbedder() = default;

    // Create embedder with configuration
    static std::unique_ptr<SpeakerEmbedder> Create(const EmbedderConfig& config);

    // Get embedding dimension
    virtual int32_t GetEmbeddingDimension() const = 0;

    // Create a new stream for processing
    virtual std::unique_ptr<Stream> CreateStream() const = 0;

    // Check if stream has enough data
    virtual bool IsStreamReady(const Stream* stream) const = 0;

    // Compute embedding from stream
    virtual std::vector<float> ComputeEmbedding(const Stream* stream) const = 0;

    // Compute embedding with RTF (Real Time Factor)
    virtual std::vector<float> ComputeEmbedding(const Stream* stream, float& rtf_out) const = 0;

    // Helper function to compute embedding from file
    virtual std::vector<float> ComputeEmbeddingFromFile(const std::string& filename) const = 0;
};

// Speaker manager for registration and identification
class SpeakerManager {
public:
    virtual ~SpeakerManager() = default;

    // Create manager with embedding dimension
    static std::unique_ptr<SpeakerManager> Create(int32_t dim);

    // Register a speaker with single embedding
    virtual bool RegisterSpeaker(const std::string& name,
                                 const std::vector<float>& embedding) = 0;

    // Register a speaker with multiple embeddings
    virtual bool RegisterSpeaker(const std::string& name,
                                 const std::vector<std::vector<float>>& embeddings) = 0;

    // Remove a speaker
    virtual bool RemoveSpeaker(const std::string& name) = 0;

    // Search for speaker
    virtual std::string SearchSpeaker(const std::vector<float>& embedding,
                                      float threshold) const = 0;

    // Get best matches
    virtual std::vector<SpeakerMatch> GetBestMatches(
        const std::vector<float>& embedding,
        float threshold,
        int32_t max_results) const = 0;

    // Verify if embedding matches a specific speaker
    virtual bool VerifySpeaker(const std::string& name,
                               const std::vector<float>& embedding,
                               float threshold) const = 0;

    // Check if speaker exists
    virtual bool ContainsSpeaker(const std::string& name) const = 0;

    // Get speaker count
    virtual int32_t GetSpeakerCount() const = 0;

    // Get all speaker names
    virtual std::vector<std::string> GetAllSpeakers() const = 0;

    // Database operations
    virtual bool SaveDatabase(const std::string& filename) const = 0;
    virtual bool LoadDatabase(const std::string& filename) = 0;
};

} // namespace speaker_recognition

#endif // SPEAKER_RECOGNITION_HPP_