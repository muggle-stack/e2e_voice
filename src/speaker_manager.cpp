#include "../include/speaker_recognition.hpp"
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <cmath>
#include <iostream>

namespace speaker_recognition {

class SpeakerManagerImpl : public SpeakerManager {
public:
    explicit SpeakerManagerImpl(int32_t embedding_dim)
        : embedding_dim_(embedding_dim) {}

    bool RegisterSpeaker(const std::string& name,
                         const std::vector<float>& embedding) override {
        if (embedding.size() != embedding_dim_) {
            std::cerr << "Embedding dimension mismatch" << std::endl;
            return false;
        }

        speakers_[name] = embedding;
        return true;
    }

    bool RegisterSpeaker(const std::string& name,
                         const std::vector<std::vector<float>>& embeddings) override {
        if (embeddings.empty()) {
            return false;
        }

        // Average multiple embeddings
        std::vector<float> avg_embedding(embedding_dim_, 0.0f);
        int count = 0;

        for (const auto& embedding : embeddings) {
            if (embedding.size() == embedding_dim_) {
                for (size_t i = 0; i < embedding_dim_; i++) {
                    avg_embedding[i] += embedding[i];
                }
                count++;
            }
        }

        if (count == 0) {
            return false;
        }

        // Average and normalize
        for (float& val : avg_embedding) {
            val /= count;
        }
        NormalizeEmbedding(avg_embedding);

        speakers_[name] = avg_embedding;
        return true;
    }

    bool RemoveSpeaker(const std::string& name) override {
        return speakers_.erase(name) > 0;
    }

    std::string SearchSpeaker(const std::vector<float>& embedding,
                             float threshold) const override {
        auto matches = GetBestMatches(embedding, threshold, 1);
        if (!matches.empty() && matches[0].score >= threshold) {
            return matches[0].name;
        }
        return "";
    }

    std::vector<SpeakerMatch> GetBestMatches(
        const std::vector<float>& embedding,
        float threshold,
        int32_t max_results) const override {

        std::vector<SpeakerMatch> matches;
        matches.reserve(speakers_.size());

        for (const auto& [name, speaker_embedding] : speakers_) {
            float score = ComputeSimilarity(embedding, speaker_embedding);
            if (score >= threshold) {
                matches.emplace_back(score, name);
            }
        }

        // Use partial_sort for better performance when max_results < total matches
        if (max_results < matches.size()) {
            std::partial_sort(matches.begin(),
                            matches.begin() + max_results,
                            matches.end(),
                            [](const SpeakerMatch& a, const SpeakerMatch& b) {
                                return a.score > b.score;
                            });
            matches.resize(max_results);
        } else {
            std::sort(matches.begin(), matches.end(),
                     [](const SpeakerMatch& a, const SpeakerMatch& b) {
                         return a.score > b.score;
                     });
        }

        return matches;
    }

    bool VerifySpeaker(const std::string& name,
                       const std::vector<float>& embedding,
                       float threshold) const override {
        auto it = speakers_.find(name);
        if (it == speakers_.end()) {
            return false;
        }

        float score = ComputeSimilarity(embedding, it->second);
        return score >= threshold;
    }

    bool ContainsSpeaker(const std::string& name) const override {
        return speakers_.find(name) != speakers_.end();
    }

    int32_t GetSpeakerCount() const override {
        return speakers_.size();
    }

    std::vector<std::string> GetAllSpeakers() const override {
        std::vector<std::string> names;
        names.reserve(speakers_.size());
        for (const auto& [name, _] : speakers_) {
            names.push_back(name);
        }
        return names;
    }

    bool SaveDatabase(const std::string& filename) const override {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }

        // Write magic number
        uint32_t magic = 0x53504B52;  // "SPKR"
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        // Write version
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Write embedding dimension
        file.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(embedding_dim_));

        // Write number of speakers
        int32_t num_speakers = speakers_.size();
        file.write(reinterpret_cast<const char*>(&num_speakers), sizeof(num_speakers));

        // Write each speaker
        for (const auto& [name, embedding] : speakers_) {
            // Write name length and name
            int32_t name_len = name.length();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(name.c_str(), name_len);

            // Write embedding
            file.write(reinterpret_cast<const char*>(embedding.data()),
                      embedding.size() * sizeof(float));
        }

        return file.good();
    }

    bool LoadDatabase(const std::string& filename) override {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file for reading: " << filename << std::endl;
            return false;
        }

        // Read magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x53504B52) {  // "SPKR"
            std::cerr << "Invalid database file format" << std::endl;
            return false;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            std::cerr << "Unsupported database version: " << version << std::endl;
            return false;
        }

        // Read embedding dimension
        int32_t file_embedding_dim;
        file.read(reinterpret_cast<char*>(&file_embedding_dim), sizeof(file_embedding_dim));
        if (file_embedding_dim != embedding_dim_) {
            std::cerr << "Embedding dimension mismatch. Expected: " << embedding_dim_
                     << ", Got: " << file_embedding_dim << std::endl;
            return false;
        }

        // Read number of speakers
        int32_t num_speakers;
        file.read(reinterpret_cast<char*>(&num_speakers), sizeof(num_speakers));

        // Clear existing speakers
        speakers_.clear();

        // Read each speaker
        for (int32_t i = 0; i < num_speakers; i++) {
            // Read name
            int32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

            std::string name(name_len, '\0');
            file.read(&name[0], name_len);

            // Read embedding
            std::vector<float> embedding(embedding_dim_);
            file.read(reinterpret_cast<char*>(embedding.data()),
                     embedding_dim_ * sizeof(float));

            if (!file) {
                std::cerr << "Error reading speaker data" << std::endl;
                return false;
            }

            speakers_[name] = embedding;
        }

        std::cout << "Loaded " << speakers_.size() << " speakers from database" << std::endl;
        return true;
    }

private:
    float ComputeSimilarity(const std::vector<float>& emb1,
                           const std::vector<float>& emb2) const {
        // Cosine similarity
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;

        for (size_t i = 0; i < embedding_dim_; i++) {
            dot_product += emb1[i] * emb2[i];
            norm1 += emb1[i] * emb1[i];
            norm2 += emb2[i] * emb2[i];
        }

        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);

        if (norm1 > 0 && norm2 > 0) {
            return dot_product / (norm1 * norm2);
        }

        return 0.0f;
    }

    void NormalizeEmbedding(std::vector<float>& embedding) const {
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

    int32_t embedding_dim_;
    std::unordered_map<std::string, std::vector<float>> speakers_;
};

// Factory function
std::unique_ptr<SpeakerManager> SpeakerManager::Create(int32_t dim) {
    return std::make_unique<SpeakerManagerImpl>(dim);
}

} // namespace speaker_recognition