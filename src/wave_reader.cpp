#include "../include/speaker_recognition.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

namespace speaker_recognition {

struct WaveHeader {
    uint32_t chunk_id;
    uint32_t chunk_size;
    uint32_t format;
    uint32_t subchunk1_id;
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint32_t subchunk2_id;
    uint32_t subchunk2_size;
};

std::unique_ptr<Wave> WaveReader::ReadFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    WaveHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WaveHeader));

    if (!file || file.gcount() != sizeof(WaveHeader)) {
        std::cerr << "Failed to read WAV header" << std::endl;
        return nullptr;
    }

    // Check RIFF header
    if (header.chunk_id != 0x46464952) {  // "RIFF"
        std::cerr << "Invalid RIFF header" << std::endl;
        return nullptr;
    }

    // Check WAVE format
    if (header.format != 0x45564157) {  // "WAVE"
        std::cerr << "Invalid WAVE format" << std::endl;
        return nullptr;
    }

    // Handle JUNK chunk if present
    if (header.subchunk1_id == 0x4B4E554A) {  // "JUNK"
        file.seekg(header.subchunk1_size, std::ios::cur);
        file.read(reinterpret_cast<char*>(&header.subchunk1_id), 4);
        file.read(reinterpret_cast<char*>(&header.subchunk1_size), 4);
        file.read(reinterpret_cast<char*>(&header.audio_format), 2);
        file.read(reinterpret_cast<char*>(&header.num_channels), 2);
        file.read(reinterpret_cast<char*>(&header.sample_rate), 4);
        file.read(reinterpret_cast<char*>(&header.byte_rate), 4);
        file.read(reinterpret_cast<char*>(&header.block_align), 2);
        file.read(reinterpret_cast<char*>(&header.bits_per_sample), 2);
        file.read(reinterpret_cast<char*>(&header.subchunk2_id), 4);
        file.read(reinterpret_cast<char*>(&header.subchunk2_size), 4);

        if (!file) {
            std::cerr << "Failed to read extended WAV header" << std::endl;
            return nullptr;
        }
    }

    // Check fmt chunk
    if (header.subchunk1_id != 0x20746d66) {  // "fmt "
        std::cerr << "Invalid fmt chunk" << std::endl;
        return nullptr;
    }

    // Check audio format (1 = PCM, 3 = IEEE float)
    if (header.audio_format != 1 && header.audio_format != 3) {
        std::cerr << "Unsupported audio format: " << header.audio_format << std::endl;
        return nullptr;
    }

    // Only support mono audio
    if (header.num_channels != 1) {
        std::cerr << "Only mono audio is supported. Channels: " << header.num_channels << std::endl;
        return nullptr;
    }

    // Check bits per sample
    if (header.bits_per_sample != 16 && header.bits_per_sample != 8 && header.bits_per_sample != 32) {
        std::cerr << "Unsupported bits per sample: " << header.bits_per_sample << std::endl;
        return nullptr;
    }

    // Handle extended format
    if (header.subchunk1_size == 18) {
        int16_t extra_size;
        file.read(reinterpret_cast<char*>(&extra_size), 2);
        if (!file || extra_size != 0) {
            std::cerr << "Extra size should be 0" << std::endl;
            return nullptr;
        }
    }

    // Find data chunk
    while (header.subchunk2_id != 0x61746164) {  // "data"
        file.seekg(header.subchunk2_size, std::ios::cur);
        file.read(reinterpret_cast<char*>(&header.subchunk2_id), 4);
        if (!file) break;
        file.read(reinterpret_cast<char*>(&header.subchunk2_size), 4);
        if (!file) break;
    }

    if (header.subchunk2_id != 0x61746164) {  // "data"
        std::cerr << "data chunk not found" << std::endl;
        return nullptr;
    }

    // Read samples
    if (header.bits_per_sample == 0) {
        std::cerr << "Invalid bits per sample: 0" << std::endl;
        return nullptr;
    }
    int32_t num_samples = header.subchunk2_size / (header.bits_per_sample / 8);
    std::vector<float> samples(num_samples);

    if (header.bits_per_sample == 16) {
        // Read and convert directly without extra allocation
        std::vector<int16_t> raw_samples(num_samples);
        file.read(reinterpret_cast<char*>(raw_samples.data()), num_samples * sizeof(int16_t));
        if (!file) {
            std::cerr << "Failed to read audio samples" << std::endl;
            return nullptr;
        }
        constexpr float scale = 1.0f / 32768.0f;
        for (int32_t i = 0; i < num_samples; i++) {
            samples[i] = raw_samples[i] * scale;
        }
    } else if (header.bits_per_sample == 8) {
        std::vector<uint8_t> raw_samples(num_samples);
        file.read(reinterpret_cast<char*>(raw_samples.data()), num_samples);
        if (!file) {
            std::cerr << "Failed to read audio samples" << std::endl;
            return nullptr;
        }
        constexpr float scale = 1.0f / 128.0f;
        for (int32_t i = 0; i < num_samples; i++) {
            samples[i] = (raw_samples[i] - 128) * scale;
        }
    } else if (header.bits_per_sample == 32) {
        if (header.audio_format == 3) {  // IEEE float
            file.read(reinterpret_cast<char*>(samples.data()), num_samples * sizeof(float));
            if (!file) {
                std::cerr << "Failed to read float audio samples" << std::endl;
                return nullptr;
            }
        } else {  // 32-bit PCM
            std::vector<int32_t> raw_samples(num_samples);
            file.read(reinterpret_cast<char*>(raw_samples.data()), num_samples * sizeof(int32_t));
            if (!file) {
                std::cerr << "Failed to read audio samples" << std::endl;
                return nullptr;
            }
            constexpr float scale = 1.0f / 2147483648.0f;
            for (int32_t i = 0; i < num_samples; i++) {
                samples[i] = raw_samples[i] * scale;
            }
        }
    }

    return std::make_unique<Wave>(std::move(samples), header.sample_rate);
}

} // namespace speaker_recognition