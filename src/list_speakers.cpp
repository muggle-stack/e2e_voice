#include "speaker_recognition.hpp"
#include "speaker_model_downloader.hpp"
#include <iostream>

using namespace speaker_recognition;

int main(int argc, char* argv[]) {
    std::string database_file = "speakers.db";

    if (argc > 1) {
        database_file = argv[1];
    }

    // Initialize model downloader
    SRModelDownloader downloader;
    if (!downloader.ensureModelsExist()) {
        std::cerr << "Failed to ensure models exist" << std::endl;
        return 1;
    }

    // Create embedder to get dimension
    EmbedderConfig config;
    config.model_path = downloader.getModelPath(SRModelDownloader::AR_MODEL_NAME);
    config.num_threads = 1;
    config.debug = false;

    auto embedder = SpeakerEmbedder::Create(config);
    if (!embedder) {
        std::cerr << "Failed to create embedder" << std::endl;
        return 1;
    }

    // Create manager
    auto manager = SpeakerManager::Create(embedder->GetEmbeddingDimension());
    if (!manager) {
        std::cerr << "Failed to create manager" << std::endl;
        return 1;
    }

    // Load database
    std::cout << "Loading database: " << database_file << std::endl;
    if (!manager->LoadDatabase(database_file)) {
        std::cerr << "Failed to load database or database doesn't exist" << std::endl;
        return 1;
    }

    // List speakers
    auto speakers = manager->GetAllSpeakers();
    std::cout << "Total speakers: " << speakers.size() << std::endl;
    std::cout << "Registered speakers:" << std::endl;
    for (const auto& speaker : speakers) {
        std::cout << "  - " << speaker << std::endl;
    }

    return 0;
}