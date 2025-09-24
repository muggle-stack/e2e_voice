#pragma once

#include <string>
#include <functional>

class SRModelDownloader {
public:
    using ProgressCallback = std::function<void(double progress)>;
    
    struct Config {
        std::string cache_dir = "~/.cache/speaker-recognition";
        std::string model_url = "https://archive.spacemit.com/spacemit-ai/openwebui/speaker-recognition.tar.gz";
        bool verify_checksum = false;
        std::string expected_checksum = "";
    };

    SRModelDownloader();
    explicit SRModelDownloader(const Config& config);
    ~SRModelDownloader();

    bool ensureModelsExist();
    bool downloadModels(ProgressCallback progress_cb = nullptr);
    bool extractModels(const std::string& archive_path);
    
    std::string getModelPath(const std::string& model_name) const;
    bool isModelAvailable(const std::string& model_name) const;
    
    // Model file names
    static const std::string ASR_MODEL_NAME;
    static const std::string ASR_MODEL_QUANT_NAME;
    static const std::string VAD_MODEL_NAME;
    static const std::string CONFIG_NAME;
    static const std::string VOCAB_NAME;
    static const std::string CMVN_NAME;
    static const std::string DECODER_NAME;
    static const std::string AR_MODEL_NAME;

private:
    Config config_;
    std::string cache_dir_expanded_;
    
    bool createCacheDirectory();
    std::string expandPath(const std::string& path);
    bool fileExists(const std::string& path) const;
    size_t getFileSize(const std::string& path) const;
    
    // cURL callbacks
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp);
    static int progressCallback(void* clientp, double dltotal, double dlnow, 
                               double ultotal, double ulnow);
    
    struct DownloadData {
        std::string* buffer;
        ProgressCallback progress_cb;
    };
    
    bool downloadFile(const std::string& url, const std::string& output_path,
                     ProgressCallback progress_cb);
    bool verifyChecksum(const std::string& file_path, const std::string& expected_checksum);
    std::string calculateSHA256(const std::string& file_path);
};