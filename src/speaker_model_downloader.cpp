#include "speaker_model_downloader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <curl/curl.h>
#include <cstdlib>
#include <sys/stat.h>

// Static constants
const std::string SRModelDownloader::AR_MODEL_NAME = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

SRModelDownloader::SRModelDownloader() : config_(Config{}) {
    cache_dir_expanded_ = expandPath(config_.cache_dir);
}

SRModelDownloader::SRModelDownloader(const Config& config) : config_(config) {
    cache_dir_expanded_ = expandPath(config_.cache_dir);
}

SRModelDownloader::~SRModelDownloader() {
}

bool SRModelDownloader::ensureModelsExist() {
    if (!createCacheDirectory()) {
        return false;
    }
    
    // Check if required models exist
    std::vector<std::string> required_models = {
        AR_MODEL_NAME
    };
    
    bool all_exist = true;
    for (const std::string& model : required_models) {
        if (!isModelAvailable(model)) {
            std::cout << "Model " << model << " not found" << std::endl;
            all_exist = false;
        }
    }
    
    if (!all_exist) {
        std::cout << "Models not found, downloading..." << std::endl;
        return downloadModels();
    }
    
    std::cout << "All required models are available" << std::endl;
    return true;
}

bool SRModelDownloader::downloadModels(ProgressCallback progress_cb) {
    std::string archive_path = cache_dir_expanded_ + "/speaker-recognition.tar.gz";
    
    if (!downloadFile(config_.model_url, archive_path, progress_cb)) {
        return false;
    }
    
    if (!extractModels(archive_path)) {
        return false;
    }
    
    // Clean up archive
    std::filesystem::remove(archive_path);
    
    return true;
}

bool SRModelDownloader::downloadFile(const std::string& url, const std::string& output_path, 
                                  ProgressCallback progress_cb) {
    CURL* curl;
    CURLcode res;
    
    curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize curl" << std::endl;
        return false;
    }
    
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
    
    DownloadData download_data;
    download_data.progress_cb = progress_cb;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // 不验证对方证书
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L); 
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "SPEAKER_RECOGNITION/1.0");
    
    if (progress_cb) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &download_data);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }
    
    res = curl_easy_perform(curl);
    
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    
    curl_easy_cleanup(curl);
    file.close();
    
    if (res != CURLE_OK) {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        std::filesystem::remove(output_path);
        return false;
    }
    
    if (response_code != 200) {
        std::cerr << "Download failed with HTTP code: " << response_code << std::endl;
        std::filesystem::remove(output_path);
        return false;
    }
    
    std::cout << "Download completed: " << output_path << std::endl;
    return true;
}

bool SRModelDownloader::extractModels(const std::string& archive_path) {
    // First extract to a temporary directory to check structure
    std::string temp_dir = cache_dir_expanded_ + "/temp_extract";
    std::filesystem::create_directories(temp_dir);
    
    // Extract to temp directory
    std::string cmd = "tar -xzf \"" + archive_path + "\" -C \"" + temp_dir + "\"";
    
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Failed to extract archive: " << archive_path << std::endl;
        std::filesystem::remove_all(temp_dir);
        return false;
    }
    
    // Check if extracted files are in a subdirectory
    std::string speaker_recognition_subdir = temp_dir + "/speaker-recognition";
    if (std::filesystem::exists(speaker_recognition_subdir) && std::filesystem::is_directory(speaker_recognition_subdir)) {
        // Move files from sensevoice subdirectory to cache root
        for (const auto& entry : std::filesystem::directory_iterator(speaker_recognition_subdir)) {
            std::filesystem::rename(entry.path(), cache_dir_expanded_ + "/" + entry.path().filename().string());
        }
    } else {
        // Move all files from temp to cache root
        for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
            std::filesystem::rename(entry.path(), cache_dir_expanded_ + "/" + entry.path().filename().string());
        }
    }
    
    // Clean up temp directory
    std::filesystem::remove_all(temp_dir);
    
    std::cout << "Models extracted successfully" << std::endl;
    return true;
}

std::string SRModelDownloader::getModelPath(const std::string& model_name) const {
    return cache_dir_expanded_ + "/" + model_name;
}

bool SRModelDownloader::isModelAvailable(const std::string& model_name) const {
    return fileExists(getModelPath(model_name));
}

bool SRModelDownloader::createCacheDirectory() {
    try {
        std::filesystem::create_directories(cache_dir_expanded_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create cache directory: " << e.what() << std::endl;
        return false;
    }
}

std::string SRModelDownloader::expandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }
    
    const char* home = std::getenv("HOME");
    if (!home) {
        home = std::getenv("USERPROFILE"); // Windows
    }
    
    if (!home) {
        return path; // Return as-is if no home directory found
    }
    
    return std::string(home) + path.substr(1);
}

bool SRModelDownloader::fileExists(const std::string& path) const {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

size_t SRModelDownloader::getFileSize(const std::string& path) const {
    try {
        return std::filesystem::file_size(path);
    } catch (const std::exception&) {
        return 0;
    }
}

size_t SRModelDownloader::writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    size_t total_size = size * nmemb;
    
    file->write(static_cast<const char*>(contents), total_size);
    return total_size;
}

int SRModelDownloader::progressCallback(void* clientp, double dltotal, double dlnow, 
                                     double ultotal, double ulnow) {
    DownloadData* data = static_cast<DownloadData*>(clientp);
    
    if (data->progress_cb && dltotal > 0) {
        double progress = dlnow / dltotal;
        data->progress_cb(progress);
    }
    
    return 0; // Continue download
}

bool SRModelDownloader::verifyChecksum(const std::string& file_path, 
                                    const std::string& expected_checksum) {
    if (!config_.verify_checksum || expected_checksum.empty()) {
        return true;
    }
    
    std::string actual_checksum = calculateSHA256(file_path);
    return actual_checksum == expected_checksum;
}

std::string SRModelDownloader::calculateSHA256(const std::string& file_path) {
    // Simple implementation using system sha256sum command
    std::string cmd = "sha256sum \"" + file_path + "\" | cut -d' ' -f1";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }
    
    char buffer[128];
    std::string result;
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr) {
            result += buffer;
        }
    }
    pclose(pipe);
    
    // Remove newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    
    return result;
}