#include "asr_realtime_api.hpp"
#include "ollama.hpp"  // 使用nlohmann::json
#include <curl/curl.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using json = nlohmann::json;

// Helper function to trim whitespace
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Base64 编码
static std::string base64Encode(const unsigned char* buffer, size_t length) {
    BIO *bio, *b64;
    BUF_MEM *buffer_ptr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL); // 不添加换行符
    BIO_write(bio, buffer, length);
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &buffer_ptr);

    std::string result(buffer_ptr->data, buffer_ptr->length);
    BIO_free_all(bio);

    return result;
}

// HMAC-SHA1 签名
static std::string hmacSha1(const std::string& key, const std::string& data) {
    unsigned char* digest;
    unsigned int digest_len;

    digest = HMAC(EVP_sha1(),
                  key.c_str(), key.length(),
                  reinterpret_cast<const unsigned char*>(data.c_str()), data.length(),
                  nullptr, &digest_len);

    return base64Encode(digest, digest_len);
}

// URL编码
static std::string urlEncode(const std::string& str) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (char c : str) {
        if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
        } else {
            escaped << std::uppercase;
            escaped << '%' << std::setw(2) << int((unsigned char)c);
            escaped << std::nouppercase;
        }
    }

    return escaped.str();
}

// CURL写入回调
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t total_size = size * nmemb;
    userp->append((char*)contents, total_size);
    return total_size;
}

// ==================== ASRRealtimeClient::Impl ====================
class ASRRealtimeClient::Impl {
public:
    Impl(const Config& config) : config_(config) {}

    std::string httpPost(const std::string& url, const std::string& body,
                        const std::vector<std::pair<std::string, std::string>>& headers) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            return "";
        }

        std::string response_string;
        struct curl_slist* header_list = nullptr;

        for (const auto& header : headers) {
            std::string header_str = header.first + ": " + header.second;
            header_list = curl_slist_append(header_list, header_str.c_str());
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, config_.timeout_seconds);

        CURLcode res = curl_easy_perform(curl);

        if (header_list) {
            curl_slist_free_all(header_list);
        }
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL POST error: " << curl_easy_strerror(res) << std::endl;
            return "";
        }

        return response_string;
    }

    std::string httpPostBinary(const std::string& url, const std::vector<uint8_t>& data,
                              const std::vector<std::pair<std::string, std::string>>& headers) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            return "";
        }

        std::string response_string;
        struct curl_slist* header_list = nullptr;

        for (const auto& header : headers) {
            std::string header_str = header.first + ": " + header.second;
            header_list = curl_slist_append(header_list, header_str.c_str());
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.data());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.size());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, config_.timeout_seconds);

        CURLcode res = curl_easy_perform(curl);

        if (header_list) {
            curl_slist_free_all(header_list);
        }
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL POST binary error: " << curl_easy_strerror(res) << std::endl;
            return "";
        }

        return response_string;
    }

    Config config_;
};

// ==================== ASRRealtimeClient ====================

ASRRealtimeClient::ASRRealtimeClient(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>(config)) {
    if (config_.load_from_env) {
        loadConfigFromEnv();
    }
}

ASRRealtimeClient::ASRRealtimeClient()
    : ASRRealtimeClient(Config()) {
}

ASRRealtimeClient::~ASRRealtimeClient() = default;

void ASRRealtimeClient::loadConfigFromEnv() {
    // 1. First try to load from .env file
    std::ifstream file(".env");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string value = trim(line.substr(eq_pos + 1));

                // Remove quotes if present
                if (value.size() >= 2 &&
                    ((value.front() == '"' && value.back() == '"') ||
                     (value.front() == '\'' && value.back() == '\''))) {
                    value = value.substr(1, value.size() - 2);
                }

                if (key == "ALIYUN_ACCESS_KEY_ID" && config_.access_key_id.empty()) {
                    config_.access_key_id = value;
                } else if (key == "ALIYUN_ACCESS_KEY_SECRET" && config_.access_key_secret.empty()) {
                    config_.access_key_secret = value;
                } else if (key == "ALIYUN_NLS_TOKEN" && config_.token.empty()) {
                    config_.token = value;
                } else if (key == "ALIYUN_ASR_APPKEY" && config_.appkey.empty()) {
                    config_.appkey = value;
                }
            }
        }
        file.close();
    }

    // 2. Then try environment variables (they override .env file if both exist)
    const char* env_access_key_id = std::getenv("ALIYUN_ACCESS_KEY_ID");
    if (env_access_key_id && config_.access_key_id.empty()) {
        config_.access_key_id = env_access_key_id;
    }

    const char* env_access_key_secret = std::getenv("ALIYUN_ACCESS_KEY_SECRET");
    if (env_access_key_secret && config_.access_key_secret.empty()) {
        config_.access_key_secret = env_access_key_secret;
    }

    const char* env_token = std::getenv("ALIYUN_NLS_TOKEN");
    if (env_token && config_.token.empty()) {
        config_.token = env_token;
    }

    const char* env_appkey = std::getenv("ALIYUN_ASR_APPKEY");
    if (env_appkey && config_.appkey.empty()) {
        config_.appkey = env_appkey;
    }

    bool has_accesskey = !config_.access_key_id.empty() && !config_.access_key_secret.empty();
    bool has_token = !config_.token.empty();

    if ((has_accesskey || has_token) && !config_.appkey.empty()) {
        std::cout << "Loaded Aliyun credentials from environment/config" << std::endl;
        if (has_token) {
            std::cout << "  Using direct token (24h validity)" << std::endl;
        }
    }
}

bool ASRRealtimeClient::initialize() {
    if (!isConfigured()) {
        last_error_ = "Missing required configuration. Please set:\n"
                     "  Either: ALIYUN_NLS_TOKEN (for quick test)\n"
                     "  Or: ALIYUN_ACCESS_KEY_ID + ALIYUN_ACCESS_KEY_SECRET\n"
                     "  And: ALIYUN_ASR_APPKEY";
        std::cerr << "Error: " << last_error_ << std::endl;
        return false;
    }

    std::cout << "Initializing Aliyun ASR Realtime Client..." << std::endl;
    std::cout << "  Region: " << config_.region << std::endl;
    std::cout << "  Sample Rate: " << config_.sample_rate << " Hz" << std::endl;
    std::cout << "  AppKey: " << config_.appkey << std::endl;

    // 如果使用AccessKey，获取Token
    if (!config_.token.empty()) {
        std::cout << "Using provided token (skipping token request)" << std::endl;
        cached_token_ = config_.token;
        token_expire_time_ = std::chrono::system_clock::now() +
                            std::chrono::hours(24);  // 假设24小时有效
    } else if (!refreshToken()) {
        return false;
    }

    std::cout << "ASRRealtimeClient initialized successfully" << std::endl;
    return true;
}

bool ASRRealtimeClient::isConfigured() const {
    bool has_accesskey = !config_.access_key_id.empty() && !config_.access_key_secret.empty();
    bool has_token = !config_.token.empty();
    return (has_accesskey || has_token) && !config_.appkey.empty();
}

bool ASRRealtimeClient::isTokenExpired() const {
    auto now = std::chrono::system_clock::now();
    return cached_token_.empty() || now >= token_expire_time_;
}

std::string ASRRealtimeClient::requestNewToken() {
    std::cout << "Requesting new token using AccessKey..." << std::endl;

    // 获取当前时间戳（ISO 8601格式）
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S")
                     << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    std::string timestamp = timestamp_stream.str();

    // 生成随机Nonce
    std::ostringstream nonce_stream;
    nonce_stream << std::time(nullptr) << rand();
    std::string nonce = nonce_stream.str();

    // 构造请求参数（按字母序排序，不包含Signature）
    std::ostringstream params;
    params << "AccessKeyId=" << config_.access_key_id
           << "&Action=CreateToken"
           << "&Format=JSON"
           << "&SignatureMethod=HMAC-SHA1"
           << "&SignatureNonce=" << nonce
           << "&SignatureVersion=1.0"
           << "&Timestamp=" << urlEncode(timestamp)
           << "&Version=2019-02-28";

    // 构造待签名字符串（按照阿里云签名规范）
    // StringToSign = HTTPMethod + "&" + percentEncode("/") + "&" + percentEncode(CanonicalizedQueryString)
    std::string canonicalized_query_string = params.str();
    std::ostringstream string_to_sign;
    string_to_sign << "POST&" << urlEncode("/") << "&" << urlEncode(canonicalized_query_string);

    std::string str_to_sign = string_to_sign.str();

    // 计算签名（使用 AccessKeySecret + "&" 作为密钥）
    std::string signing_key = config_.access_key_secret + "&";
    std::string signature = hmacSha1(signing_key, str_to_sign);

    std::cout << "Signature generated successfully" << std::endl;

    // 添加签名到参数中
    std::string full_params = params.str() + "&Signature=" + urlEncode(signature);
    std::string full_url = "https://nls-meta.cn-shanghai.aliyuncs.com/token?" + full_params;

    std::vector<std::pair<std::string, std::string>> headers = {
        {"Content-Type", "application/json"}
    };

    std::string response = impl_->httpPost(full_url, "", headers);

    if (response.empty()) {
        last_error_ = "Failed to request token: empty response";
        std::cerr << "Error: " << last_error_ << std::endl;
        return "";
    }

    try {
        json response_json = json::parse(response);

        if (response_json.contains("Token") && response_json["Token"].contains("Id")) {
            std::string token = response_json["Token"]["Id"];
            std::cout << "Token obtained successfully (expires in "
                      << config_.token_expire_seconds << " seconds)" << std::endl;
            return token;
        } else if (response_json.contains("Message")) {
            last_error_ = "Token request failed: " + response_json["Message"].get<std::string>();
            std::cerr << "Error: " << last_error_ << std::endl;
            return "";
        }
    } catch (const json::exception& e) {
        last_error_ = std::string("Failed to parse token response: ") + e.what();
        std::cerr << "Error: " << last_error_ << std::endl;
        std::cerr << "Response: " << response << std::endl;
        return "";
    }

    last_error_ = "Failed to extract token from response";
    return "";
}

bool ASRRealtimeClient::refreshToken() {
    std::string new_token = requestNewToken();
    if (new_token.empty()) {
        return false;
    }

    cached_token_ = new_token;
    token_expire_time_ = std::chrono::system_clock::now() +
                        std::chrono::seconds(config_.token_expire_seconds - 3600); // 提前1小时刷新

    return true;
}

std::string ASRRealtimeClient::getToken() {
    if (isTokenExpired()) {
        std::cout << "Token expired, refreshing..." << std::endl;
        if (!refreshToken()) {
            return "";
        }
    }
    return cached_token_;
}

std::vector<uint8_t> ASRRealtimeClient::convertToWav(const std::vector<float>& audio, int sample_rate) {
    std::vector<uint8_t> wav_data;

    // 转换为16bit PCM
    std::vector<int16_t> pcm_data(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        float sample = audio[i];
        // Clamp to [-1.0, 1.0]
        sample = std::max(-1.0f, std::min(1.0f, sample));
        pcm_data[i] = static_cast<int16_t>(sample * 32767.0f);
    }

    // WAV header
    uint32_t data_size = pcm_data.size() * sizeof(int16_t);
    uint32_t file_size = 36 + data_size;

    wav_data.resize(44 + data_size);

    // RIFF header
    memcpy(&wav_data[0], "RIFF", 4);
    memcpy(&wav_data[4], &file_size, 4);
    memcpy(&wav_data[8], "WAVE", 4);

    // fmt chunk
    memcpy(&wav_data[12], "fmt ", 4);
    uint32_t fmt_size = 16;
    memcpy(&wav_data[16], &fmt_size, 4);
    uint16_t audio_format = 1; // PCM
    memcpy(&wav_data[20], &audio_format, 2);
    uint16_t num_channels = 1;
    memcpy(&wav_data[22], &num_channels, 2);
    uint32_t sample_rate_u32 = sample_rate;
    memcpy(&wav_data[24], &sample_rate_u32, 4);
    uint32_t byte_rate = sample_rate * num_channels * sizeof(int16_t);
    memcpy(&wav_data[28], &byte_rate, 4);
    uint16_t block_align = num_channels * sizeof(int16_t);
    memcpy(&wav_data[32], &block_align, 2);
    uint16_t bits_per_sample = 16;
    memcpy(&wav_data[34], &bits_per_sample, 2);

    // data chunk
    memcpy(&wav_data[36], "data", 4);
    memcpy(&wav_data[40], &data_size, 4);
    memcpy(&wav_data[44], pcm_data.data(), data_size);

    return wav_data;
}

std::string ASRRealtimeClient::recognize(const std::vector<float>& audio) {
    return recognize(audio, config_.sample_rate);
}

std::string ASRRealtimeClient::recognize(const std::vector<float>& audio, int sample_rate) {
    if (!isConfigured()) {
        last_error_ = "ASRRealtimeClient is not configured";
        std::cerr << "Error: " << last_error_ << std::endl;
        return "";
    }

    // 获取Token
    std::string token = getToken();
    if (token.empty()) {
        last_error_ = "Failed to get valid token";
        std::cerr << "Error: " << last_error_ << std::endl;
        return "";
    }

    std::cout << "音频参数: " << audio.size() << " samples, " << sample_rate << " Hz, "
              << (float)audio.size() / sample_rate << " seconds" << std::endl;

    // 重采样到 16kHz（如果需要）
    std::vector<float> resampled_audio;
    int target_sample_rate = 16000;

    if (sample_rate != target_sample_rate) {
        std::cout << "重采样从 " << sample_rate << "Hz 到 " << target_sample_rate << "Hz..." << std::endl;
        auto resample_start = std::chrono::high_resolution_clock::now();

        // 对于 48kHz 到 16kHz，可以简单地每隔3个采样点取一个 (48/16 = 3)
        if (sample_rate == 48000 && target_sample_rate == 16000) {
            resampled_audio.reserve(audio.size() / 3);
            for (size_t i = 0; i < audio.size(); i += 3) {
                resampled_audio.push_back(audio[i]);
            }
        } else {
            // 对于其他比率，使用简单抽取
            double ratio = static_cast<double>(sample_rate) / target_sample_rate;
            size_t output_size = static_cast<size_t>(audio.size() / ratio);
            resampled_audio.reserve(output_size);

            for (size_t i = 0; i < output_size; ++i) {
                size_t src_idx = static_cast<size_t>(i * ratio);
                if (src_idx < audio.size()) {
                    resampled_audio.push_back(audio[src_idx]);
                }
            }
        }

        auto resample_end = std::chrono::high_resolution_clock::now();
        auto resample_time = std::chrono::duration<double>(resample_end - resample_start).count();
        std::cout << "重采样完成: " << resampled_audio.size() << " 采样点 (耗时 "
                  << std::fixed << std::setprecision(3) << resample_time << "s)" << std::endl;
    } else {
        resampled_audio = audio;
    }

    // 转换为WAV格式（使用16kHz）在我的代码里面不需要这个，直接使用的就是音频流，接口可以拿去自己改。
    std::vector<uint8_t> wav_data = convertToWav(resampled_audio, target_sample_rate);

    // 构造请求URL（使用16kHz）
    std::ostringstream url;
    url << "https://nls-gateway-" << config_.region << ".aliyuncs.com/stream/v1/asr"
        << "?appkey=" << config_.appkey
        << "&format=pcm"
        << "&sample_rate=" << target_sample_rate;

    if (config_.enable_punctuation) {
        url << "&enable_punctuation_prediction=true";
    }
    if (config_.enable_itn) {
        url << "&enable_inverse_text_normalization=true";
    }

    // 设置请求头
    std::vector<std::pair<std::string, std::string>> headers = {
        {"X-NLS-Token", token},
        {"Content-Type", "application/octet-stream"}
    };

    std::cout << "Recognizing audio (" << audio.size() << " samples, "
              << (float)audio.size() / sample_rate << "s)..." << std::endl;

    // 记录发送时间
    auto send_start = std::chrono::high_resolution_clock::now();
    auto send_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "发送请求时间: " << std::put_time(std::localtime(&send_time_t), "%H:%M:%S");
    auto send_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()) % 1000;
    std::cout << "." << std::setfill('0') << std::setw(3) << send_ms.count() << std::endl;

    // 发送请求
    std::string response = impl_->httpPostBinary(url.str(), wav_data, headers);

    // 记录接收时间
    auto recv_end = std::chrono::high_resolution_clock::now();
    auto recv_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "接收响应时间: " << std::put_time(std::localtime(&recv_time_t), "%H:%M:%S");
    auto recv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()) % 1000;
    std::cout << "." << std::setfill('0') << std::setw(3) << recv_ms.count() << std::endl;

    // 计算网络延迟
    auto network_latency = std::chrono::duration<double, std::milli>(recv_end - send_start).count();
    std::cout << "网络往返延迟: " << network_latency << " ms" << std::endl;

    if (response.empty()) {
        last_error_ = "Failed to get recognition result: empty response";
        std::cerr << "Error: " << last_error_ << std::endl;
        return "";
    }

    std::cout << "Response: " << response << std::endl;

    // 解析结果
    try {
        json response_json = json::parse(response);

        // 检查状态码
        if (response_json.contains("status")) {
            int status = response_json["status"];
            if (status == 20000000) {
                // 成功
                if (response_json.contains("result")) {
                    std::string result = response_json["result"];
                    std::cout << "Recognition result: " << result << std::endl;
                    return result;
                }
            } else {
                // 错误
                std::string message = response_json.value("message", "Unknown error");
                last_error_ = "ASR API error (status=" + std::to_string(status) + "): " + message;
                std::cerr << "Error: " << last_error_ << std::endl;
                return "";
            }
        }
    } catch (const json::exception& e) {
        last_error_ = std::string("Failed to parse recognition result: ") + e.what();
        std::cerr << "Error: " << last_error_ << std::endl;
        return "";
    }

    last_error_ = "No result found in response";
    return "";
}

std::string ASRRealtimeClient::httpPost(const std::string& url, const std::string& body,
                                       const std::vector<std::pair<std::string, std::string>>& headers) {
    return impl_->httpPost(url, body, headers);
}

std::string ASRRealtimeClient::httpPostBinary(const std::string& url, const std::vector<uint8_t>& data,
                                             const std::vector<std::pair<std::string, std::string>>& headers) {
    return impl_->httpPostBinary(url, data, headers);
}
