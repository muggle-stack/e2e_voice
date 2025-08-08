#include "api_comm.hpp"
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace api_comm {

// Helper function to trim whitespace
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Helper function for cURL write callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

class ApiClient::Impl {
public:
    std::string api_key;
    std::string api_url;
    mutable std::mutex mutex;
    
    Impl() {
        // Try to load API key from environment variable
        const char* env_key = std::getenv("API_KEY");
        if (env_key) {
            api_key = env_key;
        }
        
        // Try to load API URL from environment variable
        const char* env_url = std::getenv("API_URL");
        if (env_url) {
            api_url = env_url;
        } else {
            // Default to OpenAI API
            api_url = "https://api.openai.com/v1/chat/completions";
        }
    }
    
    std::string getProviderName() const {
        if (api_url.find("deepseek.com") != std::string::npos) {
            return "DeepSeek";
        } else if (api_url.find("openai.com") != std::string::npos) {
            return "OpenAI";
        } else if (api_url.find("anthropic.com") != std::string::npos) {
            return "Anthropic";
        } else if (api_url.find("moonshot.cn") != std::string::npos) {
            return "Moonshot (Kimi)";
        } else if (api_url.find("dashscope.aliyuncs.com") != std::string::npos) {
            return "Qwen";
        } else if (api_url.find("localhost") != std::string::npos || 
                   api_url.find("127.0.0.1") != std::string::npos) {
            return "Local API";
        }
        return "Custom API";
    }
    
    bool parseStreamChunk(const std::string& chunk, Response& response) {
        // Handle SSE format: data: {...}
        if (chunk.empty() || chunk == "data: [DONE]") {
            return false;
        }
        
        size_t data_pos = chunk.find("data: ");
        if (data_pos == std::string::npos) {
            return false;
        }
        
        std::string json_str = chunk.substr(data_pos + 6);
        json_str = trim(json_str);
        
        if (json_str == "[DONE]") {
            return false;
        }
        
        try {
            json j = json::parse(json_str);
            
            // Extract content from streaming response
            if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                auto& choice = j["choices"][0];
                if (choice.contains("delta") && choice["delta"].contains("content")) {
                    response.content = choice["delta"]["content"];
                    response.raw_json = j;
                    
                    if (j.contains("model")) {
                        response.model = j["model"];
                    }
                    if (j.contains("id")) {
                        response.id = j["id"];
                    }
                    
                    return true;
                }
            }
        } catch (const json::parse_error& e) {
            std::cerr << "[API] JSON parse error: " << e.what() << " for chunk: " << json_str << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[API] Exception: " << e.what() << std::endl;
        }
        
        return false;
    }
    
    struct StreamData {
        std::function<bool(const Response&)> callback;
        std::string buffer;
        Impl* impl;
        bool done = false;
    };
    
    static size_t streamWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        auto* data = static_cast<StreamData*>(userp);
        size_t total_size = size * nmemb;
        data->buffer.append(static_cast<char*>(contents), total_size);
        
        // Process complete lines
        size_t pos = 0;
        while ((pos = data->buffer.find('\n')) != std::string::npos) {
            std::string line = data->buffer.substr(0, pos);
            data->buffer.erase(0, pos + 1);
            
            if (!line.empty()) {
                Response response;
                if (data->impl->parseStreamChunk(line, response)) {
                    // Call user callback with parsed response
                    bool continue_stream = data->callback(response);
                    if (!continue_stream) {
                        data->done = true;
                        return 0; // Stop the transfer
                    }
                } else if (line.find("[DONE]") != std::string::npos) {
                    // Stream finished
                    Response final_response;
                    final_response.raw_json = json{{"done", true}};
                    data->callback(final_response);
                    data->done = true;
                }
            }
        }
        
        return total_size;
    }
};

ApiClient::ApiClient() : impl(std::make_unique<Impl>()) {
    // Initialize cURL globally (should be done once per application)
    static bool curl_initialized = false;
    if (!curl_initialized) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_initialized = true;
    }
}

ApiClient::~ApiClient() = default;

void ApiClient::setApiKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->api_key = key;
}

void ApiClient::setApiUrl(const std::string& url) {
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->api_url = url;
}

bool ApiClient::loadConfigFromEnv(const std::string& env_file) {
    std::ifstream file(env_file);
    if (!file.is_open()) {
        // Try from current directory
        file.open("./" + env_file);
        if (!file.is_open()) {
            return false;
        }
    }
    
    bool found_key = false;
    bool found_url = false;
    
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
            
            if (key == "API_KEY" || key == "DEEPSEEK_API_KEY" || key == "OPENAI_API_KEY") {
                setApiKey(value);
                found_key = true;
            } else if (key == "API_URL") {
                setApiUrl(value);
                found_url = true;
            }
        }
    }
    
    return found_key || found_url;
}

bool ApiClient::isConfigured() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    return !impl->api_key.empty() && !impl->api_url.empty();
}

std::string ApiClient::getApiProvider() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    return impl->getProviderName();
}

bool ApiClient::generate(const std::string& model,
                        const std::string& prompt,
                        std::function<bool(const Response&)> stream_callback,
                        const Options& options) {
    std::vector<Message> messages;
    messages.emplace_back("system", "你是一个乐于助人的智能助手，请用中文回答用户的问题。");
    messages.emplace_back("user", prompt);
    return generate(model, messages, stream_callback, options);
}

bool ApiClient::generate(const std::string& model,
                        const std::vector<Message>& messages,
                        std::function<bool(const Response&)> stream_callback,
                        const Options& options) {
    std::lock_guard<std::mutex> lock(impl->mutex);
    
    if (impl->api_key.empty()) {
        std::cerr << "[API] API key not set" << std::endl;
        return false;
    }
    
    if (impl->api_url.empty()) {
        std::cerr << "[API] API URL not set" << std::endl;
        return false;
    }
    
    // Prepare request JSON
    json request;
    request["model"] = model;
    request["stream"] = options.stream;
    request["temperature"] = options.temperature;
    request["max_tokens"] = options.max_tokens;
    request["top_p"] = options.top_p;
    
    // Convert messages
    json messages_json = json::array();
    for (const auto& msg : messages) {
        messages_json.push_back({{"role", msg.role}, {"content", msg.content}});
    }
    request["messages"] = messages_json;
    
    std::string request_body = request.dump();
    
    // Initialize cURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[API] Failed to initialize cURL" << std::endl;
        return false;
    }
    
    // Prepare headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_header = "Authorization: Bearer " + impl->api_key;
    headers = curl_slist_append(headers, auth_header.c_str());
    
    // Set up cURL options
    curl_easy_setopt(curl, CURLOPT_URL, impl->api_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request_body.length());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    // Check for proxy settings
    const char* https_proxy = std::getenv("https_proxy");
    if (!https_proxy) https_proxy = std::getenv("HTTPS_PROXY");
    if (https_proxy) {
        curl_easy_setopt(curl, CURLOPT_PROXY, https_proxy);
        // std::cerr << "[API] Using proxy: " << https_proxy << std::endl;
    }
    
    // Debug logging (commented out for production)
    // std::cerr << "[API] URL: " << impl->api_url << std::endl;
    // std::cerr << "[API] Request: " << request_body << std::endl;
    // std::cerr << "[API] API Key (first 10 chars): " << impl->api_key.substr(0, 10) << "..." << std::endl;
    
    bool success = false;
    
    if (options.stream) {
        // Streaming mode
        Impl::StreamData stream_data;
        stream_data.callback = stream_callback;
        stream_data.impl = impl.get();
        
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, Impl::streamWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream_data);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK && res != CURLE_WRITE_ERROR) {
            std::cerr << "[API] cURL error: " << curl_easy_strerror(res) << std::endl;
            
            // Get HTTP response code
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            std::cerr << "[API] HTTP response code: " << response_code << std::endl;
        } else {
            // Get HTTP response code
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            // std::cerr << "[API] HTTP response code: " << response_code << std::endl;
            success = stream_data.done || response_code == 200;
            // std::cerr << "[API] Stream completed, success: " << success << std::endl;
        }
    } else {
        // Non-streaming mode
        std::string response_string;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        
        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            try {
                json response_json = json::parse(response_string);
                Response response;
                
                if (response_json.contains("choices") && 
                    response_json["choices"].is_array() && 
                    !response_json["choices"].empty()) {
                    auto& choice = response_json["choices"][0];
                    if (choice.contains("message") && choice["message"].contains("content")) {
                        response.content = choice["message"]["content"];
                        response.raw_json = response_json;
                        response.model = response_json.value("model", "");
                        response.id = response_json.value("id", "");
                        
                        stream_callback(response);
                        success = true;
                    }
                }
            } catch (const json::parse_error& e) {
                std::cerr << "[API] JSON parse error: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "[API] cURL error: " << curl_easy_strerror(res) << std::endl;
        }
    }
    
    // Cleanup
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return success;
}

} // namespace api_comm