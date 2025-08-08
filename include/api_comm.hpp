#ifndef API_COMM_HPP
#define API_COMM_HPP

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <sstream>
#include "ollama.hpp"  // Use the json library from ollama.hpp

namespace api_comm {

using json = nlohmann::json;

struct Message {
    std::string role;
    std::string content;
    
    Message(const std::string& r, const std::string& c) : role(r), content(c) {}
};

struct Response {
    std::string content;
    std::string model;
    std::string id;
    json raw_json;
    
    std::string as_simple_string() const { return content; }
    json as_json() const { return raw_json; }
};

struct Options {
    float temperature = 0.7f;
    int max_tokens = 100;
    bool stream = true;
    float top_p = 1.0f;
    int top_k = 40;
    
    json to_json() const {
        json j;
        j["temperature"] = temperature;
        j["max_tokens"] = max_tokens;
        j["stream"] = stream;
        j["top_p"] = top_p;
        j["top_k"] = top_k;
        return j;
    }
};

class ApiClient {
public:
    ApiClient();
    ~ApiClient();
    
    // Set API configuration
    void setApiKey(const std::string& key);
    void setApiUrl(const std::string& url);
    
    // Load configuration from .env file
    bool loadConfigFromEnv(const std::string& env_file = ".env");
    
    // Generate response with streaming callback
    bool generate(const std::string& model, 
                 const std::string& prompt,
                 std::function<bool(const Response&)> stream_callback,
                 const Options& options = Options());
    
    // Generate response with conversation history
    bool generate(const std::string& model,
                 const std::vector<Message>& messages,
                 std::function<bool(const Response&)> stream_callback,
                 const Options& options = Options());
    
    // Check if API is configured
    bool isConfigured() const;
    
    // Get current API provider name (for display)
    std::string getApiProvider() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

// Convenience functions for singleton usage
inline ApiClient& getClient() {
    static ApiClient client;
    return client;
}

inline void setApiKey(const std::string& key) {
    getClient().setApiKey(key);
}

inline void setApiUrl(const std::string& url) {
    getClient().setApiUrl(url);
}

inline bool loadConfigFromEnv(const std::string& env_file = ".env") {
    return getClient().loadConfigFromEnv(env_file);
}

inline bool generate(const std::string& model,
                    const std::string& prompt,
                    std::function<bool(const Response&)> stream_callback,
                    const Options& options = Options()) {
    return getClient().generate(model, prompt, stream_callback, options);
}

inline bool generate(const std::string& model,
                    const std::vector<Message>& messages,
                    std::function<bool(const Response&)> stream_callback,
                    const Options& options = Options()) {
    return getClient().generate(model, messages, stream_callback, options);
}

} // namespace api_comm

#endif // API_COMM_HPP