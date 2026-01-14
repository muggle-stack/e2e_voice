#include "tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <unistd.h>  // for dup, dup2, close
#include <fcntl.h>   // for open

Tokenizer::Tokenizer(const Config& config)
    : config_(config), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

Tokenizer::~Tokenizer() {
    cleanup();
}

bool Tokenizer::initialize() {
    try {
        // Temporarily suppress stderr to avoid ONNX schema warnings
        int stderr_fd = dup(STDERR_FILENO);
        int devnull_fd = open("/dev/null", O_WRONLY);
        dup2(devnull_fd, STDERR_FILENO);
        
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Tokenizer");
        
        // Restore stderr
        dup2(stderr_fd, STDERR_FILENO);
        close(stderr_fd);
        close(devnull_fd);
        
        if (!loadVocabulary()) {
            std::cerr << "Failed to load vocabulary" << std::endl;
            return false;
        }
        
        if (!config_.decoder_model_path.empty()) {
            if (!initializeDecoder()) {
                // Note: ONNX decoder initialization failed, but this is not critical
                // The system will fall back to simple text decoding which works fine
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize tokenizer: " << e.what() << std::endl;
        return false;
    }
}

bool Tokenizer::loadVocabulary() {
    if (config_.vocab_file.empty()) {
        std::cerr << "No vocabulary file specified" << std::endl;
        return false;
    }
    
    std::ifstream file(config_.vocab_file);
    if (!file.is_open()) {
        std::cerr << "Cannot open vocabulary file: " << config_.vocab_file << std::endl;
        return false;
    }
    
    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Parse format: "token\tscore"
            size_t tab_pos = line.find('\t');
            std::string token;
            if (tab_pos != std::string::npos) {
                token = line.substr(0, tab_pos);
            } else {
                token = line;
            }
            
            id_to_token_[id] = token;
            token_to_id_[token] = id;
            id++;
        }
    }
    
    vocab_size_ = id_to_token_.size();
    std::cout << "Loaded vocabulary with " << vocab_size_ << " tokens" << std::endl;
    return true;
}

bool Tokenizer::initializeDecoder() {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        // Load ONNX extensions if specified
        if (!config_.ort_extensions_path.empty()) {
            session_options.RegisterCustomOpsLibrary(config_.ort_extensions_path.c_str());
        }
        
        decoder_session_ = std::make_unique<Ort::Session>(*env_, config_.decoder_model_path.c_str(), session_options);
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_input_nodes = decoder_session_->GetInputCount();
        input_names_.reserve(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = decoder_session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.release());
        }
        
        size_t num_output_nodes = decoder_session_->GetOutputCount();
        output_names_.reserve(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = decoder_session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.release());
        }
        
        std::cout << "ONNX decoder initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        // Silently fail - decoder is optional
        return false;
    }
}

void Tokenizer::cleanup() {
    decoder_session_.reset();
    env_.reset();
    
    for (auto name : input_names_) {
        delete[] name;
    }
    input_names_.clear();
    
    for (auto name : output_names_) {
        delete[] name;
    }
    output_names_.clear();
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) {
    if (decoder_session_) {
        // Use ONNX decoder if available
        try {
            // Convert token IDs to tensor
            std::vector<int64_t> token_shape = {1, static_cast<int64_t>(token_ids.size())};
            std::vector<int64_t> token_ids_int64(token_ids.begin(), token_ids.end());
            
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info_, token_ids_int64.data(), token_ids_int64.size(),
                token_shape.data(), token_shape.size()));
            
            auto output_tensors = decoder_session_->Run(Ort::RunOptions{nullptr},
                                                      input_names_.data(), input_tensors.data(), input_tensors.size(),
                                                      output_names_.data(), output_names_.size());
            
            // Extract decoded text (implementation depends on decoder model output format)
            // For now, fall back to simple decoding
            
        } catch (const std::exception& e) {
            std::cerr << "ONNX decoder error: " << e.what() << ", falling back to simple decode" << std::endl;
        }
    }
    
    // Simple decoding
    std::vector<std::string> tokens;
    for (int id : token_ids) {
        std::string token = idToToken(id);
        if (!token.empty() && token != "<blank>") {
            tokens.push_back(token);
        }
    }
    
    std::string result = joinTokens(tokens);
    return postProcessText(result);
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    // Simple character-level encoding for Chinese text
    std::vector<int> token_ids;
    
    // Convert UTF-8 string to individual characters
    std::vector<std::string> chars = splitByDelimiters(text);
    
    for (const std::string& ch : chars) {
        int id = tokenToId(ch);
        token_ids.push_back(id);
    }
    
    return token_ids;
}

std::string Tokenizer::idToToken(int id) const {
    auto it = id_to_token_.find(id);
    return it != id_to_token_.end() ? it->second : "<unk>";
}

int Tokenizer::tokenToId(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : unk_token_id_;
}

std::string Tokenizer::postProcessText(const std::string& text) {
    std::string result = text;

    // Remove SenseVoice special tokens and formatting
    result = std::regex_replace(result, std::regex("<\\|[^|]*\\|>"), "");  // Remove <|zh|>, <|NEUTRAL|>, etc.
    result = std::regex_replace(result, std::regex("-?\\d+\\.\\d+"), "");  // Remove scores like -20.3711
    // Note: Do NOT remove standalone numbers - they are valid ASR output (e.g., "123", "2024")
    result = std::regex_replace(result, std::regex("\\?\\s*\\?"), "");    // Remove ? ? patterns
    result = std::regex_replace(result, std::regex("▁"), " ");            // Replace SentencePiece underscores

    // Remove extra spaces and clean up
    result = std::regex_replace(result, std::regex("\\s+"), " ");

    // Remove leading/trailing spaces
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    if (!result.empty()) {
        size_t last = result.find_last_not_of(" \t\n\r");
        if (last != std::string::npos) {
            result.erase(last + 1);
        }
    }

    // If result is only punctuation (no actual speech content), return empty
    // This handles the case when SenseVoice outputs just "。" for silence/noise
    if (!result.empty()) {
        bool has_content = false;
        size_t i = 0;
        while (i < result.length()) {
            unsigned char c = static_cast<unsigned char>(result[i]);
            size_t char_len = 1;
            if (c >= 0xF0) char_len = 4;
            else if (c >= 0xE0) char_len = 3;
            else if (c >= 0xC0) char_len = 2;

            if (i + char_len <= result.length()) {
                std::string ch = result.substr(i, char_len);
                // Check if this is NOT punctuation/space
                if (ch != "。" && ch != "，" && ch != "？" && ch != "！" &&
                    ch != "." && ch != "," && ch != "?" && ch != "!" &&
                    ch != " " && ch != "、" && ch != "；" && ch != "：" &&
                    ch != "\"" && ch != "'" && ch != """ && ch != """ &&
                    ch != "'" && ch != "'" && ch != "…" && ch != "—" &&
                    ch != "\n" && ch != "\r" && ch != "\t") {
                    has_content = true;
                    break;
                }
            }
            i += char_len;
        }
        if (!has_content) {
            return "";  // Only punctuation, treat as no speech
        }
    }

    return result;
}

std::vector<std::string> Tokenizer::splitByDelimiters(const std::string& text) {
    std::vector<std::string> result;
    
    // Simple UTF-8 character splitting
    size_t i = 0;
    while (i < text.length()) {
        size_t char_len = 1;
        
        // Determine UTF-8 character length
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c >= 0xC0) {
            if (c >= 0xF0) char_len = 4;
            else if (c >= 0xE0) char_len = 3;
            else if (c >= 0xC0) char_len = 2;
        }
        
        if (i + char_len <= text.length()) {
            std::string ch = text.substr(i, char_len);
            if (!ch.empty() && ch != " " && ch != "\t" && ch != "\n") {
                result.push_back(ch);
            }
        }
        
        i += char_len;
    }
    
    return result;
}

std::string Tokenizer::joinTokens(const std::vector<std::string>& tokens) {
    if (tokens.empty()) {
        return "";
    }
    
    std::string result;
    for (const std::string& token : tokens) {
        result += token;
    }
    
    return result;
}