#include "asr_model.hpp"
#include "audio_processor.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>

ASRModel::ASRModel(const Config& config)
    : config_(config), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    initializeLanguageMaps();
}

ASRModel::~ASRModel() {
    cleanup();
}

bool ASRModel::initialize() {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ASRModel");
        
        if (!initializeSession()) {
            return false;
        }
        
        if (!loadConfig()) {
            std::cerr << "Warning: Could not load config file, using defaults" << std::endl;
        }
        
        // Initialize audio processor
        AudioProcessor::Config audio_config;
        audio_config.sample_rate = config_.sample_rate;
        audio_config.cmvn_file = config_.config_path; // Assuming CMVN is in config
        audio_processor_ = std::make_unique<AudioProcessor>(audio_config);
        if (!audio_processor_->initialize()) {
            std::cerr << "Failed to initialize audio processor" << std::endl;
            return false;
        }
        
        // Initialize tokenizer
        Tokenizer::Config tokenizer_config;
        tokenizer_config.vocab_file = config_.vocab_path;
        tokenizer_config.decoder_model_path = config_.decoder_path;
        tokenizer_ = std::make_unique<Tokenizer>(tokenizer_config);
        if (!tokenizer_->initialize()) {
            std::cerr << "Failed to initialize tokenizer" << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ASR model: " << e.what() << std::endl;
        return false;
    }
}

bool ASRModel::initializeSession() {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);  // 增加线程数
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);  // 并行执行模式
        
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names and shapes
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.reserve(num_input_nodes);
        input_shapes_.reserve(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.release());
            
            auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_shapes_.push_back(input_shape);
        }
        
        // Output names and shapes
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.reserve(num_output_nodes);
        output_shapes_.reserve(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.release());
            
            auto output_shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            output_shapes_.push_back(output_shape);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX session: " << e.what() << std::endl;
        return false;
    }
}

void ASRModel::cleanup() {
    session_.reset();
    env_.reset();
    audio_processor_.reset();
    tokenizer_.reset();
    
    for (auto name : input_names_) {
        delete[] name;
    }
    input_names_.clear();
    
    for (auto name : output_names_) {
        delete[] name;
    }
    output_names_.clear();
}

bool ASRModel::loadConfig() {
    // In a real implementation, you would parse YAML config file
    // For now, we'll use hardcoded values
    return true;
}

void ASRModel::initializeLanguageMaps() {
    language_dict_["auto"] = 0;
    language_dict_["zh"] = 3;
    language_dict_["en"] = 4;
    language_dict_["yue"] = 7;
    language_dict_["ja"] = 11;
    language_dict_["ko"] = 12;
    language_dict_["nospeech"] = 13;
    
    textnorm_dict_["withitn"] = 14;
    textnorm_dict_["woitn"] = 15;
}

std::string ASRModel::recognize(const std::vector<float>& audio) {
    return recognize(audio.data(), audio.size());
}

std::string ASRModel::recognize(const float* audio, size_t length) {
    if (!session_ || !audio_processor_ || !tokenizer_) {
        std::cerr << "ASR model not properly initialized" << std::endl;
        return "";
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Extract features
        auto feature_start = std::chrono::high_resolution_clock::now();
        std::vector<float> audio_vec(audio, audio + length);
        auto features = audio_processor_->extractFeatures(audio_vec);
        auto feature_end = std::chrono::high_resolution_clock::now();
        auto feature_time = std::chrono::duration<double>(feature_end - feature_start).count();
        
        // Flatten features for ONNX input
        auto flatten_start = std::chrono::high_resolution_clock::now();
        std::vector<float> flattened_features;
        size_t feature_dim = features.empty() ? 0 : features[0].size();
        size_t sequence_length = features.size();
        
        flattened_features.reserve(sequence_length * feature_dim);
        for (const auto& frame : features) {
            flattened_features.insert(flattened_features.end(), frame.begin(), frame.end());
        }
        auto flatten_end = std::chrono::high_resolution_clock::now();
        auto flatten_time = std::chrono::duration<double>(flatten_end - flatten_start).count();
        
        // Prepare input tensors
        std::vector<int64_t> feature_shape = {config_.batch_size, static_cast<int64_t>(sequence_length), static_cast<int64_t>(feature_dim)};
        std::vector<int64_t> length_shape = {config_.batch_size};
        std::vector<int64_t> language_shape = {config_.batch_size};
        std::vector<int64_t> textnorm_shape = {config_.batch_size};
        
        std::vector<int32_t> feat_length = {static_cast<int32_t>(sequence_length)};
        std::vector<int32_t> language_id = {getLanguageId(config_.language)};
        std::vector<int32_t> textnorm_id = {getTextnormId(config_.use_itn)};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, flattened_features.data(), flattened_features.size(), 
            feature_shape.data(), feature_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, feat_length.data(), feat_length.size(), 
            length_shape.data(), length_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, language_id.data(), language_id.size(), 
            language_shape.data(), language_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, textnorm_id.data(), textnorm_id.size(), 
            textnorm_shape.data(), textnorm_shape.size()));
        
        // Run inference
        auto inference_start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                          input_names_.data(), input_tensors.data(), input_tensors.size(),
                                          output_names_.data(), output_names_.size());
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_time = std::chrono::duration<double>(inference_end - inference_start).count();
        
        // Get logits and decode
        auto decode_start = std::chrono::high_resolution_clock::now();
        float* logits_data = output_tensors[0].GetTensorMutableData<float>();
        auto logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int seq_len = static_cast<int>(logits_shape[1]);
        int vocab_size = static_cast<int>(logits_shape[2]);
        
        std::vector<float> logits_vec(logits_data, logits_data + seq_len * vocab_size);
        auto token_ids = decodeCTC(logits_vec, seq_len);
        
        // Decode tokens to text
        std::string result = tokenizer_->decode(token_ids);
        result = postProcess(token_ids);
        auto decode_end = std::chrono::high_resolution_clock::now();
        auto decode_time = std::chrono::duration<double>(decode_end - decode_start).count();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        double audio_duration = static_cast<double>(length) / config_.sample_rate;
        double rtf = duration / audio_duration;
        
        std::cout << "=== Performance Breakdown ===" << std::endl;
        std::cout << "Feature extraction: " << feature_time << "s (" << (feature_time/duration*100) << "%)" << std::endl;
        std::cout << "Data flattening: " << flatten_time << "s (" << (flatten_time/duration*100) << "%)" << std::endl;
        std::cout << "ONNX inference: " << inference_time << "s (" << (inference_time/duration*100) << "%)" << std::endl;
        std::cout << "Token decoding: " << decode_time << "s (" << (decode_time/duration*100) << "%)" << std::endl;
        std::cout << "Total time: " << duration << "s, Audio duration: " << audio_duration 
                  << "s, RTF: " << rtf << std::endl;
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "ASR inference error: " << e.what() << std::endl;
        return "";
    }
}

std::vector<std::string> ASRModel::recognizeBatch(const std::vector<std::vector<float>>& audio_batch) {
    std::vector<std::string> results;
    results.reserve(audio_batch.size());
    
    for (const auto& audio : audio_batch) {
        results.push_back(recognize(audio));
    }
    
    return results;
}

std::vector<int> ASRModel::decodeCTC(const std::vector<float>& logits, int sequence_length) {
    std::vector<int> tokens;
    int vocab_size = logits.size() / sequence_length;
    
    int prev_token = -1;
    for (int t = 0; t < sequence_length; ++t) {
        // Find max probability token at time step t
        int max_token = 0;
        float max_prob = logits[t * vocab_size];
        
        for (int v = 1; v < vocab_size; ++v) {
            if (logits[t * vocab_size + v] > max_prob) {
                max_prob = logits[t * vocab_size + v];
                max_token = v;
            }
        }
        
        // CTC decoding: skip blank tokens and repeated tokens
        if (max_token != blank_id_ && max_token != prev_token) {
            tokens.push_back(max_token);
        }
        prev_token = max_token;
    }
    
    return tokens;
}

std::string ASRModel::postProcess(const std::vector<int>& token_ids) {
    // This would typically involve rich transcription post-processing
    // For now, just return the decoded text
    return tokenizer_->decode(token_ids);
}

int ASRModel::getLanguageId(const std::string& language) {
    auto it = language_dict_.find(language);
    return it != language_dict_.end() ? it->second : language_dict_["auto"];
}

int ASRModel::getTextnormId(bool use_itn) {
    return use_itn ? textnorm_dict_["withitn"] : textnorm_dict_["woitn"];
}
