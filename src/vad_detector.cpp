#include "vad_detector.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unistd.h>  // for dup, dup2, close
#include <fcntl.h>   // for open

VADDetector::VADDetector(const Config& config)
    : config_(config), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

VADDetector::~VADDetector() {
    cleanup();
}

bool VADDetector::initialize() {
    try {
        // Temporarily suppress stderr to avoid ONNX schema warnings
        int stderr_fd = dup(STDERR_FILENO);
        int devnull_fd = open("/dev/null", O_WRONLY);
        dup2(devnull_fd, STDERR_FILENO);
        
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VADDetector");
        
        // Restore stderr
        dup2(stderr_fd, STDERR_FILENO);
        close(stderr_fd);
        close(devnull_fd);
        
        if (!initializeSession()) {
            return false;
        }
        
        reset();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize VAD detector: " << e.what() << std::endl;
        return false;
    }
}

bool VADDetector::initializeSession() {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names and shapes
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.reserve(num_input_nodes);
        input_shapes_.reserve(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_.push_back(std::string(input_name.get()));
            
            auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_shapes_.push_back(input_shape);
        }
        
        // Convert string names to const char* for ONNX runtime
        input_names_.reserve(input_names_str_.size());
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }
        
        // Output names and shapes
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_str_.reserve(num_output_nodes);
        output_names_.reserve(num_output_nodes);
        output_shapes_.reserve(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(std::string(output_name.get()));
            
            auto output_shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            output_shapes_.push_back(output_shape);
        }
        
        // Convert string names to const char* for ONNX runtime
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX session: " << e.what() << std::endl;
        return false;
    }
}

void VADDetector::cleanup() {
    session_.reset();
    env_.reset();
    
    // Clear name vectors
    input_names_str_.clear();
    output_names_str_.clear();
    input_names_.clear();
    output_names_.clear();
    
    // Clear other vectors
    input_shapes_.clear();
    output_shapes_.clear();
    state_h_.clear();
    state_c_.clear();
    context_.clear();
    prob_history_.clear();
}

void VADDetector::reset() {
    // Initialize states according to Silero VAD model
    state_h_.assign(2 * 1 * 128, 0.0f); // (2, 1, 128)
    state_c_.assign(2 * 1 * 128, 0.0f); // (2, 1, 128)
    context_.assign(1 * config_.context_size, 0.0f); // (1, context_size)
    prob_history_.clear();
}

float VADDetector::detectVAD(const std::vector<float>& audio) {
    return detectVAD(audio.data(), audio.size());
}

float VADDetector::detectVAD(const float* audio, size_t length) {
    if (!session_) {
        std::cerr << "VAD detector not initialized" << std::endl;
        return 0.0f;
    }
    
    try {
        // Prepare input data
        std::vector<float> input_audio;
        if (length != config_.window_size) {
            // Resample or pad to correct size
            input_audio.resize(config_.window_size);
            if (length > config_.window_size) {
                std::copy(audio, audio + config_.window_size, input_audio.begin());
            } else {
                std::copy(audio, audio + length, input_audio.begin());
                std::fill(input_audio.begin() + length, input_audio.end(), 0.0f);
            }
        } else {
            input_audio.assign(audio, audio + length);
        }
        
        
        // Concatenate context and current audio
        std::vector<float> x(config_.context_size + config_.window_size);
        std::copy(context_.begin(), context_.end(), x.begin());
        std::copy(input_audio.begin(), input_audio.end(), x.begin() + config_.context_size);
        
        // Update context for next frame
        if (x.size() >= config_.context_size) {
            std::copy(x.end() - config_.context_size, x.end(), context_.begin());
        }
        
        // Prepare input tensors
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(x.size())};
        std::vector<int64_t> sr_shape = {1};
        std::vector<int64_t> state_shape = {2, 1, 128};
        
        std::vector<int64_t> sr_value = {config_.sample_rate};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, x.data(), x.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, state_h_.data(), state_h_.size(), state_shape.data(), state_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, sr_value.data(), sr_value.size(), sr_shape.data(), sr_shape.size()));
        
        // Run inference
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                          input_names_.data(), input_tensors.data(), input_tensors.size(),
                                          output_names_.data(), output_names_.size());
        
        // Get probability result
        float* prob_data = output_tensors[0].GetTensorMutableData<float>();
        float prob = prob_data[0];
        
        // Update LSTM states (both h and c states)
        if (output_tensors.size() > 2) {
            // Update h state
            float* new_state_h = output_tensors[1].GetTensorMutableData<float>();
            std::copy(new_state_h, new_state_h + state_h_.size(), state_h_.begin());
            
            // Update c state
            float* new_state_c = output_tensors[2].GetTensorMutableData<float>();
            std::copy(new_state_c, new_state_c + state_c_.size(), state_c_.begin());
        } else if (output_tensors.size() > 1) {
            // Fallback: only update h state if c state not available
            float* new_state_h = output_tensors[1].GetTensorMutableData<float>();
            std::copy(new_state_h, new_state_h + state_h_.size(), state_h_.begin());
        }
        
        // Apply smoothing
        prob_history_.push_back(prob);
        if (prob_history_.size() > config_.history_size) {
            prob_history_.pop_front();
        }
        
        float smoothed_prob = std::accumulate(prob_history_.begin(), prob_history_.end(), 0.0f) / prob_history_.size();
        
        return smoothed_prob;
        
    } catch (const std::exception& e) {
        std::cerr << "VAD inference error: " << e.what() << std::endl;
        return 0.0f;
    }
}

void VADDetector::resampleIfNeeded(const float* input, size_t input_length, 
                                  std::vector<float>& output) {
    // Simple resampling implementation
    if (input_length == config_.window_size) {
        output.assign(input, input + input_length);
        return;
    }
    
    output.resize(config_.window_size);
    
    if (input_length > config_.window_size) {
        // Downsample by simple decimation
        float ratio = static_cast<float>(input_length) / config_.window_size;
        for (size_t i = 0; i < config_.window_size; ++i) {
            size_t src_idx = static_cast<size_t>(i * ratio);
            output[i] = input[src_idx];
        }
    } else {
        // Upsample by linear interpolation
        float ratio = static_cast<float>(input_length) / config_.window_size;
        for (size_t i = 0; i < config_.window_size; ++i) {
            float src_pos = i * ratio;
            size_t src_idx = static_cast<size_t>(src_pos);
            float frac = src_pos - src_idx;
            
            if (src_idx + 1 < input_length) {
                output[i] = input[src_idx] * (1.0f - frac) + input[src_idx + 1] * frac;
            } else {
                output[i] = input[src_idx];
            }
        }
    }
}