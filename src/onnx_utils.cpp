#include "onnx_utils.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#endif

namespace speaker_recognition {

namespace {

class ScopedStderrSilencer {
public:
    ScopedStderrSilencer() {
#if defined(_WIN32)
        original_fd_ = _dup(_fileno(stderr));
        if (original_fd_ == -1) {
            return;
        }

        FILE* null_file = _fdopen(_open("nul", _O_WRONLY), "w");
        if (!null_file) {
            _close(original_fd_);
            original_fd_ = -1;
            return;
        }

        null_stream_ = null_file;
        fflush(stderr);
        _dup2(_fileno(null_stream_), _fileno(stderr));
#else
        original_fd_ = dup(fileno(stderr));
        if (original_fd_ == -1) {
            return;
        }

        null_stream_ = fopen("/dev/null", "w");
        if (!null_stream_) {
            close(original_fd_);
            original_fd_ = -1;
            return;
        }

        fflush(stderr);
        dup2(fileno(null_stream_), fileno(stderr));
#endif
    }

    ~ScopedStderrSilencer() {
#if defined(_WIN32)
        if (original_fd_ != -1) {
            fflush(stderr);
            _dup2(original_fd_, _fileno(stderr));
            _close(original_fd_);
        }

        if (null_stream_) {
            fclose(null_stream_);
        }
#else
        if (original_fd_ != -1) {
            fflush(stderr);
            dup2(original_fd_, fileno(stderr));
            close(original_fd_);
        }

        if (null_stream_) {
            fclose(null_stream_);
        }
#endif
    }

private:
    int original_fd_ = -1;
    FILE* null_stream_ = nullptr;
};

} // namespace

OnnxModel::OnnxModel(const std::string& model_path, int32_t num_threads, const std::string& provider) {
    // Disable all ONNX Runtime logs including Schema errors
    setenv("ORT_DISABLE_ALL_LOGS", "1", 1);

    ScopedStderrSilencer silencer;

    // Initialize ONNX Runtime environment with ERROR level to suppress warnings
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "speaker_recognition");

    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (num_threads > 0) {
        session_options.SetIntraOpNumThreads(num_threads);
    }

    // Add execution provider if specified
    if (provider == "cuda") {
        // Note: CUDA provider would require additional setup
        std::cerr << "CUDA provider requested but not implemented in this version" << std::endl;
    } else if (provider == "spacemit") {
        std::cerr << "SpaceMIT provider requested but not implemented in this version" << std::endl;
    }

    // Create session
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // Create memory info
    memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // Get input and output names
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session_->GetInputCount();
    input_names_allocated_.reserve(num_inputs);
    input_names_ptr_.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
        input_names_allocated_.push_back(session_->GetInputNameAllocated(i, allocator));
        input_names_ptr_.push_back(input_names_allocated_.back().get());
    }

    size_t num_outputs = session_->GetOutputCount();
    output_names_allocated_.reserve(num_outputs);
    output_names_ptr_.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; i++) {
        output_names_allocated_.push_back(session_->GetOutputNameAllocated(i, allocator));
        output_names_ptr_.push_back(output_names_allocated_.back().get());
    }

    // Get embedding dimension from output shape
    auto output_info = session_->GetOutputTypeInfo(0);
    auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    if (shape.size() >= 2) {
        embedding_dim_ = static_cast<int32_t>(shape[1]);
    } else {
        embedding_dim_ = 256;  // Default value
    }
}

std::vector<float> OnnxModel::RunInference(const std::vector<float>& input, const std::vector<int64_t>& input_shape) const {
    try {
        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            *memory_info_,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_ptr_.data(),
            &input_tensor,
            1,
            output_names_ptr_.data(),
            output_names_ptr_.size()
        );

        // Get output data
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // Copy output to vector
        std::vector<float> result(output_data, output_data + output_size);

        return result;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return std::vector<float>();
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return std::vector<float>();
    }
}

} // namespace speaker_recognition