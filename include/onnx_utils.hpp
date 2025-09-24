#ifndef ONNX_UTILS_HPP_
#define ONNX_UTILS_HPP_

#include "onnx_compat.h"  // Include compatibility fixes first
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace speaker_recognition {

class OnnxModel {
public:
    OnnxModel(const std::string& model_path, int32_t num_threads, const std::string& provider);
    ~OnnxModel() = default;

    std::vector<float> RunInference(const std::vector<float>& input, const std::vector<int64_t>& input_shape) const;
    int32_t GetEmbeddingDim() const { return embedding_dim_; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    // Store AllocatedStringPtr to keep ownership of strings
    std::vector<Ort::AllocatedStringPtr> input_names_allocated_;
    std::vector<Ort::AllocatedStringPtr> output_names_allocated_;
    std::vector<const char*> input_names_ptr_;
    std::vector<const char*> output_names_ptr_;
    int32_t embedding_dim_;
};

} // namespace speaker_recognition

#endif // ONNX_UTILS_HPP_