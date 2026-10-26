#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void set_error(char* buffer, std::size_t capacity, const std::string& message) noexcept {
    if (buffer == nullptr || capacity == 0) return;
    std::snprintf(buffer, capacity, "%s", message.c_str());
}

struct Runtime {
    Ort::Env env;
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::size_t block_size = 0;
    std::size_t vocab_size = 0;

    explicit Runtime(const char* model_path, std::size_t expected_block_size, std::size_t expected_vocab_size)
        : env(ORT_LOGGING_LEVEL_WARNING, "FROSS-C++-ONNX"), options() {
        if (model_path == nullptr || *model_path == '\0') {
            throw std::invalid_argument("ONNX model path is empty");
        }
        if (expected_block_size == 0 || expected_vocab_size == 0) {
            throw std::invalid_argument("expected model dimensions must be positive");
        }
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, model_path, options);

        const auto input_type_info = session->GetInputTypeInfo(0);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        const auto input_shape = input_info.GetShape();
        if (input_shape.size() != 2) {
            throw std::runtime_error("ONNX input must have rank 2: [batch, block_size]");
        }
        // Some PyTorch/ONNX exporter versions preserve symbolic dimensions even
        // when the example input has fixed shape. Python passes the architecture's
        // trusted dimensions, and we reject any conflicting static dimensions.
        if (input_shape[0] > 0 && input_shape[0] != 1) {
            throw std::runtime_error("ONNX model batch dimension is not 1");
        }
        if (input_shape[1] > 0 &&
            static_cast<std::size_t>(input_shape[1]) != expected_block_size) {
            throw std::runtime_error("ONNX model block size differs from checkpoint");
        }
        block_size = expected_block_size;

        const auto output_type_info = session->GetOutputTypeInfo(0);
        const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
        const auto output_shape = output_info.GetShape();
        if (output_shape.size() != 3) {
            throw std::runtime_error("ONNX output must have rank 3: [batch, block_size, vocab_size]");
        }
        if (output_shape[0] > 0 && output_shape[0] != 1) {
            throw std::runtime_error("ONNX model output batch dimension is not 1");
        }
        if (output_shape[1] > 0 &&
            static_cast<std::size_t>(output_shape[1]) != block_size) {
            throw std::runtime_error("ONNX output block size differs from checkpoint");
        }
        if (output_shape[2] > 0 &&
            static_cast<std::size_t>(output_shape[2]) != expected_vocab_size) {
            throw std::runtime_error("ONNX vocabulary size differs from checkpoint");
        }
        vocab_size = expected_vocab_size;
    }

    std::vector<float> forward(const std::int64_t* token_ids, std::size_t actual_length) {
        if (token_ids == nullptr || actual_length == 0 || actual_length > block_size) {
            throw std::invalid_argument("sequence length must be between 1 and block_size");
        }

        // Right padding is safe: the causal attention mask prevents positions after
        // the last real token from affecting its logits.
        std::vector<std::int64_t> padded(block_size, 0);
        std::copy_n(token_ids, actual_length, padded.begin());
        const std::int64_t shape[] = {1, static_cast<std::int64_t>(block_size)};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<std::int64_t>(
            memory_info, padded.data(), padded.size(), shape, 2);
        const char* input_names[] = {"input_ids"};
        const char* output_names[] = {"logits"};
        Ort::RunOptions run_options;
        auto outputs = session->Run(run_options, input_names, &input, 1, output_names, 1);
        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error("ONNX session returned no logits tensor");
        }
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto output_shape = info.GetShape();
        if (output_shape.size() != 3 || static_cast<std::size_t>(output_shape[2]) != vocab_size) {
            throw std::runtime_error("ONNX logits shape changed unexpectedly");
        }
        const float* data = outputs[0].GetTensorData<float>();
        const std::size_t offset = (actual_length - 1) * vocab_size;
        return std::vector<float>(data + offset, data + offset + vocab_size);
    }
};
}  // namespace

extern "C" {
void* fross_runtime_create(
    const char* model_path,
    std::size_t expected_block_size,
    std::size_t expected_vocab_size,
    char* error,
    std::size_t error_capacity
) noexcept {
    try {
        return new Runtime(model_path, expected_block_size, expected_vocab_size);
    } catch (const std::exception& exc) {
        set_error(error, error_capacity, exc.what());
    } catch (...) {
        set_error(error, error_capacity, "unknown error creating C++ runtime");
    }
    return nullptr;
}

void fross_runtime_destroy(void* handle) noexcept {
    delete static_cast<Runtime*>(handle);
}

int fross_runtime_forward(
    void* handle,
    const std::int64_t* token_ids,
    std::size_t actual_length,
    float* logits_out,
    std::size_t output_capacity,
    char* error,
    std::size_t error_capacity
) noexcept {
    try {
        if (handle == nullptr) throw std::invalid_argument("runtime handle is null");
        auto* runtime = static_cast<Runtime*>(handle);
        if (logits_out == nullptr || output_capacity < runtime->vocab_size) {
            throw std::invalid_argument("output buffer is smaller than model vocabulary");
        }
        const auto logits = runtime->forward(token_ids, actual_length);
        std::copy(logits.begin(), logits.end(), logits_out);
        return static_cast<int>(logits.size());
    } catch (const std::exception& exc) {
        set_error(error, error_capacity, exc.what());
    } catch (...) {
        set_error(error, error_capacity, "unknown error during C++ inference");
    }
    return -1;
}
}
