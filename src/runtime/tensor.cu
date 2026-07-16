#include "tensor.hpp"
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <iostream>
namespace runtime {
namespace {
size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::BF16:
            return sizeof(__nv_bfloat16);
    }
    throw std::runtime_error(
        "Tensor: unsupported data type"
    );
}
int64_t calculate_numel(
    const std::vector<int64_t>& shape
) {
    if (shape.empty()) {
        throw std::invalid_argument(
            "Tensor: shape cannot be empty"
        );
    }
    int64_t numel = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument(
                "Tensor: dimensions must be positive"
            );
        }
        numel *= dimension;
    }
    return numel;
}
void check_cuda(
    cudaError_t error,
    const char* operation
) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            " failed: " +
            cudaGetErrorString(error)
        );
    }
}
} // namespace

Tensor::Tensor(
    const std::vector<int64_t>& shape,
    DataType dtype
) {
    allocate(shape, dtype);
}
Tensor::~Tensor() {
    release();
}
Tensor::Tensor(
    Tensor&& other
) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      numel_(other.numel_),
      bytes_(other.bytes_),
      dtype_(other.dtype_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.bytes_ = 0;
    other.shape_.clear();
}
Tensor& Tensor::operator=(
    Tensor&& other
) noexcept {
    if (this == &other) {
        return *this;
    }
    release();
    data_ = other.data_;
    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    bytes_ = other.bytes_;
    dtype_ = other.dtype_;
    other.data_ = nullptr;
    other.numel_ = 0;
    other.bytes_ = 0;
    other.shape_.clear();
    return *this;
}
void Tensor::allocate(
    const std::vector<int64_t>& shape,
    DataType dtype
) {
    release();

    numel_ = calculate_numel(shape);
    dtype_ = dtype;
    bytes_ = static_cast<size_t>(numel_) * dtype_size(dtype);

    check_cuda(
        cudaMalloc(&data_, bytes_),
        "Tensor cudaMalloc"
    );
    shape_ = shape;
}
void Tensor::release() {
    if (data_ != nullptr) {
        cudaFree(data_);
        data_ = nullptr;
    }
    shape_.clear();
    numel_ = 0;
    bytes_ = 0;
}
void* Tensor::data() noexcept {
    return data_;
}
const void* Tensor::data() const noexcept {
    return data_;
}
__nv_bfloat16* Tensor::data_bf16() noexcept {
    return static_cast<__nv_bfloat16*>(data_);
}
const __nv_bfloat16* Tensor::data_bf16() const noexcept {
    return static_cast<const __nv_bfloat16*>(data_);
}
const std::vector<int64_t>& Tensor::shape() const noexcept {
    return shape_;
}
int64_t Tensor::numel() const noexcept {
    return numel_;
}
size_t Tensor::bytes() const noexcept {
    return bytes_;
}
DataType Tensor::dtype() const noexcept {
    return dtype_;
}
bool Tensor::empty() const noexcept {
    return data_ == nullptr;
}
} // namespace runtime