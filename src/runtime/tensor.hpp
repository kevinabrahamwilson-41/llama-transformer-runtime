#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
namespace runtime {
enum class DataType {
    BF16
};
class Tensor {
public:
    Tensor() = default;
    Tensor(
        const std::vector<int64_t>& shape,
        DataType dtype = DataType::BF16
    );
    ~Tensor();
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    void allocate(
        const std::vector<int64_t>& shape,
        DataType dtype = DataType::BF16
    );
    void release();
    void* data() noexcept;
    const void* data() const noexcept;
    __nv_bfloat16* data_bf16() noexcept;
    const __nv_bfloat16* data_bf16() const noexcept;
    const std::vector<int64_t>& shape() const noexcept;
    int64_t numel() const noexcept;
    size_t bytes() const noexcept;
    DataType dtype() const noexcept;
    bool empty() const noexcept;
private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    int64_t numel_ = 0;
    size_t bytes_ = 0;
    DataType dtype_ = DataType::BF16;
};
} // namespace runtime