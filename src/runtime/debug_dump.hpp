#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

inline void dump_bf16(
    const char* path,
    const __nv_bfloat16* device_ptr,
    int elements
)
{
    std::vector<__nv_bfloat16> host(
        elements
    );

    cudaError_t error =
        cudaMemcpy(
            host.data(),
            device_ptr,
            elements * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost
        );

    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CUDA memcpy failed: ") +
            cudaGetErrorString(error)
        );
    }

    FILE* file =
        fopen(path, "w");

    if (!file)
    {
        throw std::runtime_error(
            std::string("Failed to open dump: ") +
            path
        );
    }

    for (int i = 0; i < elements; ++i)
    {
        fprintf(
            file,
            "%.9g\n",
            __bfloat162float(host[i])
        );
    }

    fclose(file);
}