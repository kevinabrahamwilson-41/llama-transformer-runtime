// runtime/debug_dump.hpp

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
inline void save_checkpoint(
    const char* name,
    const __nv_bfloat16* device_ptr,
    size_t elements
){
    namespace fs = std::filesystem;

    const std::string DEBUG_DIR = "cuda_debug";

    fs::create_directories(DEBUG_DIR);

    if(device_ptr == nullptr){
        printf(
            "[CHECKPOINT] %s: NULL\n",
            name
        );
        return;
    }

    // --------------------------------------------------
    // GPU BF16 -> CPU BF16
    // --------------------------------------------------

    std::vector<__nv_bfloat16> host_bf16(elements);

    cudaError_t err = cudaMemcpy(
        host_bf16.data(),
        device_ptr,
        elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost
    );

    if(err != cudaSuccess){
        printf(
            "[CHECKPOINT] %s: cudaMemcpy FAILED: %s\n",
            name,
            cudaGetErrorString(err)
        );
        return;
    }

    // --------------------------------------------------
    // BF16 -> FP32
    // --------------------------------------------------

    std::vector<float> host_float(elements);

    for(size_t i = 0; i < elements; i++){
        host_float[i] =
            __bfloat162float(host_bf16[i]);
    }

    // --------------------------------------------------
    // Save FP32 with header
    // --------------------------------------------------

    const std::string filename =
        DEBUG_DIR + "/" +
        std::string(name) +
        ".bin";

    std::ofstream file(
        filename,
        std::ios::binary
    );

    if(!file){
        printf(
            "[CHECKPOINT] %s: FAILED OPENING FILE\n",
            name
        );
        return;
    }

    uint64_t count =
        static_cast<uint64_t>(elements);

    uint32_t dtype = 0; // FP32

    uint32_t reserved = 0;

    file.write(
        reinterpret_cast<const char*>(&count),
        sizeof(count)
    );

    file.write(
        reinterpret_cast<const char*>(&dtype),
        sizeof(dtype)
    );

    file.write(
        reinterpret_cast<const char*>(&reserved),
        sizeof(reserved)
    );

    file.write(
        reinterpret_cast<const char*>(host_float.data()),
        elements * sizeof(float)
    );
    file.close();
}