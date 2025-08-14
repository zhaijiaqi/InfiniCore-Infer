#ifndef INFINICORE_INFER_AWQ_GEMM_CUDA_HPP
#define INFINICORE_INFER_AWQ_GEMM_CUDA_HPP

#include "awq_gemm.hpp"
#include "infinicore_infer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __CUDACC__
#include <cublas_v2.h>
#endif

namespace awq_gemm {
namespace cuda {

// 前向声明
class QuantizationParamsManager;

// 矩阵信息结构
struct MatrixInfo {
    size_t row_stride;
    size_t leading_dim;  // 添加leading dimension成员
    size_t ld() const { return leading_dim; }
    size_t stride;
};

// GEMM信息结构
struct GemmInfo {
    size_t m, n, k, batch;
    bool is_transed;
    MatrixInfo a_matrix, b_matrix, c_matrix;
};

// AWQ描述符类
class AWQDescriptor {
public:
    infiniDtype_t _dtype;
    int _device_id;
    GemmInfo _info;

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        float beta,
        const void *a,
        const void *b,
        float alpha,
        void *stream) const;
};

// CUDA AWQ量化参数（GPU可见）
struct CudaAWQParams {
    float global_scale;                    // 全局缩放因子
    float* group_scales;                   // 分组缩放因子（GPU内存）
    int group_size;                        // 量化分组大小
    bool per_channel;                      // 是否使用通道量化
    int num_groups;                        // 分组数量
};

// CUDA kernel配置参数
struct CudaGemmConfig {
    static constexpr int BLOCK_SIZE_M = 128;      // M维度的线程块大小
    static constexpr int BLOCK_SIZE_N = 128;      // N维度的线程块大小  
    static constexpr int BLOCK_SIZE_K = 32;       // K维度的线程块大小
    static constexpr int WARP_SIZE = 32;          // warp大小
    static constexpr int SHARED_MEM_SIZE = 48 * 1024; // 共享内存大小
};

#ifdef __CUDACC__

// CUDA设备函数：AWQ量化
__device__ __forceinline__ int8_t awq_quantize_value(
    float input_val, 
    float scale, 
    int8_t min_val = -128, 
    int8_t max_val = 127);

// CUDA设备函数：反量化
__device__ __forceinline__ float awq_dequantize_value(
    int32_t quantized_val, 
    float act_scale, 
    float weight_scale);

// 主要的融合AWQ量化GEMM kernel
__global__ void awq_quantized_gemm_kernel(
    const __half* __restrict__ A,          // 输入激活值 [M, K] (FP16)
    const int8_t* __restrict__ B,          // 预量化权重 [K, N] (INT8)
    const float* __restrict__ B_scales,    // 权重缩放因子 [N] 或 [N/group_size, N]
    const CudaAWQParams params,            // AWQ量化参数
    __half* __restrict__ C,                // 输出 [M, N] (FP16)
    int M, int K, int N,                   // 矩阵维度
    float alpha = 1.0f, float beta = 0.0f); // 缩放因子

// 优化版本：使用Tensor Core的融合kernel
__global__ void awq_quantized_gemm_tensorcore_kernel(
    const __half* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ B_scales,
    const CudaAWQParams params,
    __half* __restrict__ C,
    int M, int K, int N,
    float alpha = 1.0f, float beta = 0.0f);

// 小矩阵优化版本
__global__ void awq_quantized_gemm_small_kernel(
    const __half* __restrict__ A,
    const int8_t* __restrict__ B, 
    const float* __restrict__ B_scales,
    const CudaAWQParams params,
    __half* __restrict__ C,
    int M, int K, int N,
    float alpha = 1.0f, float beta = 0.0f);

#endif // __CUDACC__

// 主机端接口函数 - 都使用cudaError_t类型
extern "C" {
    cudaError_t launchAWQQuantizedGemm(
        const void* A,                         // FP16激活值
        const void* B,                         // INT8权重
        const void* B_scales,                  // 权重缩放因子
        const AWQQuantizeParams& host_params,  // 主机端参数
        void* C,                               // FP16输出
        int M, int K, int N,                   // 矩阵维度
        float alpha, float beta,
        cudaStream_t stream);

    // 新的接口函数，支持权重名称和量化参数管理器
    cudaError_t launchAWQQuantizedGemmWithName(
        const void* A,                         // FP16激活值
        const void* B,                         // INT8权重
        const void* B_scales,                  // 权重缩放因子
        const std::string& weight_name,        // 权重名称
        void* C,                               // FP16输出
        int M, int K, int N,                   // 矩阵维度
        float alpha, float beta,
        cudaStream_t stream);

    // GPU内存管理辅助函数
    cudaError_t allocateAWQParams(
        const AWQQuantizeParams& host_params,
        CudaAWQParams& device_params);

    cudaError_t freeAWQParams(CudaAWQParams& device_params);
}

// 性能分析和自动调优
struct AWQKernelSelector {
    enum KernelType {
        STANDARD_KERNEL,        // 标准kernel
        TENSORCORE_KERNEL,      // Tensor Core优化
        SMALL_MATRIX_KERNEL     // 小矩阵优化
    };
    
    static KernelType selectOptimalKernel(int M, int K, int N, int device_id);
};

} // namespace cuda
} // namespace awq_gemm

#endif // INFINICORE_INFER_AWQ_GEMM_CUDA_HPP 