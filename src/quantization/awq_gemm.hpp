#ifndef INFINICORE_INFER_AWQ_GEMM_HPP
#define INFINICORE_INFER_AWQ_GEMM_HPP

#include "../tensor.hpp"
#include <memory>

#ifdef __cplusplus

namespace awq_gemm {

// AWQ量化参数结构
struct AWQQuantizeParams {
    float scale;                    // 全局缩放因子
    std::vector<float> group_scales; // 分组缩放因子（可选）
    int group_size;                 // 量化分组大小，默认128
    bool per_channel;               // 是否使用通道量化
    
    AWQQuantizeParams() : scale(1.0f), group_size(128), per_channel(false) {}
};

// 融合的AWQ量化+GEMM操作
// 输入: FP16 激活值 (A) 和 INT8 权重 (B)
// 输出: FP16/INT32 结果 (C)
// 公式: C = quantize(A) × B_int8，其中quantize使用AWQ算法
infiniStatus_t awqQuantizedGemm(
    infiniopHandle_t handle,
    std::shared_ptr<Tensor> output_tensor,        // 输出张量 [M, N]
    std::shared_ptr<Tensor const> activation_tensor,    // 激活张量 [M, K] (FP16)
    std::shared_ptr<Tensor const> weight_tensor,        // 权重张量 [K, N] (INT8, 预量化)
    std::shared_ptr<Tensor const> weight_scales,        // 权重量化缩放因子 [N] 或 [N/group_size, N]
    const AWQQuantizeParams& quant_params,        // AWQ量化参数
    float alpha = 1.0f,                          // 乘法因子
    float beta = 0.0f,                           // 加法因子
    infinirtStream_t stream = nullptr);

// 新的AWQ量化+GEMM操作，支持权重名称和量化参数管理器
infiniStatus_t awqQuantizedGemmWithName(
    infiniopHandle_t handle,
    std::shared_ptr<Tensor> output_tensor,        // 输出张量 [M, N]
    std::shared_ptr<Tensor const> activation_tensor,    // 激活张量 [M, K] (FP16)
    std::shared_ptr<Tensor const> weight_tensor,        // 权重张量 [K, N] (INT8, 预量化)
    std::shared_ptr<Tensor const> weight_scales,        // 权重量化缩放因子 [N] 或 [N/group_size, N]
    const std::string& weight_name,               // 权重名称（用于获取量化参数）
    float alpha = 1.0f,                          // 乘法因子
    float beta = 0.0f,                           // 加法因子
    infinirtStream_t stream = nullptr);

// 创建AWQ量化参数的辅助函数
// 基于激活值统计信息计算最优量化参数
AWQQuantizeParams createAWQParams(
    std::shared_ptr<Tensor const> activation_tensor,
    int group_size = 128,
    bool per_channel = false);

// 在线AWQ权重量化
// 将FP16权重量化为INT8，同时生成缩放因子
infiniStatus_t quantizeWeightsAWQ(
    infiniopHandle_t handle,
    std::shared_ptr<Tensor> quantized_weights,   // 输出: INT8权重 [K, N]
    std::shared_ptr<Tensor> weight_scales,       // 输出: 缩放因子
    std::shared_ptr<Tensor const> fp_weights,    // 输入: FP16权重 [K, N]
    std::shared_ptr<Tensor const> sample_activations, // 样本激活值用于AWQ校准
    const AWQQuantizeParams& params,
    infinirtStream_t stream = nullptr);

// 检查是否支持AWQ量化GEMM
bool isAWQGemmSupported(infiniDevice_t device);

} // namespace awq_gemm

#endif // __cplusplus

#endif // INFINICORE_INFER_AWQ_GEMM_HPP 