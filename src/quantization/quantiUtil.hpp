#pragma once

#include "../tensor.hpp"
#include "infinicore_infer.h"
#include <memory>
#include <vector>

namespace Quantization {

// 量化参数结构体
struct QuantizationParams {
    float scale;        // 量化缩放因子
    int zero_point;     // 零点偏移
    
    QuantizationParams();
    QuantizationParams(float s, int zp);
};

// 重新设计的量化函数接口 - 与InfiniCore框架标准一致

// 激活值量化函数 - 使用底层接口
infiniStatus_t quantize_activation_fp16_to_int8(
    void* output,                           // INT8输出数据
    const void* input,                      // FP16输入数据
    size_t total_elements,                  // 元素总数
    float& scale,                          // 输出：量化缩放因子
    int& zero_point);                      // 输出：零点偏移

infiniStatus_t dequantize_activation_int8_to_fp16(
    void* output,                           // FP16输出数据
    const void* input,                      // INT8输入数据
    size_t total_elements,                  // 元素总数
    float scale,                           // 量化缩放因子
    int zero_point);                       // 零点偏移

// AWQ权重量化函数 - 使用底层接口
infiniStatus_t quantize_weight_awq(
    void* output,                           // INT8输出权重
    const void* weight,                     // FP16输入权重
    size_t total_elements,                  // 元素总数
    float& scale,                          // 输出：量化缩放因子
    int& zero_point);                      // 输出：零点偏移

infiniStatus_t dequantize_weight_int8_to_fp16(
    void* output,                           // FP16输出数据
    const void* input,                      // INT8输入数据
    size_t total_elements,                  // 元素总数
    float scale,                           // 量化缩放因子
    int zero_point);                       // 零点偏移

// KV缓存量化函数 - 使用底层接口
infiniStatus_t quantize_kv_cache_fp16_to_int8(
    void* output,                           // INT8输出数据
    const void* input,                      // FP16输入数据
    size_t total_elements,                  // 元素总数
    float& scale,                          // 输出：量化缩放因子
    int& zero_point);                      // 输出：零点偏移

infiniStatus_t dequantize_kv_cache_int8_to_fp16(
    void* output,                           // FP16输出数据
    const void* input,                      // INT8输入数据
    size_t total_elements,                  // 元素总数
    float scale,                           // 量化缩放因子
    int zero_point);                       // 零点偏移

// 保留高级接口作为便利函数（可选）
std::shared_ptr<Tensor> quantize_activation_fp16_to_int8_tensor(
    std::shared_ptr<Tensor> input, 
    float& scale, 
    int& zero_point,
    std::shared_ptr<MemoryPool> memory_pool = nullptr);

std::shared_ptr<Tensor> quantize_weight_awq_tensor(
    std::shared_ptr<Tensor> weight,
    float& scale,
    int& zero_point,
    std::shared_ptr<MemoryPool> memory_pool = nullptr);

} // namespace Quantization 