#include "quantiUtil.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cfloat>  // 添加FLT_MAX支持

// 添加FP16支持
#ifdef __CUDACC__
#include <cuda_fp16.h>
#else
// 如果不是CUDA环境，定义简单的FP16替代
typedef uint16_t __half;
inline float __half2float(__half h) {
    // 简化的FP16到FP32转换，实际应该使用正确的IEEE 754转换
    union { uint16_t i; __half h; } u;
    u.h = h;
    return static_cast<float>(u.i) / 65536.0f; // 简化版本
}
inline __half __float2half(float f) {
    // 简化的FP32到FP16转换
    return static_cast<__half>(f * 65536.0f);
}
#endif

namespace Quantization {

// 量化参数存储结构体的实现
QuantizationParams::QuantizationParams() : scale(1.0f), zero_point(0) {}

QuantizationParams::QuantizationParams(float s, int zp) : scale(s), zero_point(zp) {}

// 底层接口实现 - 激活值量化
infiniStatus_t quantize_activation_fp16_to_int8(
    void* output,                           // INT8输出数据
    const void* input,                      // FP16输入数据
    size_t total_elements,                  // 元素总数
    float& scale,                          // 输出：量化缩放因子
    int& zero_point) {                     // 输出：零点偏移
    
    // 输入验证
    if (!output || !input || total_elements == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto input_data = static_cast<const __half*>(input);
    auto output_data = static_cast<int8_t*>(output);
    
    // 计算激活值的最大值和最小值，用于确定量化参数
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    for (size_t i = 0; i < total_elements; i++) {
        float val = __half2float(input_data[i]);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    // 对称量化参数计算
    zero_point = 0;
    scale = std::max(std::abs(min_val), std::abs(max_val)) / 127.0f;
    
    // 避免除零
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    
    // 执行量化
    for (size_t i = 0; i < total_elements; i++) {
        float val = __half2float(input_data[i]);
        int quantized = static_cast<int>(std::round(val / scale));
        quantized = std::clamp(quantized, -127, 127);
        output_data[i] = static_cast<int8_t>(quantized);
    }
    
    return INFINI_STATUS_SUCCESS;
}

// 底层接口实现 - 激活值反量化
infiniStatus_t dequantize_activation_int8_to_fp16(
    void* output,                           // FP16输出数据
    const void* input,                      // INT8输入数据
    size_t total_elements,                  // 元素总数
    float scale,                           // 量化缩放因子
    int zero_point) {                      // 零点偏移
    
    // 输入验证
    if (!output || !input || total_elements == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto input_data = static_cast<const int8_t*>(input);
    auto output_data = static_cast<__half*>(output);
    
    // 执行反量化
    for (size_t i = 0; i < total_elements; i++) {
        float val = scale * (static_cast<float>(input_data[i]) - zero_point);
        output_data[i] = __float2half(val);
    }
    
    return INFINI_STATUS_SUCCESS;
}

// 底层接口实现 - AWQ权重量化
infiniStatus_t quantize_weight_awq(
    void* output,                           // INT8输出权重
    const void* weight,                     // FP16输入权重
    size_t total_elements,                  // 元素总数
    float& scale,                          // 输出：量化缩放因子
    int& zero_point) {                     // 输出：零点偏移
    
    // 输入验证
    if (!output || !weight || total_elements == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto weight_data = static_cast<const __half*>(weight);
    auto output_data = static_cast<int8_t*>(output);
    
    // 简化版本：对整个张量计算全局量化参数
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    for (size_t i = 0; i < total_elements; i++) {
        float val = __half2float(weight_data[i]);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    // 对称量化参数计算
    zero_point = 0;
    scale = std::max(std::abs(min_val), std::abs(max_val)) / 127.0f;
    
    // 避免除零
    if (scale == 0.0f) {
        scale = 1.0f;
    }
    
    // 执行量化
    for (size_t i = 0; i < total_elements; i++) {
        float val = __half2float(weight_data[i]);
        int quantized = static_cast<int>(std::round(val / scale));
        quantized = std::clamp(quantized, -127, 127);
        output_data[i] = static_cast<int8_t>(quantized);
    }
    
    return INFINI_STATUS_SUCCESS;
}

// 底层接口实现 - 权重反量化
infiniStatus_t dequantize_weight_int8_to_fp16(
    void* output,                           // FP16输出数据
    const void* input,                      // INT8输入数据
    size_t total_elements,                  // 元素总数
    float scale,                           // 量化缩放因子
    int zero_point) {                      // 零点偏移
    
    return dequantize_activation_int8_to_fp16(output, input, total_elements, scale, zero_point);
}

// 底层接口实现 - KV缓存量化
infiniStatus_t quantize_kv_cache_fp16_to_int8(
    void* output,                           // INT8输出数据
    const void* input,                      // FP16输入数据
    size_t total_elements,                  // 元素总数
    float& scale,                          // 输出：量化缩放因子
    int& zero_point) {                     // 输出：零点偏移
    
    // KV缓存使用与激活值相同的量化策略
    return quantize_activation_fp16_to_int8(output, input, total_elements, scale, zero_point);
}

// 底层接口实现 - KV缓存反量化
infiniStatus_t dequantize_kv_cache_int8_to_fp16(
    void* output,                           // FP16输出数据
    const void* input,                      // INT8输入数据
    size_t total_elements,                  // 元素总数
    float scale,                           // 量化缩放因子
    int zero_point) {                      // 零点偏移
    
    return dequantize_activation_int8_to_fp16(output, input, total_elements, scale, zero_point);
}

