#pragma once

#include "../tensor.hpp"
#include "infinicore_infer.h"

namespace Quantization {

// INT8量化GEMM算子
void quantized_gemm_int8(infiniopHandle_t handle,
                        void* workspace, 
                        size_t workspace_size,
                        void* C_data,           // FP16输出
                        const void* A_data,     // INT8激活
                        const void* B_data,     // INT8权重
                        float A_scale,          // A的量化参数
                        int A_zero_point,
                        float B_scale,          // B的量化参数  
                        int B_zero_point,
                        float alpha,
                        float beta,
                        infiniopTensorDescriptor_t A_desc,
                        infiniopTensorDescriptor_t B_desc,
                        infiniopTensorDescriptor_t C_desc,
                        infinirtStream_t stream);

// 量化注意力Q*K计算
void quantized_attention_qk_int8(infiniopHandle_t handle,
                                void* workspace,
                                size_t workspace_size,
                                void* qk_output,        // FP16输出
                                const void* q_data,     // INT8 Query
                                const void* k_data,     // INT8 Key
                                float q_scale,
                                int q_zero_point,
                                float k_scale,
                                int k_zero_point,
                                float alpha,
                                infiniopTensorDescriptor_t q_desc,
                                infiniopTensorDescriptor_t k_desc,
                                infiniopTensorDescriptor_t qk_desc,
                                infinirtStream_t stream);

// 量化注意力与V计算
void quantized_attention_v_int8(infiniopHandle_t handle,
                               void* workspace,
                               size_t workspace_size,
                               void* output,           // FP16输出
                               const void* attn_data,  // FP16注意力权重
                               const void* v_data,     // INT8 Value
                               float v_scale,
                               int v_zero_point,
                               infiniopTensorDescriptor_t attn_desc,
                               infiniopTensorDescriptor_t v_desc,
                               infiniopTensorDescriptor_t output_desc,
                               infinirtStream_t stream);

// 量化FFN门控计算
void quantized_ffn_gate_up_int8(infiniopHandle_t handle,
                               void* workspace,
                               size_t workspace_size,
                               void* output,           // FP16输出
                               const void* gate_data,  // INT8 Gate
                               const void* up_data,    // INT8 Up
                               float gate_scale,
                               int gate_zero_point,
                               float up_scale,
                               int up_zero_point,
                               infiniopTensorDescriptor_t gate_desc,
                               infiniopTensorDescriptor_t up_desc,
                               infiniopTensorDescriptor_t output_desc,
                               infinirtStream_t stream);

// 量化层归一化
void quantized_rms_norm_int8(infiniopHandle_t handle,
                           void* workspace,
                           size_t workspace_size,
                           void* output,           // FP16输出
                           const void* input,      // INT8输入
                           const void* weight,     // FP16权重
                           float input_scale,
                           int input_zero_point,
                           float epsilon,
                           infiniopTensorDescriptor_t input_desc,
                           infiniopTensorDescriptor_t weight_desc,
                           infiniopTensorDescriptor_t output_desc,
                           infinirtStream_t stream);

// 混合精度GEMM：FP16激活 * INT8权重
void mixed_precision_gemm(infiniopHandle_t handle,
                         void* workspace,
                         size_t workspace_size,
                         void* C_data,           // FP16输出
                         const void* A_data,     // FP16激活
                         const void* B_data,     // INT8权重
                         float B_scale,          // 权重量化参数
                         int B_zero_point,
                         float alpha,
                         float beta,
                         infiniopTensorDescriptor_t A_desc,
                         infiniopTensorDescriptor_t B_desc,
                         infiniopTensorDescriptor_t C_desc,
                         infinirtStream_t stream);

} // namespace Quantization 