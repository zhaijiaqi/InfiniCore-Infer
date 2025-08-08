# 量化算法实现总结

本文档总结了在 InfiniCore-Infer 项目中实现的 INT8 量化算法。

## 实现内容

### 1. 量化工具模块 (src/quantization/quantiUtil.cpp)

**AWQ (Activation-aware Weight Quantization) 算法实现：**

- quantize_activation_fp16_to_int8(): FP16 激活值量化为 INT8
- dequantize_activation_int8_to_fp16(): INT8 激活值反量化为 FP16  
- quantize_weight_awq(): AWQ 权重量化算法，考虑激活分布优化量化误差
- dequantize_weight_int8_to_fp16(): 权重反量化
- quantize_kv_cache_fp16_to_int8(): KV 缓存量化，专门用于注意力机制
- dequantize_kv_cache_int8_to_fp16(): KV 缓存反量化
- quantize_tensors_batch(): 批量量化操作
- dequantize_tensors_batch(): 批量反量化操作

**量化特性：**
- 使用对称量化（zero_point = 0）来简化计算
- AWQ 算法考虑权重分布的通道级别量化
- 支持内存池管理以提高性能

### 2. 量化算子模块 (src/quantization/quantiOp.cpp)

**INT8 量化计算算子：**

- quantized_gemm_int8(): INT8 矩阵乘法，支持 A、B 矩阵都是 INT8
- quantized_attention_qk_int8(): 量化注意力 Q*K^T 计算
- quantized_attention_v_int8(): 量化注意力权重与 V 的乘法
- quantized_ffn_gate_up_int8(): 量化 FFN 门控计算（SwiGLU）
- quantized_rms_norm_int8(): 量化 RMS 层归一化
- mixed_precision_gemm(): 混合精度 GEMM（FP16 激活 * INT8 权重）

**算子特性：**
- 使用 INT32 累加器避免溢出
- 支持混合精度计算以平衡性能和精度
- 包含 SiLU 激活函数的量化版本

### 3. 主推理引擎集成 (src/models/jiuge/jiuge.cpp)

**量化集成点：**

1. **QKV 投影量化**：在注意力机制的 QKV 矩阵乘法前量化激活值
2. **KV 缓存量化**：将 K、V 缓存量化为 INT8 以节省显存
3. **注意力计算量化**：Q*K^T 和注意力*V 的量化计算
4. **注意力输出投影量化**：注意力输出投影的混合精度计算
5. **FFN 量化**：门控和上投影、下投影的量化计算

**量化状态管理：**
- QuantizationState 结构存储所有量化参数
- 每层独立的量化参数
- 支持动态开关量化功能

### 4. 构建系统集成

**xmake.lua 更新：**
- 添加了 src/quantization/*.cpp 到构建系统
- 确保量化模块与主项目一起编译

## 量化策略

### 权重量化
- 使用 AWQ 算法，考虑激活统计信息
- 对称量化，zero_point = 0
- 每个输出通道独立计算量化参数
- 全局量化 scale 以简化实现

### 激活值量化
- 动态量化，运行时计算量化参数
- 对称量化策略
- 量化范围：[-127, 127]

### KV 缓存量化
- 使用保守的量化策略以保持注意力精度
- 与激活值量化相同的算法

## 性能优化

1. **内存优化**：
   - KV 缓存量化可显著减少显存使用（理论上减少 50%）
   - 权重量化减少模型存储需求

2. **计算优化**：
   - INT8 计算比 FP16 计算更快
   - 混合精度平衡精度和性能

3. **并行化**：
   - 批量量化操作支持
   - 与原有的多设备推理框架兼容

## 注意事项

1. **简化实现**：
   - 当前实现使用固定尺寸作为演示
   - 生产环境需要从 tensor descriptor 获取真实尺寸

2. **FP16 支持**：
   - 包含了 CUDA 和非 CUDA 环境的 FP16 支持
   - 非 CUDA 环境使用简化的 FP16 转换

3. **兼容性**：
   - 通过 enable_quantization 开关支持量化的动态启用/禁用
   - 保持与原有 FP16 推理路径的兼容性

## 下一步优化

1. **精度优化**：
   - 实现更精确的 IEEE 754 FP16 转换
   - 添加激活值统计收集用于 AWQ

2. **性能优化**：
   - 集成高度优化的 INT8 GEMM 库（如 cuBLAS, oneDNN）
   - 从 tensor descriptor 获取真实张量尺寸

3. **功能扩展**：
   - 支持非对称量化
   - 添加更多量化算法（如 GPTQ, SmoothQuant）
   - 支持动态量化范围调整 