# AWQ GEMM 性能改进

## 概述

本次改进将AWQ GEMM实现从自定义CUDA kernel改为使用高性能的`cublasGemmStridedBatchedEx`函数，显著提升了性能并支持INT8*INT8->INT32的矩阵乘法。

## 主要改进

### 1. 使用cublasGemmStridedBatchedEx

- **性能提升**: 利用NVIDIA优化的cuBLAS库，相比自定义kernel有显著性能提升
- **INT8支持**: 原生支持INT8*INT8->INT32的矩阵乘法
- **Tensor Core**: 自动利用Tensor Core加速（如果硬件支持）
- **批处理**: 支持批量矩阵乘法操作

### 2. 新的Descriptor接口

实现了类似cuDNN的Descriptor模式：

```cpp
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
```

### 3. 数据类型支持

支持多种数据类型组合：

- **FP16**: `CUDA_R_16F` -> `CUDA_R_16F`
- **BF16**: `CUDA_R_16BF` -> `CUDA_R_16BF`  
- **FP32**: `CUDA_R_32F` -> `CUDA_R_32F`
- **INT8**: `CUDA_R_8I` * `CUDA_R_8I` -> `CUDA_R_32I`

### 4. 兼容性

- 保持与现有API的兼容性
- 支持不同的CUDA后端（ILUVATAR、SUGON等）
- 自动处理矩阵转置

## 性能优势

1. **更高的吞吐量**: cuBLAS库经过NVIDIA深度优化
2. **更好的内存访问模式**: 优化的内存布局和访问策略
3. **硬件加速**: 自动利用Tensor Core和最新的GPU特性
4. **减少开发维护**: 使用成熟的库而不是自定义实现

## 使用示例

```cpp
// 创建AWQ描述符
AWQDescriptor desc;
desc._dtype = INFINI_DTYPE_I8;
desc._device_id = device_id;

// 设置矩阵信息
desc._info.m = M;
desc._info.n = N; 
desc._info.k = K;
desc._info.batch = 1;

// 执行计算
infiniStatus_t status = desc.calculate(
    workspace, workspace_size,
    output, beta,
    input_a, input_b, alpha,
    stream
);
```

## 编译要求

- CUDA 11.0+
- cuBLAS库
- 支持Tensor Core的GPU（推荐）

## 测试

运行测试程序验证实现：

```bash
# 编译测试程序
nvcc -o test_awq test_awq_gemm.cpp -lcublas -lcudart

# 运行测试
./test_awq
```

## 未来改进

1. **混合精度**: 支持FP16/INT8混合精度计算
2. **动态量化**: 运行时量化参数调整
3. **更多优化**: 针对特定矩阵大小的优化策略
4. **内存优化**: 减少内存分配和拷贝开销 