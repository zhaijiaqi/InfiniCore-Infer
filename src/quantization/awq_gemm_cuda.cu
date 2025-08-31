#include "awq_gemm_cuda.hpp"
#include "awq_quantization_params.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cuda_fp16.h>
#include <vector>

namespace awq_gemm {
namespace cuda {

// 量化和反量化的CUDA kernel实现

// 量化kernel
__global__ void awq_quantize_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    int8_t min_val,
    int8_t max_val,
    int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float quantized = input[idx] / scale;
        quantized = fmaxf(fminf(quantized, static_cast<float>(max_val)), static_cast<float>(min_val));
        output[idx] = static_cast<int8_t>(roundf(quantized));
    }
}

// 反量化kernel（按列使用权重scale，并融合 alpha/beta）
__global__ void awq_dequantize_kernel(
	const int32_t* __restrict__ input,
	float* __restrict__ output,
	float act_scale,
	const float* __restrict__ weight_scales,
	float alpha, float beta,
	int M, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < M * N) {
		int col = idx % N;
		float val = static_cast<float>(input[idx]) * act_scale * weight_scales[col];
		val = alpha * val + (beta != 0.0f ? beta * output[idx] : 0.0f);
		output[idx] = val;
	}
}

// 激活量化kernel - 支持FP16输入
__global__ void quantize_activation_kernel(
    const void* __restrict__ input_activation,    // 输入激活 (FP16或FP32)
    int8_t* __restrict__ quantized_activation,     // 输出INT8激活 [M, K]
    const float* __restrict__ scales,              // 量化因子 [N]
    int M, int K, int N,
    bool is_fp16) {  // 是否为FP16输入
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    
    // 获取对应的量化因子（这里简化处理，使用第一个scale）
    float scale = scales[0];
    
	// 读取输入值
	float input_val = 0.0f;
	if (is_fp16) {
		const __half* fp16_data = static_cast<const __half*>(input_activation);
		input_val = __half2float(fp16_data[idx]);
	} else {
		const float* fp32_data = static_cast<const float*>(input_activation);
		input_val = fp32_data[idx];
	}
    
    // 进行量化
    float quantized = input_val / scale;
    quantized = fmaxf(fminf(quantized, 127.0f), -128.0f);
    int8_t quantized_val = static_cast<int8_t>(roundf(quantized));
    
    // 写入量化后的值
    quantized_activation[idx] = quantized_val;
}

// 新增：计算FP16激活张量的 max(|x|)（逐张量scale）
__global__ void reduce_max_abs_fp16_kernel(
    const __half* __restrict__ input,
    float* __restrict__ block_max,
    int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    // 线程内累积
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float v = fabsf(__half2float(input[i]));
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // 归约到block最大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
    }
}

// 新的高性能AWQ GEMM实现，使用cublasGemmStridedBatchedEx
infiniStatus_t AWQDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    // 去除多余输出

    // 设置数据类型
    cudaDataType a_type, b_type, c_type;
#ifdef ENABLE_ILUVATAR_API
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif

    // 根据量化参数设置数据类型
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
#ifdef ENABLE_ILUVATAR_API
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
#ifdef ENABLE_ILUVATAR_API
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
#if defined ENABLE_ILUVATAR_API
        compute_type = CUDA_R_32F;
#elif defined ENABLE_SUGON_CUDA_API
        compute_type = CUBLAS_COMPUTE_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        break;
	case INFINI_DTYPE_I8:
		// INT8*INT8 -> INT32 GEMM
		a_type = CUDA_R_8I;
		b_type = CUDA_R_8I;
		c_type = CUDA_R_32I;
        // std::cout << "AWQDescriptor::calculate: Using INT8*INT8->INT32 GEMM" << std::endl;
#ifdef ENABLE_ILUVATAR_API
		compute_type = CUBLAS_COMPUTE_32I_PEDANTIC;
#else
		compute_type = CUBLAS_COMPUTE_32I;
#endif
		break;
	default:
		// std::cout << "AWQDescriptor::calculate: ERROR - Bad tensor dtype: " << _dtype << std::endl;
		return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 行主序 -> 列主序 映射：计算 C^T = B^T * A^T
    // 此处 B 为列主序 [K,N] 的权重（由打包阶段保证），A 为量化后激活（行主序 [M,K]）
    // 选择 op_a = T（权重转置）、op_b = N（激活视作列主序[K,M]）
    auto op_a = CUBLAS_OP_T;  // A' = B^T（N x K）
    auto op_b = CUBLAS_OP_N;  // B' = A^T（K x M）

    int M = static_cast<int>(_info.m);
    int N = static_cast<int>(_info.n);
    int K = static_cast<int>(_info.k);

    // 列主序调用维度：C^T 形状 [N, M]
    int m_col = N, n_col = M, k_col = K;

    const void* A_mat = b; // 权重（列主序 [K,N]），经op_a=T后作为 [N,K]
    const void* B_mat = a; // 激活量化（行主序[M,K]），等价列主序[K,M]，op_b=N

    // 列主序下的leading dim（以“元素数”为单位，不能随意‘对齐’放大）
    int lda = K; // A' 来源矩阵（权重）原始行数 K
    int ldb = K; // B' 列主序 [K,M]
    int ldc = N; // C^T 列主序 [N,M]

    // 句柄/流
    cublasHandle_t handle;
    int device_id = _device_id;
    cudaError_t cuda_err = cudaGetDevice(&device_id);
    if (cuda_err != cudaSuccess) {
        // std::cout << "AWQDescriptor::calculate: ERROR - Failed to get device" << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cublasStatus_t create_status = cublasCreate(&handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
        // std::cout << "AWQDescriptor::calculate: ERROR - Failed to create cublas handle" << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (stream) {
        cublasSetStream(handle, (cudaStream_t)stream);
    }

    long long strideA = 0, strideB = 0, strideC = 0;
    int batchCount = 1;

    // INT8 GEMM 的 alpha/beta 必须为 int32
    int32_t alpha_i = 1;
    int32_t beta_i = 0;

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        op_a, op_b,
        m_col, n_col, k_col,
        &alpha_i,
        A_mat, b_type, lda, strideA,   // A' = B^T
        B_mat, a_type, ldb, strideB,   // B' = A^T
        &beta_i,
        c, c_type, ldc, strideC,       // C^T (INT32)
        batchCount,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        // std::cout << "AWQDescriptor::calculate: ERROR - cublasGemmStridedBatchedEx failed with status: " << status << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

// 兼容旧的接口函数，现在调用新的Descriptor实现
cudaError_t launchAWQQuantizedGemm(
    const void* A,
    const void* B,
    const void* B_scales,
    const AWQQuantizeParams& host_params,
    void* C,
    int M, int K, int N,
    float alpha, float beta,
    cudaStream_t stream) {
    
    // 去除多余输出
    
    // 检查CUDA设备状态
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        // std::cout << "CUDA AWQ GEMM: ERROR - Failed to get device: " << cudaGetErrorString(err) << std::endl;
        return err;
    }

    // 分配4字节对齐的临时缓冲区用于量化后的激活
    void* quantized_A = nullptr;
    size_t quantized_size = M * K * sizeof(int8_t);
    // 确保量化激活缓冲区4字节对齐（对于INT32计算要求）
    err = cudaMalloc(&quantized_A, quantized_size);
    if (err != cudaSuccess) {
        // std::cout << "CUDA AWQ GEMM: ERROR - Failed to allocate aligned quantized activation buffer: " << cudaGetErrorString(err) << std::endl;
        return err;
    }

    // 准备一个函数作用域的激活scale变量，供量化与反量化共用
    float act_scale_value = 1.0f / 127.0f;
    
    // 使用quantize_activation_kernel对A进行量化（动态按张量计算激活scale）
    {
        int threads = 256;
        int total = M * K;
        int blocks = (total + threads - 1) / threads;
        
        // 假设输入是FP16（根据jiuge模型的数据类型）
        bool is_fp16 = true;  // 可以根据实际需要修改
        
        // 步骤1：计算激活的 max(|x|)
        float* d_block_max = nullptr;
        err = cudaMalloc(&d_block_max, blocks * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(quantized_A);
            return err;
        }

        size_t shm_size = threads * sizeof(float);
        reduce_max_abs_fp16_kernel<<<blocks, threads, shm_size, stream>>>(
            static_cast<const __half*>(A),
            d_block_max,
            total
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_block_max);
            cudaFree(quantized_A);
            return err;
        }

        // 将每个block的最大值拷回主机并求全局最大
        std::vector<float> h_block_max(blocks, 0.0f);
        err = cudaMemcpyAsync(h_block_max.data(), d_block_max, blocks * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            cudaFree(d_block_max);
            cudaFree(quantized_A);
            return err;
        }
        // 同步以确保数据可读
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            cudaFree(d_block_max);
            cudaFree(quantized_A);
            return err;
        }
        float max_abs = 0.0f;
        for (int i = 0; i < blocks; ++i) {
            if (h_block_max[i] > max_abs) max_abs = h_block_max[i];
        }
        cudaFree(d_block_max);

        // 步骤2：计算对称量化scale
        act_scale_value = (max_abs > 0.0f) ? (max_abs / 127.0f) : (1.0f / 127.0f);
        
        // 将scale复制到设备
        float* d_scales = nullptr;
        cudaError_t scale_err = cudaMalloc(&d_scales, sizeof(float));
        if (scale_err != cudaSuccess) {
            cudaFree(quantized_A);
            return scale_err;
        }
        cudaMemcpyAsync(d_scales, &act_scale_value, sizeof(float), cudaMemcpyHostToDevice, stream);
        
        // 执行量化kernel
        quantize_activation_kernel<<<blocks, threads, 0, stream>>>(
            A,  // 输入激活（FP16或FP32）
            static_cast<int8_t*>(quantized_A),
            d_scales,
            M, K, N,
            is_fp16  // 是否为FP16输入
        );
        
        // 检查kernel执行是否成功
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_scales);
            cudaFree(quantized_A);
            return err;
        }
        
        cudaFree(d_scales);
    }

    // 创建AWQDescriptor并执行计算
    AWQDescriptor desc;
    desc._dtype = INFINI_DTYPE_I8;  // 使用INT8量化
    desc._device_id = device;
    
    // 设置矩阵信息
    desc._info.m = M;
    desc._info.n = N;
    desc._info.k = K;
    desc._info.batch = 1;
    desc._info.is_transed = false;
    
    // 设置矩阵布局 - 确保满足INT32计算的4字节对齐要求
    // 对于CUBLAS_COMPUTE_32I，leading dimension必须是4的倍数
	// 设置矩阵布局 - 对 INT8 计算要求K、N按4对齐更友好（以元素为单位）
	auto align_to_4 = [](int value) { return (value + 3) & ~3; };

	desc._info.a_matrix.row_stride = 1;
	desc._info.a_matrix.leading_dim = align_to_4(K);
	desc._info.a_matrix.stride = 0; // 单批次

	desc._info.b_matrix.row_stride = 1;
	desc._info.b_matrix.leading_dim = align_to_4(N);
	desc._info.b_matrix.stride = 0;

	desc._info.c_matrix.row_stride = 1;
	desc._info.c_matrix.leading_dim = align_to_4(N);
	desc._info.c_matrix.stride = 0;

    // 去除多余输出

    // 分配INT32输出缓冲区（天然4字节对齐）
    void* int32_output = nullptr;
    size_t int32_size = M * N * sizeof(int32_t);
    cudaError_t int32_err = cudaMalloc(&int32_output, int32_size);
    if (int32_err != cudaSuccess) {
        // std::cout << "CUDA AWQ GEMM: ERROR - Failed to allocate INT32 output buffer: " << cudaGetErrorString(int32_err) << std::endl;
        cudaFree(quantized_A);
        return int32_err;
    }


    // 执行计算 - 使用量化后的激活（A）与列主序[K,N]的INT8权重（B）
    infiniStatus_t status = desc.calculate(
        nullptr,  // workspace
        0,        // workspace_size
        int32_output,  // 输出到INT32缓冲区
        beta,     // beta
        quantized_A,    // a = 激活 INT8（行主序[M,K]）
        B,              // b = 权重 INT8（列主序[K,N]）
        alpha,    // alpha
        stream    // stream
    );

    if (status != INFINI_STATUS_SUCCESS) {
        std::cout << "CUDA AWQ GEMM: ERROR - GEMM calculation failed" << std::endl;
        cudaFree(int32_output);
        cudaFree(quantized_A);
        return cudaErrorUnknown;
    }

	// 反量化（per-channel）并融合 alpha/beta 到 FP32 输出
	{
		int threads = 256;
		int total = M * N;
		int blocks = (total + threads - 1) / threads;

		// 使用动态计算得到的激活scale（与量化时一致）
		float act_scale = act_scale_value;

		// 去除多余输出

		awq_dequantize_kernel<<<blocks, threads, 0, stream>>>(
			static_cast<const int32_t*>(int32_output),
			static_cast<float*>(C),
			act_scale,
			static_cast<const float*>(B_scales),
			alpha, beta,
			M, N
		);

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			// std::cout << "CUDA AWQ GEMM: ERROR - Dequantization kernel failed: " << cudaGetErrorString(err) << std::endl;
			cudaFree(int32_output);
			cudaFree(quantized_A);
			return err;
		}
	}

    // 释放临时缓冲区
    cudaFree(int32_output);
    cudaFree(quantized_A);

    // 去除多余输出
    return cudaSuccess;
}

// 新的接口函数，支持权重名称和量化参数管理器
cudaError_t launchAWQQuantizedGemmWithName(
    const void* A,
    const void* B,
    const void* B_scales,
    const std::string& weight_name,
    void* C,
    int M, int K, int N,
    float alpha, float beta,
    cudaStream_t stream) {
    
    // 从量化参数管理器获取参数
    auto* quant_manager = awq_gemm::getQuantizationManager();
    AWQQuantizeParams awq_params;
    
    if (quant_manager && quant_manager->hasParams(weight_name)) {
        awq_params = quant_manager->createAWQParams(weight_name);
        // 去除多余输出
    } else {
        // 使用默认参数
        awq_params.scale = 1.0f;
        awq_params.group_size = 128;
        awq_params.per_channel = false;
        // 去除多余输出
    }
    
    return launchAWQQuantizedGemm(A, B, B_scales, awq_params, C, M, K, N, alpha, beta, stream);
}

// GPU内存管理函数
cudaError_t allocateAWQParams(
    const AWQQuantizeParams& host_params,
    CudaAWQParams& device_params) {
    
    device_params.global_scale = host_params.scale;
    device_params.group_size = host_params.group_size;
    device_params.per_channel = host_params.per_channel;
    device_params.num_groups = host_params.group_scales.size();
    
    if (device_params.per_channel && device_params.num_groups > 0) {
        cudaError_t err = cudaMalloc(&device_params.group_scales, 
                                    device_params.num_groups * sizeof(float));
        if (err != cudaSuccess) {
            return err;
        }
        
        err = cudaMemcpy(device_params.group_scales, 
                        host_params.group_scales.data(),
                        device_params.num_groups * sizeof(float),
                        cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(device_params.group_scales);
            return err;
        }
    } else {
        device_params.group_scales = nullptr;
    }
    
    return cudaSuccess;
}

cudaError_t freeAWQParams(CudaAWQParams& device_params) {
    if (device_params.group_scales != nullptr) {
        cudaFree(device_params.group_scales);
        device_params.group_scales = nullptr;
    }
    return cudaSuccess;
}

// 性能选择器
awq_gemm::cuda::AWQKernelSelector::KernelType awq_gemm::cuda::AWQKernelSelector::selectOptimalKernel(
    int M, int K, int N, int device_id) {
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    // 现在总是使用cublas实现
    return STANDARD_KERNEL;
}

} // namespace cuda
} // namespace awq_gemm 