#ifndef JIUGE_WEIGHT_HPP
#define JIUGE_WEIGHT_HPP

#include "jiuge_impl.hpp"

#include <cmath>
inline std::shared_ptr<Tensor> getInEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}

inline std::shared_ptr<Tensor> getOutNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getOutEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape);
    }
}

inline std::shared_ptr<Tensor> getAttnNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnQKV(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(w->dt_qkv);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_qkv, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, (nh + 2 * nkvh) / ndev * dh});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_qkv, shape);
    }
}

inline std::shared_ptr<Tensor> getAttnQKVBias(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * dsize(w->dt_mat);
    auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh});
    return Tensor::weight((char *)(w->attn_qkv_b[layer]) + offset, w->dt_mat, shape);
}

inline std::shared_ptr<Tensor> getAttnO(JiugeMeta const *meta,
                                        JiugeWeights const *w, size_t layer,
                                        size_t idev, size_t ndev) {
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * d * (nh / ndev * dh) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, nh / ndev * dh});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({nh / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getFFNGateUp(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * (2 * di / ndev) * d * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({2 * di / ndev, d});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, 2 * di / ndev});
        return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                              w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNDown(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * d * (di / ndev) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, di / ndev});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({di / ndev, d});
        return Tensor::weight((char *)(w->ffn_down[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getSinTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(JiugeMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

// 新增：创建cublas可直接使用的列主序INT8权重 [K, N]
inline std::shared_ptr<Tensor> getAttnQKVPacked4Byte(
	JiugeMeta const *meta,
	JiugeWeights const *w,
	size_t layer, size_t idev, size_t ndev,
	std::shared_ptr<MemoryPool> memory_pool) {
	// 原始权重（期望为 INT8），形状 [K, N]（K=d, N=(nh+2*nkvh)/ndev*dh）
	auto original_weight = getAttnQKV(meta, w, layer, idev, ndev);
	if (!original_weight) {
		std::cout << "[权重打包] 错误：获取原始权重失败" << std::endl;
		return nullptr;
	}
	// 仅处理 INT8/BYTE；若非 INT8 则原样返回
	if (original_weight->dtype() != INFINI_DTYPE_I8 && original_weight->dtype() != INFINI_DTYPE_BYTE) {
		std::cout << "[权重打包] 层 " << layer << " 不是INT8，直接返回原始权重" << std::endl;
		return original_weight;
	}
	auto shape = original_weight->shape();
	if (shape.size() != 2) {
		std::cout << "[权重打包] 错误：权重形状维度不是2" << std::endl;
		return original_weight;
	}
	size_t K = shape[0];
	size_t N = shape[1];
	std::cout << "[权重打包] 层 " << layer << " 原始权重形状: [" << K << ", " << N << "], dtype=" << (original_weight->dtype()==3? "INFINI_DTYPE_I8" :"INFINI_DTYPE_BYTE") << std::endl;
	// 目标：构造列主序的 INT8 [K, N]：每列连续存放K个元素
	auto packed_weight = Tensor::buffer(INFINI_DTYPE_I8, {K, N}, memory_pool);
	if (!packed_weight) {
		std::cout << "[权重打包] 错误：创建packed权重失败" << std::endl;
		return original_weight;
	}
	// 将原始[行主序 KxN]重排为[列主序 KxN]
	const void* src_dev_ptr = original_weight->data();
	void* dst_dev_ptr = packed_weight->data();
	const size_t bytes = K * N * sizeof(uint8_t);
	// 拷贝到主机
	std::vector<uint8_t> host_src(bytes);
	if (original_weight->deviceType() != INFINI_DEVICE_CPU) {
		RUN_INFINI(infinirtMemcpy(host_src.data(), src_dev_ptr, bytes, INFINIRT_MEMCPY_D2H));
	} else {
		std::memcpy(host_src.data(), src_dev_ptr, bytes);
	}
	// 重排到列主序缓冲
	std::vector<uint8_t> host_dst(bytes);
	bool is_signed = (original_weight->dtype() == INFINI_DTYPE_I8);
	if (!is_signed) {
		std::cout << "[权重打包] 警告：源为无符号BYTE，按原值搬运到INT8缓冲，请确保量化流程一致" << std::endl;
	}
	for (size_t row = 0; row < K; ++row) {
		for (size_t col = 0; col < N; ++col) {
			// 原 row-major 索引: row * N + col
			// 目的 col-major 索引（列优先存储）: col * K + row
			host_dst[col * K + row] = host_src[row * N + col];
		}
	}
	// 回写到设备
	if (packed_weight->deviceType() != INFINI_DEVICE_CPU) {
		RUN_INFINI(infinirtMemcpy(dst_dev_ptr, host_dst.data(), bytes, INFINIRT_MEMCPY_H2D));
	} else {
		std::memcpy(dst_dev_ptr, host_dst.data(), bytes);
	}
	std::cout << "[权重打包] 层 " << layer << " 已重排为列主序INT8权重 [K=" << K << ", N=" << N << "] (适配cublas)" << std::endl;
	return packed_weight;
}

#endif
