#include "jiuge_impl.hpp"
#include "jiuge_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer.h"
#include "../../quantization/awq_gemm.hpp"
#include "../../quantization/awq_quantization_params.hpp"
#include "../../3rdparty/InfiniCore/include/infiniop/ops/quantize.h"

#include <random>
#include <thread>
#include <vector>

// 预加载量化参数的结构体
struct QuantizationParams {
    std::vector<awq_gemm::AWQQuantizeParams> awq_params_list;
    std::vector<std::shared_ptr<Tensor>> weight_scales_list;
    std::vector<bool> has_quant_params;
};

// 预加载量化参数的函数
QuantizationParams preloadQuantizationParams(const JiugeMeta &meta, DeviceResource &rsrc,
                                            uint32_t nlayer, bool enable_quantization) {
    QuantizationParams params;
    if (enable_quantization && awq_gemm::isAWQGemmSupported(rsrc.device)) {
        auto* quant_manager = awq_gemm::getQuantizationManager();
        // 预加载所有层的量化参数
        for (uint32_t layer = 0; layer < nlayer; layer++) {
            // 只查找分开的q、k、v权重名称
            std::string q_weight_name = "model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight";
            std::string k_weight_name = "model.layers." + std::to_string(layer) + ".self_attn.k_proj.weight";
            std::string v_weight_name = "model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight";
            // 这里只处理q_proj，后续如需k/v可自行扩展
            // std::string target_weight_name = q_weight_name;
            std::vector<std::string> target_weight_names = {q_weight_name, k_weight_name, v_weight_name};
            for (const auto& target_weight_name : target_weight_names) {
                if (quant_manager && quant_manager->hasParams(target_weight_name)) {
                    auto awq_params = quant_manager->createAWQParams(target_weight_name);
                    params.awq_params_list.push_back(awq_params);
                    params.has_quant_params.push_back(true);
                    // 按层输出scale
                    std::cout << "[量化参数] 层 " << layer << " 权重: " << target_weight_name
                              << " scale: ";
                    if (awq_params.per_channel && !awq_params.group_scales.empty()) {
                        std::cout << "[";
                        for (size_t i = 0; i < awq_params.group_scales.size(); ++i) {
                            std::cout << awq_params.group_scales[i];
                            if (i != awq_params.group_scales.size() - 1) std::cout << ", ";
                        }
                        std::cout << "]";
                    } else {
                        std::cout << awq_params.scale << std::endl;
                    }
                    // 预分配权重缩放因子缓冲区
                    auto weight_scales = Tensor::buffer(INFINI_DTYPE_F32, {rsrc.w_attn_qkv[layer]->shape()[1]}, rsrc.memory_pool);
                    params.weight_scales_list.push_back(weight_scales);
                    // 设置缩放因子
                    auto* scale_data = static_cast<float*>(weight_scales->data());
                    if (rsrc.device == INFINI_DEVICE_CPU) {
                        if (awq_params.per_channel && !awq_params.group_scales.empty()) {
                            std::copy(awq_params.group_scales.begin(), awq_params.group_scales.end(), scale_data);
                        } else {
                            std::fill_n(scale_data, weight_scales->shape()[0], awq_params.scale);
                        }
                    } else {
                        std::vector<float> temp_scales;
                        if (awq_params.per_channel && !awq_params.group_scales.empty()) {
                            temp_scales = awq_params.group_scales;
                        } else {
                            temp_scales.resize(weight_scales->shape()[0], awq_params.scale);
                        }
                        RUN_INFINI(infinirtMemcpyAsync(scale_data, temp_scales.data(),
                                                      temp_scales.size() * sizeof(float),
                                                      INFINIRT_MEMCPY_H2D, rsrc.stream));
                    }
                } else {
                    // 没有量化参数，使用默认参数
                    auto awq_params = awq_gemm::createAWQParams(nullptr, 128, false);
                    params.awq_params_list.push_back(awq_params);
                    params.has_quant_params.push_back(false);
                    auto weight_scales = Tensor::buffer(INFINI_DTYPE_F32, {rsrc.w_attn_qkv[layer]->shape()[1]}, rsrc.memory_pool);
                    params.weight_scales_list.push_back(weight_scales);
                    auto* scale_data = static_cast<float*>(weight_scales->data());
                    if (rsrc.device == INFINI_DEVICE_CPU) {
                        std::fill_n(scale_data, weight_scales->shape()[0], 1.0f / 127.0f);
                    } else {
                        std::vector<float> temp_scales(weight_scales->shape()[0], 1.0f / 127.0f);
                        RUN_INFINI(infinirtMemcpyAsync(scale_data, temp_scales.data(),
                                                      temp_scales.size() * sizeof(float),
                                                      INFINIRT_MEMCPY_H2D, rsrc.stream));
                    }
                }
            }
        }
    }
    return params;
}


void createDeviceResource(DeviceResource *rsrc, const JiugeMeta *meta,
                          const JiugeWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm, bool enable_quantization) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        if (enable_quantization && awq_gemm::isAWQGemmSupported(device)) {
            std::cout << "[pack] layer " << layer << std::endl;
            w_attn_qkv.push_back(
                getAttnQKVPacked4Byte(meta, weights, layer, idev, ndev, memory_pool)
            );
        } else {
            w_attn_qkv.push_back(
                getAttnQKV(meta, weights, layer, idev, ndev));
        }
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
        w_ffn_gate_up.push_back(
            getFFNGateUp(meta, weights, layer, idev, ndev));
        w_ffn_down.push_back(
            getFFNDown(meta, weights, layer, idev, ndev));
    }

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        getInEmbd(meta, weights),
        getOutNorm(meta, weights),
        getOutEmbd(meta, weights),
        getSinTable(meta),
        getCosTable(meta),
        w_attn_norm,
        w_attn_qkv,
        b_attn_qkv,
        w_attn_out,
        w_ffn_norm,
        w_ffn_gate_up,
        w_ffn_down,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(DeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();
    for (auto &t : res.w_attn_norm) {
        t.reset();
    }
    res.w_attn_norm.clear();
    for (auto &t : res.w_attn_qkv) {
        t.reset();
    }
    res.w_attn_qkv.clear();
    for (auto &t : res.b_attn_qkv) {
        t.reset();
    }
    res.b_attn_qkv.clear();
    for (auto &t : res.w_attn_out) {
        t.reset();
    }
    res.w_attn_out.clear();
    for (auto &t : res.w_ffn_norm) {
        t.reset();
    }
    res.w_ffn_norm.clear();
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();
    }
    res.w_ffn_gate_up.clear();
    for (auto &t : res.w_ffn_down) {
        t.reset();
    }
    res.w_ffn_down.clear();
    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void inferDeviceBatch(const JiugeMeta &meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, bool enable_quantization, const QuantizationParams &quant_params) {
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;
    // bool use_awq = enable_quantization;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Prepare operators and workspace
    size_t workspace_size = 0, temp_size = 0;
    // attn & mlp rmsnorm
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc(),
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopQuantizeDescriptor_t desc_quant;
    RUN_INFINI(infiniopCreateQuantizeDescriptor(
        rsrc.handle, &desc_quant, logits_in->desc(), logits_out->desc()));
    RUN_INFINI(infiniopGetQuantizeWorkspaceSize(desc_quant, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // Attention
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_o;
    infiniopRearrangeDescriptor_t desc_qkv_bias;
    if (has_qkv_bias) {
        RUN_INFINI(infiniopCreateRearrangeDescriptor(
            rsrc.handle, &desc_qkv_bias, qkv_buf->desc(),
            TensorDesc::create(dt_logits, {ntok, (nh + nkvh * 2) * dh}, {0, 1})->desc()));
    }
    // TODO: modify Descriptor when use awq
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_qkv, qkv_buf->desc(),
        logits_in->desc(), rsrc.w_attn_qkv[0]->desc()));
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_o, logits_in->desc(),
        o_buf->desc(), rsrc.w_attn_out[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_o, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    qkv_buf->dimSplit(1, {nh + nkvh * 2, dh}); // (ntok, nh + 2 * nkvh, dh)
    auto qkv_buf_q = qkv_buf->slice(1, 0, nh);
    auto qkv_buf_k = qkv_buf->slice(1, nh, nkvh);
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_q, qkv_buf_q->desc(), qkv_buf_q->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_k, qkv_buf_k->desc(), qkv_buf_k->desc(),
        pos_ids_buf->desc(), rsrc.sin_table->desc(),
        rsrc.cos_table->desc()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // attention inner
    auto desc_kv_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_q_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    auto desc_qk_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_qk_softmaxs = std::vector<infiniopCausalSoftmaxDescriptor_t>(nreq);
    auto desc_attn_v_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
    auto desc_attn_v_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
    size_t token_offset = 0;
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;
    o_buf->dimSplit(1, {nh, dh});
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;
        auto o = o_buf->slice({{0, token_offset, seq_len}});
        auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
        auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
        // auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
        // kv cache tensors can share the same descriptor
        // [nkvh, dh, total_len]
        auto full_kv = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);

        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv->desc(), k->desc()));

        // [nkvh, ngroup, seq_len, dh]
        q->dimSplit(1, {nkvh, ngroup})->permute({1, 2, 0, 3});
        auto q_t = TensorDesc::create(dt_logits, {nkvh, ngroup, seq_len, dh});
        // [seq_len, nkvh, ngroup, dh] -> [nkvh, ngroup, seq_len, dh]
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_q_rearranges[req],
                                                     q_t->desc(), q->desc()));
        // [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh]
        auto attn_v_t = q_t;
        auto attn_v = TensorDesc::createWithOrder(dt_logits, {nkvh, ngroup, seq_len, dh}, {1, 2, 0, 3});
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_attn_v_rearranges[req],
                                                     attn_v->desc(), attn_v_t->desc()));
        q_t = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, dh});
        auto qk = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, total_len});
        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_qk_gemms[req], qk->desc(), q_t->desc(), full_kv->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qk_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // [nkvh, total_len, dh]
        auto full_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_attn_v_gemms[req], q_t->desc(), qk->desc(), full_v->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        qk = TensorDesc::create(dt_logits, {nkvh * ngroup, seq_len, total_len});
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc.handle, &desc_qk_softmaxs[req], qk->desc(), qk->desc()));
        RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        token_offset += seq_len;
    }
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    // MLP descriptors
    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(),
        logits_out->desc(), rsrc.w_ffn_gate_up[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down, logits_in->desc(),
        gate_buf->desc(), rsrc.w_ffn_down[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Output and sample
    infiniopRMSNormDescriptor_t desc_norm_out;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->slice(0, 0, 1)->desc(),
        logits_out->slice(0, 0, 1)->desc(),
        rsrc.w_out_norm->desc(), meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopGemmDescriptor_t desc_out_embd;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_out_embd, prob_buf->desc(),
        logits_out->slice(0, 0, nreq)->desc(),
        rsrc.w_out_embd->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_out_embd, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRandomSampleDescriptor_t desc_sample;
    RUN_INFINI(infiniopCreateRandomSampleDescriptor(
        rsrc.handle, &desc_sample,
        TensorDesc::create(INFINI_DTYPE_I64, {}, {})->desc(),
        TensorDesc::create(dt_logits, {dvoc}, {1})->desc()));
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc_sample, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // Allocate workspace
    std::shared_ptr<Storage> workspace_storage = Storage::createFromPool(workspace_size, rsrc.memory_pool);
    void *workspace = workspace_storage->memory();

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));
        // qkv_proj
        if (has_qkv_bias) {
            RUN_INFINI(infiniopRearrange(
                desc_qkv_bias,
                qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        }
        // TODO: quant
        RUN_INFINI(infiniopQuantize(
            desc_quant, workspace, workspace_size,
            logits_in->data(), logits_out->data(), stream));

        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, has_qkv_bias ? 1.0 : 0.0, stream));
        // rope
        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            qkv_buf->data(), qkv_buf->data(),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(), stream));
        RUN_INFINI(infiniopRoPE(
            desc_rope_k, workspace, workspace_size,
            qkv_buf->data(nh * dh), qkv_buf->data(nh * dh),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(),
            stream));

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto o = o_buf->slice({{0, token_offset, seq_len}});
            auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
            auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
            // self attention
            // concat
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),
                k->data(), stream));
            RUN_INFINI(infiniopRearrange(
                desc_kv_rearranges[req],
                kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),
                v->data(), stream));
            // qk
            RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
            RUN_INFINI(infiniopGemm(
                desc_qk_gemms[req], workspace, workspace_size,
                qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 1. / sqrt(dh), 0.0, stream));
            // softmax
            RUN_INFINI(infiniopCausalSoftmax(
                desc_qk_softmaxs[req], workspace, workspace_size,
                qk_buf->data(), qk_buf->data(), stream));
            // attn val
            RUN_INFINI(infiniopGemm(
                desc_attn_v_gemms[req], workspace, workspace_size,
                attn_val_buf->data(), qk_buf->data(), kv_caches[req]->v[idev][layer]->data(), 1.0, 0.0, stream));
            // rearrange attn val
            RUN_INFINI(infiniopRearrange(
                desc_attn_v_rearranges[req],
                o->data(),
                attn_val_buf->data(), stream));

            token_offset += seq_len;
        }
        // o_proj
        RUN_INFINI(infiniopGemm(
            desc_attn_o, workspace, workspace_size,
            logits_in->data(), o_buf->data(),
            rsrc.w_attn_out[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. FFN
        // rms_norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer]->data(),
            1.0, 0.0, stream));
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_in->data(), gate_buf->data(),
            rsrc.w_ffn_down[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    // Sample and Output
    if (idev == 0) {
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            token_offset += seq_len;
            RUN_INFINI(infiniopRMSNorm(
                desc_norm_out, workspace, workspace_size,
                logits_out->data(req * d),
                logits_in->data((token_offset - 1) * d),
                rsrc.w_out_norm->data(), stream));
        }
        RUN_INFINI(infiniopGemm(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(), logits_out->data(),
            rsrc.w_out_embd->data(), 1.0, 0.0, stream));
        std::random_device _rd;
        std::mt19937 gen(_rd());
        token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
            // prob_buf->debug();
            RUN_INFINI(infiniopRandomSample(
                desc_sample, workspace, workspace_size,
                result_buf->data(req),
                prob_buf->data(req * dvoc),
                random_val,
                topp[req], topk[req], temperature[req],
                stream));
            // result_buf->debug();
            token_offset += seq_len;
        }
        RUN_INFINI(infinirtStreamSynchronize(stream));
        RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                  sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
        for (uint32_t req = 0; req < nreq; req++) {
            output[req] = result_cpu[req];
        }
    }

    // Clean up
    infiniopDestroyRMSNormDescriptor(desc_norm);
    infiniopDestroyQuantizeDescriptor(desc_quant);
    if (has_qkv_bias) {
        infiniopDestroyRearrangeDescriptor(desc_qkv_bias);
    }
    infiniopDestroyGemmDescriptor(desc_attn_qkv);
    infiniopDestroyGemmDescriptor(desc_attn_o);
    infiniopDestroyRoPEDescriptor(desc_rope_q);
    infiniopDestroyRoPEDescriptor(desc_rope_k);
    for (uint32_t req = 0; req < nreq; req++) {
        infiniopDestroyRearrangeDescriptor(desc_kv_rearranges[req]);
        infiniopDestroyRearrangeDescriptor(desc_q_rearranges[req]);
        infiniopDestroyGemmDescriptor(desc_qk_gemms[req]);
        infiniopDestroyCausalSoftmaxDescriptor(desc_qk_softmaxs[req]);
        infiniopDestroyGemmDescriptor(desc_attn_v_gemms[req]);
        infiniopDestroyRearrangeDescriptor(desc_attn_v_rearranges[req]);
    }
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);
    infiniopDestroySwiGLUDescriptor(desc_swiglu);
    infiniopDestroyGemmDescriptor(desc_ffn_down);
    infiniopDestroyRMSNormDescriptor(desc_norm_out);
    infiniopDestroyGemmDescriptor(desc_out_embd);
    infiniopDestroyRandomSampleDescriptor(desc_sample);
}

__C void
inferBatch(struct JiugeModel *model,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm, bool enable_quantization) {
    // Create Device Resource
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm, enable_quantization);
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    QuantizationParams quant_params;
    if (enable_quantization)
        quant_params = preloadQuantizationParams(meta, *rsrc, meta.nlayer, enable_quantization);

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output, enable_quantization, quant_params);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
}

JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids, bool enable_quantization) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i], enable_quantization);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct JiugeModel *
createJiugeModel(const JiugeMeta *meta,
                 const JiugeWeights *weights,
                 const char *model_path,
                 bool enable_quantization,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    JiugeModel *model = new JiugeModel(meta, weights, device, device_ids, enable_quantization);
    return model;
}

__C void destroyJiugeModel(struct JiugeModel *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}
