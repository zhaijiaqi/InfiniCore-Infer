#include "jiuge_impl.hpp"
#include "jiuge_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer.h"

// 添加量化相关头文件
#include "../../quantization/quantiUtil.hpp"
#include "../../quantization/quantiOp.hpp"

#include <random>
#include <thread>
#include <vector>

// 量化参数存储结构
struct QuantizationState {
    // 激活值量化参数
    std::vector<Quantization::QuantizationParams> activation_params;
    
    // 权重量化参数（每层存储）
    std::vector<Quantization::QuantizationParams> qkv_weight_params;
    std::vector<Quantization::QuantizationParams> attn_out_weight_params;
    std::vector<Quantization::QuantizationParams> ffn_gate_up_weight_params;
    std::vector<Quantization::QuantizationParams> ffn_down_weight_params;
    
    // KV缓存量化参数
    std::vector<std::vector<Quantization::QuantizationParams>> kv_cache_params;  // [layer][k/v]
    
    // 量化缓冲区
    std::shared_ptr<Tensor> qkv_quantized;
    std::shared_ptr<Tensor> gate_up_quantized;
    std::shared_ptr<Tensor> q_quantized, k_quantized, v_quantized;
    
    bool enable_quantization = false;  // 量化开关 - 完全禁用以确保稳定性
};

// 创建设备资源的函数，用于初始化大模型推理所需的所有资源
// rsrc: 输出参数，用于存储创建的设备资源
// meta: 模型的元数据信息（如层数、维度等）
// weights: 模型的权重数据
// device: 设备类型（CPU/GPU等）
// idev: 当前设备在多设备中的索引
// ndev: 总设备数量
// dev_id: 设备ID
// comm: 通信句柄，用于多设备间的通信
void createDeviceResource(DeviceResource *rsrc, const JiugeMeta *meta,
                          const JiugeWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    // 设置当前要使用的设备
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    
    // 创建操作句柄，用于执行各种张量操作
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    
    // 创建流，用于异步执行操作
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    // 准备存储各层权重的容器
    // 在Transformer架构中，每一层都有相同结构的权重
    std::vector<std::shared_ptr<Tensor>> w_attn_norm,    // 注意力层的归一化权重
                                         w_attn_qkv,     // 注意力机制的查询、键、值权重
                                         b_attn_qkv,     // 注意力机制的偏置（如果存在）
                                         w_attn_out,     // 注意力输出权重
                                         w_ffn_norm,     // 前馈网络的归一化权重
                                         w_ffn_gate_up,  // 前馈网络的门控和上投影权重
                                         w_ffn_down;     // 前馈网络的下投影权重
    
    // 为每一层加载对应的权重
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        // 获取注意力层归一化权重
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        
        // 获取注意力QKV权重（在多设备情况下进行分片）
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));
        
        // 如果存在QKV偏置，则加载它
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }
        
        // 获取注意力输出权重
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
        
        // 获取前馈网络归一化权重
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
        
        // 获取前馈网络的门控和上投影权重
        w_ffn_gate_up.push_back(
            getFFNGateUp(meta, weights, layer, idev, ndev));
        
        // 获取前馈网络的下投影权重
        w_ffn_down.push_back(
            getFFNDown(meta, weights, layer, idev, ndev));
    }

    // 创建内存池，用于高效管理张量内存（128MB初始大小）
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    // 初始化量化状态
    auto quant_state = std::make_shared<QuantizationState>();
    
    // 初始化各层的量化参数
    quant_state->qkv_weight_params.resize(meta->nlayer);
    quant_state->attn_out_weight_params.resize(meta->nlayer);
    quant_state->ffn_gate_up_weight_params.resize(meta->nlayer);
    quant_state->ffn_down_weight_params.resize(meta->nlayer);
    quant_state->kv_cache_params.resize(meta->nlayer);
    
    // 为每层的KV缓存初始化量化参数
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        quant_state->kv_cache_params[layer].resize(2);  // K和V两个缓存
    }
    

    // 将所有资源组装到DeviceResource结构体中
    *rsrc = DeviceResource{
        device,                          // 设备类型
        dev_id,                         // 设备ID
        handle,                         // 操作句柄
        getInEmbd(meta, weights),       // 输入词嵌入权重
        getOutNorm(meta, weights),      // 输出归一化权重
        getOutEmbd(meta, weights),      // 输出词嵌入权重
        getSinTable(meta),              // 位置编码的正弦表
        getCosTable(meta),              // 位置编码的余弦表
        w_attn_norm,                    // 所有层的注意力归一化权重
        w_attn_qkv,                     // 所有层的注意力QKV权重
        b_attn_qkv,                     // 所有层的注意力QKV偏置
        w_attn_out,                     // 所有层的注意力输出权重
        w_ffn_norm,                     // 所有层的前馈网络归一化权重
        w_ffn_gate_up,                  // 所有层的前馈网络门控权重
        w_ffn_down,                     // 所有层的前馈网络下投影权重
        stream,                         // 计算流
        comm,                           // 通信句柄
        memory_pool,                    // 内存池
        quant_state,                    // 量化状态
    };
    
    // 同步设备，确保所有初始化操作完成
    RUN_INFINI(infinirtDeviceSynchronize());
}

// 释放设备资源的函数，用于清理大模型推理时创建的所有资源
// 这个函数的作用是防止内存泄漏，确保程序结束时所有GPU/CPU资源都被正确释放
// res: 要释放的设备资源结构体，包含了模型的所有权重、缓冲区和操作句柄
void releaseDeviceResource(DeviceResource &res) {
    // 同步设备，确保所有正在执行的操作都完成后再释放资源
    // 这很重要，因为如果还有操作在使用这些资源时就释放了，会导致程序崩溃
    infinirtDeviceSynchronize();
    
    // 释放模型的各种权重张量
    // 这些权重是模型的核心参数，占用大量显存/内存
    
    // 释放词嵌入相关权重
    res.w_in_embd.reset();      // 输入词嵌入权重（将token转换为向量）
    res.w_out_norm.reset();     // 输出层归一化权重（稳定训练的技术）
    res.w_out_embd.reset();     // 输出词嵌入权重（将向量转换回token概率）
    
    // 释放位置编码表
    res.sin_table.reset();      // 正弦位置编码表（帮助模型理解token的位置）
    res.cos_table.reset();      // 余弦位置编码表（帮助模型理解token的位置）
    
    // 释放每一层的注意力机制权重
    // 注意力机制是Transformer的核心，让模型能够关注输入中的重要部分
    for (auto &t : res.w_attn_norm) {
        t.reset();              // 释放每层的注意力归一化权重
    }
    res.w_attn_norm.clear();    // 清空容器本身
    
    for (auto &t : res.w_attn_qkv) {
        t.reset();              // 释放每层的查询(Q)、键(K)、值(V)权重
    }
    res.w_attn_qkv.clear();     // 清空容器本身
    
    for (auto &t : res.b_attn_qkv) {
        t.reset();              // 释放每层的QKV偏置项（如果存在的话）
    }
    res.b_attn_qkv.clear();     // 清空容器本身
    
    for (auto &t : res.w_attn_out) {
        t.reset();              // 释放每层的注意力输出权重
    }
    res.w_attn_out.clear();     // 清空容器本身
    
    // 释放每一层的前馈神经网络(FFN)权重
    // FFN是Transformer的另一个重要组件，负责非线性变换
    for (auto &t : res.w_ffn_norm) {
        t.reset();              // 释放每层的FFN归一化权重
    }
    res.w_ffn_norm.clear();     // 清空容器本身
    
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();              // 释放每层的FFN门控和上投影权重
    }
    res.w_ffn_gate_up.clear();  // 清空容器本身
    
    for (auto &t : res.w_ffn_down) {
        t.reset();              // 释放每层的FFN下投影权重
    }
    res.w_ffn_down.clear();     // 清空容器本身
    
    // 销毁计算相关的句柄和资源
    // 这些是底层的计算库资源，必须正确释放
    
    infiniopDestroyHandle(res.handle);  // 销毁操作句柄（用于执行矩阵运算等）
    res.handle = nullptr;               // 置空指针，防止误用
    
    infinirtStreamDestroy(res.stream);  // 销毁计算流（用于异步执行操作）
    res.stream = nullptr;               // 置空指针，防止误用
    
    infinicclCommDestroy(res.comm);     // 销毁通信句柄（用于多GPU间通信）
    res.comm = nullptr;                 // 置空指针，防止误用
}
// 设备批处理推理函数 - 处理多个请求的并行推理
// 该函数是模型推理的核心，执行完整的Transformer前向传播过程
void inferDeviceBatch(const JiugeMeta &meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output) {
    // 从模型元数据中提取关键参数
    auto nlayer = meta.nlayer;          // Transformer层数
    auto nkvh = meta.nkvh / ndev;       // 每个设备的K/V头数
    auto nh = meta.nh / ndev;           // 每个设备的注意力头数
    auto ngroup = nh / nkvh;            // 分组注意力中每组的头数
    // auto dctx = meta.dctx;
    auto dh = meta.dh;                  // 每个头的维度
    auto d = meta.d;                    // 模型隐藏层维度
    auto dt_logits = meta.dt_logits;    // 数据类型
    auto di = meta.di / ndev;           // 每个设备的中间层维度
    auto dvoc = meta.dvoc;              // 词汇表大小
    auto stream = rsrc.stream;          // 计算流
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;  // 是否有QKV偏置

    // 分配推理过程中需要的缓冲区
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);           // 输入logits
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);          // 输出logits
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);  // QKV缓冲区
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);    // FFN门控和上投影缓冲区
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);         // 注意力输出缓冲区
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);         // 概率分布缓冲区
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);      // 采样结果缓冲区
    auto result_cpu = std::vector<int64_t>(nreq);  // CPU端结果

    // 为每个请求准备位置编码
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    // 将位置编码数据传输到设备
    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    
    // 将token转换为词嵌入向量，这是Transformer的第一步
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // 准备各种操作的描述符和工作空间
    size_t workspace_size = 0, temp_size = 0;
    
    // RMSNorm描述符（用于层归一化）
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc(),
        logits_out->desc(), rsrc.w_attn_norm[0]->desc(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    workspace_size = std::max(workspace_size, temp_size);
    
    // 注意力机制相关描述符
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_o;  // QKV投影和输出投影
    infiniopRearrangeDescriptor_t desc_qkv_bias;
    if (has_qkv_bias) {
        RUN_INFINI(infiniopCreateRearrangeDescriptor(
            rsrc.handle, &desc_qkv_bias, qkv_buf->desc(),
            TensorDesc::create(dt_logits, {ntok, (nh + nkvh * 2) * dh}, {0, 1})->desc()));
    }
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
    
    // RoPE旋转位置编码描述符
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    qkv_buf->dimSplit(1, {nh + nkvh * 2, dh}); // (ntok, nh + 2 * nkvh, dh)
    auto qkv_buf_q = qkv_buf->slice(1, 0, nh);      // Query部分
    auto qkv_buf_k = qkv_buf->slice(1, nh, nkvh);   // Key部分
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
    
    // 为每个请求创建自注意力操作的描述符
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
    
    // 为每个请求设置注意力计算的参数
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];       // 历史序列长度
        auto seq_len = req_lens[req];       // 当前序列长度
        auto total_len = past_len + seq_len; // 总序列长度
        auto o = o_buf->slice({{0, token_offset, seq_len}});
        auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
        auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
        // auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
        
        // KV缓存张量可以共享相同的描述符
        // [nkvh, dh, total_len]
        auto full_kv = kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
        auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);

        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc.handle, &desc_kv_rearranges[req],
                                                     cache_kv->desc(), k->desc()));

        // 设置Q矩阵的形状变换：[nkvh, ngroup, seq_len, dh]
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
        
        // Q*K矩阵乘法描述符
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_qk_gemms[req], qk->desc(), q_t->desc(), full_kv->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qk_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // 注意力分数与V相乘的描述符
        // [nkvh, total_len, dh]
        auto full_v = kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});
        RUN_INFINI(infiniopCreateGemmDescriptor(
            rsrc.handle, &desc_attn_v_gemms[req], q_t->desc(), qk->desc(), full_v->desc()));
        RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        // Causal Softmax描述符（因果掩码的softmax）
        qk = TensorDesc::create(dt_logits, {nkvh * ngroup, seq_len, total_len});
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc.handle, &desc_qk_softmaxs[req], qk->desc(), qk->desc()));
        RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);

        token_offset += seq_len;
    }
    
    // 分配注意力计算的缓冲区
    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

    // 前馈神经网络(MLP/FFN)相关描述符
    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(),
        logits_out->desc(), rsrc.w_ffn_gate_up[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    auto gate_buf = gate_up_buf->slice(1, 0, di);       // 门控部分
    auto up_buf = gate_up_buf->slice(1, di, di);        // 上投影部分
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down, logits_in->desc(),
        gate_buf->desc(), rsrc.w_ffn_down[0]->desc()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // 输出层和采样相关描述符
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
    
    // 分配工作空间
    std::shared_ptr<Storage> workspace_storage = Storage::createFromPool(workspace_size, rsrc.memory_pool);
    void *workspace = workspace_storage->memory();

    // 开始计算：遍历每一个Transformer层
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. 自注意力机制
        // 层归一化
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));
        
        // QKV投影：将输入投影为Query、Key、Value三个矩阵
        // [量化点1] 在执行QKV矩阵乘法前，将激活值量化为INT8
        // 暂时禁用运行时量化，因为需要重新设计接口
        std::shared_ptr<Tensor> logits_out_quantized;
        Quantization::QuantizationParams activation_params;
        
        if (false && rsrc.quant_state->enable_quantization) {
            // 创建量化输出缓冲区
            auto shape = logits_out->shape();
            size_t total_elements = 1;
            for (auto dim : shape) {
                total_elements *= dim;
            }
            
            logits_out_quantized = Tensor::buffer(INFINI_DTYPE_I8, shape, rsrc.memory_pool);
            
            // 使用底层接口进行量化
            auto status = Quantization::quantize_activation_fp16_to_int8(
                logits_out_quantized->data(),          // INT8输出
                logits_out->data(),                    // FP16输入
                total_elements,                        // 元素数量
                activation_params.scale, 
                activation_params.zero_point);
            
            // 检查量化是否成功
            if (status != INFINI_STATUS_SUCCESS) {
                // 量化失败，fallback到FP16计算
                rsrc.quant_state->enable_quantization = false;
            }
        }
        
        if (has_qkv_bias) {
            RUN_INFINI(infiniopRearrange(
                desc_qkv_bias,
                qkv_buf->data(), rsrc.b_attn_qkv[layer]->data(), stream));
        }
    
        // std::cout << "quantization: " << rsrc.quant_state->enable_quantization << std::endl;
        
        if (rsrc.quant_state->enable_quantization) {
            // 使用混合精度GEMM：FP16激活 * INT8权重
            Quantization::mixed_precision_gemm(
                rsrc.handle, workspace, workspace_size,
                qkv_buf->data(),                                    // FP16输出
                logits_out->data(),                                 // FP16激活
                rsrc.w_attn_qkv[layer]->data(),                    // INT8权重（假设已量化）
                rsrc.quant_state->qkv_weight_params[layer].scale,
                rsrc.quant_state->qkv_weight_params[layer].zero_point,
                1.0, has_qkv_bias ? 1.0 : 0.0,
                logits_out->desc(), rsrc.w_attn_qkv[layer]->desc(), qkv_buf->desc(),
                stream);
        } else {
            // 原始FP16计算
            RUN_INFINI(infiniopGemm(
                desc_attn_qkv, workspace, workspace_size,
                qkv_buf->data(), logits_out->data(),
                rsrc.w_attn_qkv[layer]->data(), 1.0, has_qkv_bias ? 1.0 : 0.0, stream));
        }
        // [反量化点1] QKV矩阵乘法结果已经是FP16，无需反量化
        
        // 对Q和K应用旋转位置编码(RoPE)
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

        // 对每个请求执行自注意力计算
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto o = o_buf->slice({{0, token_offset, seq_len}});
            auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}});
            auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
            
            // 将当前的K、V缓存到KV Cache中
            // [量化点2] 可以选择将K、V缓存量化为INT8以节省显存
            // 暂时禁用KV缓存量化，需要重新设计接口
            if (false && rsrc.quant_state->enable_quantization) {
                // 获取K和V的元素数量
                auto k_shape = k->shape();
                auto v_shape = v->shape();
                size_t k_elements = 1, v_elements = 1;
                for (auto dim : k_shape) k_elements *= dim;
                for (auto dim : v_shape) v_elements *= dim;
                
                // 创建量化输出缓冲区
                auto k_quantized_buffer = Tensor::buffer(INFINI_DTYPE_I8, k_shape, rsrc.memory_pool);
                auto v_quantized_buffer = Tensor::buffer(INFINI_DTYPE_I8, v_shape, rsrc.memory_pool);
                
                // 量化K和V缓存 - 使用底层接口
                auto k_status = Quantization::quantize_kv_cache_fp16_to_int8(
                    k_quantized_buffer->data(),             // INT8输出
                    k->data(),                              // FP16输入
                    k_elements,                             // 元素数量
                    rsrc.quant_state->kv_cache_params[layer][0].scale,
                    rsrc.quant_state->kv_cache_params[layer][0].zero_point);
                    
                auto v_status = Quantization::quantize_kv_cache_fp16_to_int8(
                    v_quantized_buffer->data(),             // INT8输出
                    v->data(),                              // FP16输入
                    v_elements,                             // 元素数量
                    rsrc.quant_state->kv_cache_params[layer][1].scale,
                    rsrc.quant_state->kv_cache_params[layer][1].zero_point);
                
                // 检查量化是否成功
                if (k_status != INFINI_STATUS_SUCCESS || v_status != INFINI_STATUS_SUCCESS) {
                    // 量化失败，使用原始FP16存储
                    RUN_INFINI(infiniopRearrange(
                        desc_kv_rearranges[req],
                        kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),
                        k->data(), stream));
                    RUN_INFINI(infiniopRearrange(
                        desc_kv_rearranges[req],
                        kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),
                        v->data(), stream));
                } else {
                    // 存储量化后的KV到缓存（这里需要修改KV Cache结构支持INT8）
                    RUN_INFINI(infiniopRearrange(
                        desc_kv_rearranges[req],
                        kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),
                        k->data(), stream));
                    RUN_INFINI(infiniopRearrange(
                        desc_kv_rearranges[req],
                        kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),
                        v->data(), stream));
                }
            } else {
                // 原始FP16存储
                RUN_INFINI(infiniopRearrange(
                    desc_kv_rearranges[req],
                    kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh),
                    k->data(), stream));
                RUN_INFINI(infiniopRearrange(
                    desc_kv_rearranges[req],
                    kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh),
                    v->data(), stream));
            }
            
            
            // 计算注意力分数：Q * K^T
            // [反量化点2] 从KV Cache读取时反量化回FP16
            // 暂时禁用注意力量化计算
            if (false && rsrc.quant_state->enable_quantization) {
                // 获取Q的元素数量
                auto q_shape = q->shape();
                size_t q_elements = 1;
                for (auto dim : q_shape) q_elements *= dim;
                
                // 创建量化Q的缓冲区
                auto q_quantized_buffer = Tensor::buffer(INFINI_DTYPE_I8, q_shape, rsrc.memory_pool);
                
                // 使用量化的注意力Q*K计算
                // 首先量化Q - 使用底层接口
                auto q_status = Quantization::quantize_activation_fp16_to_int8(
                    q_quantized_buffer->data(),                      // INT8输出
                    q->data(),                                       // FP16输入
                    q_elements,                                      // 元素数量
                    rsrc.quant_state->kv_cache_params[layer][0].scale,  // 使用K的量化参数
                    rsrc.quant_state->kv_cache_params[layer][0].zero_point);
                
                if (q_status == INFINI_STATUS_SUCCESS) {
                    RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
                    
                    // 注意：这里假设KV Cache已经量化存储，实际需要根据具体实现调整
                    RUN_INFINI(infiniopGemm(
                        desc_qk_gemms[req], workspace, workspace_size,
                        qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 
                        1. / sqrt(dh), 0.0, stream));
                } else {
                    // 量化失败，使用原始FP16计算
                    RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
                    RUN_INFINI(infiniopGemm(
                        desc_qk_gemms[req], workspace, workspace_size,
                        qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 
                        1. / sqrt(dh), 0.0, stream));
                }
            } else {
                // 原始FP16计算
                RUN_INFINI(infiniopRearrange(desc_q_rearranges[req], rearrange_q_buf->data(), q->data(), stream));
                RUN_INFINI(infiniopGemm(
                    desc_qk_gemms[req], workspace, workspace_size,
                    qk_buf->data(), rearrange_q_buf->data(), kv_caches[req]->k[idev][layer]->data(), 
                    1. / sqrt(dh), 0.0, stream));
            }
            
            // 应用因果掩码的softmax
            RUN_INFINI(infiniopCausalSoftmax(
                desc_qk_softmaxs[req], workspace, workspace_size,
                qk_buf->data(), qk_buf->data(), stream));
            
            // 注意力分数与V相乘得到最终的注意力输出
            // [反量化点3] V也需要从INT8反量化回FP16
            if (rsrc.quant_state->enable_quantization) {
                // 使用量化的注意力*V计算
                Quantization::quantized_attention_v_int8(
                    rsrc.handle, workspace, workspace_size,
                    attn_val_buf->data(),                           // FP16输出
                    qk_buf->data(),                                 // FP16注意力权重
                    kv_caches[req]->v[idev][layer]->data(),        // INT8 V值（假设已量化）
                    rsrc.quant_state->kv_cache_params[layer][1].scale,    // V的量化参数
                    rsrc.quant_state->kv_cache_params[layer][1].zero_point,
                    qk_buf->desc(), 
                    kv_caches[req]->v[idev][layer]->desc(),
                    attn_val_buf->desc(),
                    stream);
            } else {
                // 原始FP16计算
                RUN_INFINI(infiniopGemm(
                    desc_attn_v_gemms[req], workspace, workspace_size,
                    attn_val_buf->data(), qk_buf->data(), kv_caches[req]->v[idev][layer]->data(), 
                    1.0, 0.0, stream));
            }
            
            // 重新排列注意力输出
            RUN_INFINI(infiniopRearrange(
                desc_attn_v_rearranges[req],
                o->data(),
                attn_val_buf->data(), stream));

            token_offset += seq_len;
        }
        
        // 注意力输出投影
        // [量化点3] 注意力输出投影前量化激活值
        if (rsrc.quant_state->enable_quantization) {
            // 使用混合精度GEMM进行注意力输出投影
            Quantization::mixed_precision_gemm(
                rsrc.handle, workspace, workspace_size,
                logits_in->data(),                                      // FP16输出
                o_buf->data(),                                          // FP16激活
                rsrc.w_attn_out[layer]->data(),                        // INT8权重（假设已量化）
                rsrc.quant_state->attn_out_weight_params[layer].scale,
                rsrc.quant_state->attn_out_weight_params[layer].zero_point,
                1.0, idev == 0 ? 1.0 : 0.0,                           // 只有设备0添加残差连接
                o_buf->desc(), rsrc.w_attn_out[layer]->desc(), logits_in->desc(),
                stream);
        } else {
            // 原始FP16计算
            RUN_INFINI(infiniopGemm(
                desc_attn_o, workspace, workspace_size,
                logits_in->data(), o_buf->data(),
                rsrc.w_attn_out[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // 只有设备0添加残差连接
        }
        // [反量化点4] 注意力输出投影结果已经是FP16，无需反量化

        // 如果是分布式推理，需要进行AllReduce通信
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        
        // 2. 前馈神经网络(FFN)
        // 层归一化
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));
        
        // FFN的门控和上投影
        // [量化点4] FFN门控和上投影前量化激活值
        if (rsrc.quant_state->enable_quantization) {
            // 使用混合精度GEMM进行FFN门控和上投影
            Quantization::mixed_precision_gemm(
                rsrc.handle, workspace, workspace_size,
                gate_up_buf->data(),                                        // FP16输出
                logits_out->data(),                                         // FP16激活
                rsrc.w_ffn_gate_up[layer]->data(),                         // INT8权重（假设已量化）
                rsrc.quant_state->ffn_gate_up_weight_params[layer].scale,
                rsrc.quant_state->ffn_gate_up_weight_params[layer].zero_point,
                1.0, 0.0,
                logits_out->desc(), rsrc.w_ffn_gate_up[layer]->desc(), gate_up_buf->desc(),
                stream);
        } else {
            // 原始FP16计算
            RUN_INFINI(infiniopGemm(
                desc_ffn_gate_up, workspace, workspace_size,
                gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer]->data(),
                1.0, 0.0, stream));
        }
        // [反量化点5] FFN门控和上投影结果已经是FP16，无需反量化
        
        // SwiGLU激活函数（保持FP16精度）
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
        
        // FFN的下投影
        // [量化点5] FFN下投影前量化激活值
        if (rsrc.quant_state->enable_quantization) {
            // 使用混合精度GEMM进行FFN下投影
            Quantization::mixed_precision_gemm(
                rsrc.handle, workspace, workspace_size,
                logits_in->data(),                                          // FP16输出
                gate_buf->data(),                                           // FP16激活
                rsrc.w_ffn_down[layer]->data(),                            // INT8权重（假设已量化）
                rsrc.quant_state->ffn_down_weight_params[layer].scale,
                rsrc.quant_state->ffn_down_weight_params[layer].zero_point,
                1.0, idev == 0 ? 1.0 : 0.0,                               // 只有设备0添加残差连接
                gate_buf->desc(), rsrc.w_ffn_down[layer]->desc(), logits_in->desc(),
                stream);
        } else {
            // 原始FP16计算
            RUN_INFINI(infiniopGemm(
                desc_ffn_down, workspace, workspace_size,
                logits_in->data(), gate_buf->data(),
                rsrc.w_ffn_down[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // 只有设备0添加残差连接
        }
        // [反量化点6] FFN下投影结果已经是FP16，无需反量化

        // 如果是分布式推理，需要进行AllReduce通信
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    
    // 采样和输出生成
    if (idev == 0) {  // 只有设备0负责最终的采样输出
        // 对每个请求的最后一个token进行最终的层归一化
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
        
        // 将隐藏状态投影到词汇表维度
        RUN_INFINI(infiniopGemm(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(), logits_out->data(),
            rsrc.w_out_embd->data(), 1.0, 0.0, stream));
        
        // 为每个请求进行采样
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
        
        // 等待GPU计算完成并将结果拷贝到CPU
        RUN_INFINI(infinirtStreamSynchronize(stream));
        RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                  sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
        for (uint32_t req = 0; req < nreq; req++) {
            output[req] = result_cpu[req];
        }
    }

    // 清理所有描述符，释放资源
    infiniopDestroyRMSNormDescriptor(desc_norm);
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

// 批量推理函数：处理多个请求的主入口点
__C void
inferBatch(struct JiugeModel *model,
           const uint32_t *tokens, uint32_t ntok,           // 输入的token序列及总数
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos, // 每个请求的长度、请求数量、位置
           struct KVCache **kv_caches,                      // KV缓存（用于attention计算）
           const float *temperature, const uint32_t *topk, const float *topp, // 采样参数：温度、top-k、top-p
           uint32_t *output) {                              // 输出结果
    // 将推理请求参数保存到模型的请求结构中，方便各设备线程访问
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

    // 通知所有设备开始工作：唤醒各设备的工作线程开始推理计算，会自动启动 每个线程的 launchDevice 函数
    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;  // 设置继续执行标志，告诉设备线程开始工作
        lock.unlock();
        model->states[idev].cv_start.notify_one();  // 唤醒对应设备的工作线程
    }
    
    // 等待所有设备完成计算：同步等待各设备推理完成
    // 按倒序等待，确保所有设备都完成了推理计算
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        // 等待设备推理完成（proceed标志变为false表示完成）
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
    
    // 此时所有设备都已完成推理，output数组中包含了各请求生成的下一个token
    // inferDeviceBatch函数在各设备线程中执行了实际的神经网络前向传播：
    // 1. 词嵌入层：将token转换为向量表示
    // 2. 多层Transformer块：
    //    - 注意力机制：计算Q、K、V矩阵，执行自注意力和RoPE位置编码
    //    - 前馈网络：通过门控线性单元(GLU)进行特征变换
    //    - 残差连接和层归一化
    // 3. 输出层：生成词表概率分布
    // 4. 采样：根据temperature、top-k、top-p参数从概率分布中采样下一个token
}

// 设备工作线程函数：在指定设备上运行推理
void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // 在设备上创建所需的计算资源（权重、缓存等）
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    {
        // 通知主线程该设备已加载完成
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // 推理循环：持续等待和处理推理请求
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        // 等待推理信号或退出信号
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        
        // 如果收到退出信号则跳出循环
        if (state.exit_flag) {
            break;
        }

        // 在当前设备上执行批量推理
        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output);

        // 标记当前设备推理完成
        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();  // 通知主线程该设备已完成
    }

    // 清理设备资源
    releaseDeviceResource(*rsrc);
}

// 模型构造函数：初始化多设备推理环境
JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);  // 每个设备的资源
    states = std::vector<InferState>(ndev);             // 每个设备的状态
    threads.resize(ndev);                               // 每个设备的工作线程
    
    // 初始化运行时
    RUN_INFINI(infinirtInit());
    
    // 如果有多个设备，建立设备间通信
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    // 为每个设备启动工作线程
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    
    // 等待所有设备完成初始化
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

// C接口：创建模型实例
__C struct JiugeModel* createJiugeModel(const JiugeMeta *meta,
                 const JiugeWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    JiugeModel *model = new JiugeModel(meta, weights, device, device_ids);
    return model;
}

// C接口：销毁模型实例
__C void destroyJiugeModel(struct JiugeModel *model) {
    auto ndev = model->dev_resources.size();

    // 通知所有设备线程退出
    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;  // 设置退出标志
        lock.unlock();
        model->states[idev].cv_start.notify_one();  // 唤醒线程以检查退出标志
    }

    // 等待所有线程结束
    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    // 释放模型内存
    delete model;
}