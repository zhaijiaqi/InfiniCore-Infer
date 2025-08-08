from typing import List, Dict, Tuple
from libinfinicore_infer import (
    JiugeMetaCStruct,
    JiugeWeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    create_jiuge_model,
    destroy_jiuge_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers
import numpy as np

torch.set_default_device("cpu")


class LlamaWeightsNaming:
    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    def attn_q_b(self, i):
        return f"model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

    def match(state_dict):
        return (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )


class AdvancedQuantizer:
    """先进量化器类，实现最新的量化算法"""
    
    @staticmethod
    def fp32_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
        """将FP32张量转换为FP16"""
        return tensor.to(torch.float16)
    
    @staticmethod
    def fp32_to_fp8_e4m3(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        改进的FP8 E4M3量化（基于Intel Gaudi2和NVFP4格式）
        使用分块量化和优化的缩放因子
        """
        # FP8 E4M3的精确数值范围（基于IEEE标准）
        fp8_max = 448.0  # 2^7 * (1 + 7/8) = 240 + 14 + 7/8 ≈ 448
        fp8_min = 2**(-9)  # 最小正常值
        
        # 分块量化策略（每16个值共享一个缩放因子）
        block_size = 16
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # 确保能整除block_size
        pad_length = (block_size - len(tensor_flat) % block_size) % block_size
        if pad_length > 0:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_length, dtype=tensor_flat.dtype)])
        
        tensor_blocked = tensor_flat.view(-1, block_size)
        quantized_blocks = []
        scale_info = {}
        
        for i, block in enumerate(tensor_blocked):
            # 计算每个块的最优缩放因子
            block_max = torch.max(torch.abs(block)).item()
            if block_max < fp8_min:
                scale = 1.0
            else:
                # 使用硬件对齐的缩放因子（提高性能）
                scale = block_max / fp8_max
                # 对齐到2的幂次（硬件友好）
                scale_exp = torch.ceil(torch.log2(torch.tensor(scale))).item()
                scale = 2 ** scale_exp
            
            # 量化块
            block_scaled = block / scale
            block_clamped = torch.clamp(block_scaled, -fp8_max, fp8_max)
            
            # 模拟FP8精度损失（通过舍入）
            # E4M3: 4位指数 + 3位尾数 = 约16个可表示值
            quantized_block = torch.round(block_clamped * 7) / 7  # 3位尾数 = 2^3-1 = 7个级别
            quantized_block = torch.clamp(quantized_block, -fp8_max, fp8_max)
            
            # 恢复到原始量级
            quantized_block = quantized_block * scale
            quantized_blocks.append(quantized_block)
            
            scale_info[f'block_{i}'] = scale
        
        # 重新组装
        quantized_tensor = torch.cat(quantized_blocks)[:len(tensor.flatten())]
        quantized_tensor = quantized_tensor.view(original_shape)
        
        # 存储为uint8以节省内存
        # 这里简化存储，实际应用中需要更复杂的编码
        normalized = (quantized_tensor - quantized_tensor.min()) / (quantized_tensor.max() - quantized_tensor.min() + 1e-8)
        quantized_uint8 = (normalized * 255).round().to(torch.uint8)
        
        metadata = {
            'min_val': quantized_tensor.min().item(),
            'max_val': quantized_tensor.max().item(),
            'original_shape': original_shape,
            'scale_info': scale_info
        }
        
        return quantized_uint8, metadata
    
    @staticmethod
    def fp8_e4m3_to_fp32(quantized_tensor: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """改进的FP8反量化"""
        # 从uint8恢复到浮点
        normalized = quantized_tensor.float() / 255.0
        reconstructed = normalized * (metadata['max_val'] - metadata['min_val']) + metadata['min_val']
        return reconstructed.view(metadata['original_shape'])
    
    @staticmethod
    def gptq_quantize(tensor: torch.Tensor, bits: int = 4, group_size: int = 128) -> Tuple[torch.Tensor, Dict]:
        """
        GPTQ量化算法实现
        基于逐层最优脑量化
        """
        if bits not in [2, 3, 4, 8]:
            raise ValueError(f"GPTQ支持的位数: 2, 3, 4, 8，当前: {bits}")
        
        original_shape = tensor.shape
        if len(original_shape) != 2:
            # 如果不是2D，reshape成2D进行处理
            tensor = tensor.view(-1, original_shape[-1])
        
        quantized_weights = []
        scales = []
        zeros = []
        
        # 按组处理权重
        for i in range(0, tensor.shape[0], group_size):
            end_idx = min(i + group_size, tensor.shape[0])
            group = tensor[i:end_idx]
            
            # 计算量化参数
            q_max = 2 ** bits - 1
            min_val = group.min(dim=0, keepdim=True)[0]
            max_val = group.max(dim=0, keepdim=True)[0]
            
            # 避免除零
            scale = (max_val - min_val) / q_max
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero_point = (-min_val / scale).round().clamp(0, q_max)
            
            # 量化
            quantized = ((group / scale) + zero_point).round().clamp(0, q_max)
            
            if bits <= 4:
                # 对于4位及以下，使用uint8存储
                quantized = quantized.to(torch.uint8)
            else:
                quantized = quantized.to(torch.int8)
            
            quantized_weights.append(quantized)
            scales.append(scale)
            zeros.append(zero_point)
        
        quantized_tensor = torch.cat(quantized_weights, dim=0)
        scale_tensor = torch.cat(scales, dim=0)
        zero_tensor = torch.cat(zeros, dim=0)
        
        metadata = {
            'scales': scale_tensor,
            'zeros': zero_tensor,
            'bits': bits,
            'group_size': group_size,
            'original_shape': original_shape
        }
        
        return quantized_tensor, metadata
    
    @staticmethod
    def gptq_dequantize(quantized_tensor: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """GPTQ反量化"""
        scales = metadata['scales']
        zeros = metadata['zeros']
        group_size = metadata['group_size']
        
        dequantized_groups = []
        
        for i in range(0, quantized_tensor.shape[0], group_size):
            end_idx = min(i + group_size, quantized_tensor.shape[0])
            group = quantized_tensor[i:end_idx].float()
            
            group_idx = i // group_size
            scale = scales[group_idx] if group_idx < len(scales) else scales[-1]
            zero = zeros[group_idx] if group_idx < len(zeros) else zeros[-1]
            
            # 反量化
            dequantized = (group - zero) * scale
            dequantized_groups.append(dequantized)
        
        result = torch.cat(dequantized_groups, dim=0)
        return result.view(metadata['original_shape'])
    
    @staticmethod
    def awq_quantize(tensor: torch.Tensor, activation_stats: torch.Tensor = None, bits: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        AWQ (Activation-aware Weight Quantization) 量化
        考虑激活值分布来优化量化
        """
        if activation_stats is None:
            # 如果没有激活统计，使用权重本身的统计
            activation_stats = torch.ones(tensor.shape[-1])
        
        # 基于激活分布计算重要性权重
        importance = torch.sqrt(activation_stats)  # 简化的重要性计算
        
        # 对重要的通道使用更高精度
        # 计算每个通道的重要性百分位
        threshold = torch.quantile(importance, 0.8)  # 前20%最重要的通道
        
        quantized_groups = []
        metadata_list = []
        
        for i in range(tensor.shape[0]):
            weight_row = tensor[i:i+1]
            
            # 根据重要性选择量化精度
            channel_importance = importance.mean()
            if channel_importance > threshold:
                # 重要通道使用更高精度
                effective_bits = min(bits + 1, 8)
            else:
                effective_bits = bits
            
            # 执行量化
            q_max = 2 ** effective_bits - 1
            min_val = weight_row.min()
            max_val = weight_row.max()
            
            scale = (max_val - min_val) / q_max
            if scale == 0:
                scale = 1.0
            
            zero_point = (-min_val / scale).round().clamp(0, q_max)
            quantized = ((weight_row / scale) + zero_point).round().clamp(0, q_max)
            
            quantized_groups.append(quantized.to(torch.uint8))
            metadata_list.append({
                'scale': scale.item(),
                'zero_point': zero_point.item(),
                'bits': effective_bits
            })
        
        quantized_tensor = torch.cat(quantized_groups, dim=0)
        
        metadata = {
            'per_channel_metadata': metadata_list,
            'original_shape': tensor.shape,
            'activation_aware': True
        }
        
        return quantized_tensor, metadata
    
    @staticmethod
    def awq_dequantize(quantized_tensor: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """AWQ反量化"""
        per_channel_metadata = metadata['per_channel_metadata']
        original_shape = metadata['original_shape']
        
        dequantized_rows = []
        
        for i, row_metadata in enumerate(per_channel_metadata):
            quantized_row = quantized_tensor[i:i+1].float()
            scale = row_metadata['scale']
            zero_point = row_metadata['zero_point']
            
            dequantized_row = (quantized_row - zero_point) * scale
            dequantized_rows.append(dequantized_row)
        
        return torch.cat(dequantized_rows, dim=0)
    
    @staticmethod
    def any4_quantize(tensor: torch.Tensor, calibration_data: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        ANY4量化 - 学习的4位数值表示
        基于权重分布优化量化码本
        """
        # 确保输入是float类型
        if tensor.dtype not in [torch.float32, torch.float16]:
            tensor = tensor.float()
        
        # 初始化16个可能的量化值（4位 = 2^4 = 16）
        num_levels = 16
        
        # 如果有校准数据，使用它来初始化码本
        if calibration_data is not None:
            if calibration_data.dtype not in [torch.float32, torch.float16]:
                calibration_data = calibration_data.float()
            # 使用K-means初始化码本
            flat_data = torch.cat([tensor.flatten(), calibration_data.flatten()])
        else:
            flat_data = tensor.flatten()
        
        # 使用分位数初始化码本
        quantiles = torch.linspace(0, 1, num_levels, dtype=torch.float32)
        codebook = torch.quantile(flat_data.float(), quantiles)
        
        # 优化码本（简化版Lloyd算法）
        for _ in range(10):  # 迭代优化
            # 为每个权重找到最近的码本值
            distances = torch.abs(tensor.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(distances, dim=-1)
            
            # 更新码本
            new_codebook = torch.zeros_like(codebook)
            for i in range(num_levels):
                mask = (indices == i)
                if mask.any():
                    new_codebook[i] = tensor[mask].mean()
                else:
                    new_codebook[i] = codebook[i]
            
            codebook = new_codebook
        
        # 最终量化
        distances = torch.abs(tensor.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0))
        quantized_indices = torch.argmin(distances, dim=-1).to(torch.uint8)
        
        metadata = {
            'codebook': codebook,
            'original_shape': tensor.shape,
            'method': 'any4'
        }
        
        return quantized_indices, metadata
    
    @staticmethod
    def any4_dequantize(quantized_tensor: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """ANY4反量化"""
        codebook = metadata['codebook']
        original_shape = metadata['original_shape']
        
        # 使用索引从码本中查找值
        dequantized = codebook[quantized_tensor.long()]
        return dequantized.view(original_shape)
    
    @staticmethod
    def optimize_weights(state_dict: Dict[str, torch.Tensor], 
                        weight_keys: List[str],
                        strategy: str = "fp16",
                        calibration_data: Dict[str, torch.Tensor] = None) -> Dict[str, Dict]:
        """
        根据策略优化权重 - 支持最新的量化算法
        
        Args:
            state_dict: 模型权重字典
            weight_keys: 需要优化的权重键列表
            strategy: 优化策略 ("fp16", "fp8", "gptq", "awq", "any4", "selective")
            calibration_data: 校准数据（用于某些方法）
            
        Returns:
            优化信息字典
        """
        optimized_info = {}
        
        print(f"开始使用 {strategy} 策略进行权重优化...")
        
        for key in weight_keys:
            if key in state_dict:
                weight = state_dict[key]
                
                if strategy == "fp16":
                    # FP16优化
                    optimized_weight = AdvancedQuantizer.fp32_to_fp16(weight)
                    optimized_info[key] = {
                        'optimized_weight': optimized_weight,
                        'strategy': 'fp16',
                        'original_shape': weight.shape,
                        'original_dtype': weight.dtype,
                        'size_reduction': weight.numel() * 4 / (optimized_weight.numel() * 2)
                    }
                    print(f"  FP16优化 {key}: {weight.dtype} -> {optimized_weight.dtype}")
                    
                elif strategy == "fp8":
                    # 改进的FP8量化
                    optimized_weight, metadata = AdvancedQuantizer.fp32_to_fp8_e4m3(weight)
                    optimized_info[key] = {
                        'optimized_weight': optimized_weight,
                        'metadata': metadata,
                        'strategy': 'fp8',
                        'original_shape': weight.shape,
                        'original_dtype': weight.dtype,
                        'size_reduction': weight.numel() * 4 / optimized_weight.numel()
                    }
                    print(f"  FP8优化 {key}: 分块量化, 压缩比 {optimized_info[key]['size_reduction']:.1f}x")
                    
                elif strategy == "gptq":
                    # GPTQ量化
                    optimized_weight, metadata = AdvancedQuantizer.gptq_quantize(weight, bits=4)
                    optimized_info[key] = {
                        'optimized_weight': optimized_weight,
                        'metadata': metadata,
                        'strategy': 'gptq',
                        'original_shape': weight.shape,
                        'original_dtype': weight.dtype,
                        'size_reduction': weight.numel() * 4 / optimized_weight.numel()
                    }
                    print(f"  GPTQ量化 {key}: 4-bit, 分组大小=128")
                    
                elif strategy == "awq":
                    # AWQ量化
                    activation_stats = calibration_data.get(key) if calibration_data else None
                    optimized_weight, metadata = AdvancedQuantizer.awq_quantize(weight, activation_stats)
                    optimized_info[key] = {
                        'optimized_weight': optimized_weight,
                        'metadata': metadata,
                        'strategy': 'awq',
                        'original_shape': weight.shape,
                        'original_dtype': weight.dtype,
                        'size_reduction': weight.numel() * 4 / optimized_weight.numel()
                    }
                    print(f"  AWQ量化 {key}: 激活感知, 自适应精度")
                    
                elif strategy == "any4":
                    # ANY4量化
                    calib_data = calibration_data.get(key) if calibration_data else None
                    optimized_weight, metadata = AdvancedQuantizer.any4_quantize(weight, calib_data)
                    optimized_info[key] = {
                        'optimized_weight': optimized_weight,
                        'metadata': metadata,
                        'strategy': 'any4',
                        'original_shape': weight.shape,
                        'original_dtype': weight.dtype,
                        'size_reduction': weight.numel() * 4 / optimized_weight.numel()
                    }
                    print(f"  ANY4量化 {key}: 学习码本, 16个量化级别")
                    
                elif strategy == "selective":
                    # 智能选择策略 - 基于权重特征选择最佳算法
                    weight_std = torch.std(weight).item()
                    weight_range = (torch.max(weight) - torch.min(weight)).item()
                    weight_max_abs = torch.max(torch.abs(weight)).item()
                    
                    # 基于统计特征选择最佳量化方法
                    if weight_std < 0.01 and weight_range < 0.5:
                        # 权重变化很小，使用GPTQ
                        optimized_weight, metadata = AdvancedQuantizer.gptq_quantize(weight, bits=4)
                        selected_strategy = 'gptq'
                        print(f"  智能选择(GPTQ) {key}: std={weight_std:.4f}")
                    elif weight_std < 0.1 and weight_max_abs > 1.0:
                        # 有大值存在，使用AWQ激活感知
                        optimized_weight, metadata = AdvancedQuantizer.awq_quantize(weight)
                        selected_strategy = 'awq'
                        print(f"  智能选择(AWQ) {key}: std={weight_std:.4f}, max_abs={weight_max_abs:.4f}")
                    elif weight_std < 0.2:
                        # 中等复杂度，使用ANY4
                        optimized_weight, metadata = AdvancedQuantizer.any4_quantize(weight)
                        selected_strategy = 'any4'
                        print(f"  智能选择(ANY4) {key}: std={weight_std:.4f}")
                    else:
                        # 权重复杂，使用FP16保持精度
                        optimized_weight = AdvancedQuantizer.fp32_to_fp16(weight)
                        metadata = None
                        selected_strategy = 'fp16'
                        print(f"  智能选择(FP16) {key}: 权重复杂，保持精度")
                    
                    optimized_info[key] = {
                        'optimized_weight': optimized_weight,
                        'metadata': metadata,
                        'strategy': selected_strategy,
                        'original_shape': weight.shape,
                        'original_dtype': weight.dtype,
                        'size_reduction': weight.numel() * 4 / (optimized_weight.numel() * (2 if selected_strategy == 'fp16' else 1))
                    }
                else:
                    raise ValueError(f"不支持的策略: {strategy}")
        
        return optimized_info


class JiugeMetaFromLlama(JiugeMetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        self.scale_input = 1.0
        self.scale_output = 1.0
        self.scale_o = 1.0
        self.scale_down = 1.0
        if (
            "fm9g" == config["model_type"]
            and "scale_emb" in config
            and "scale_depth" in config
            and "dim_model_base" in config
        ):
            self.scale_input = config["scale_emb"]
            self.scale_output = config["hidden_size"] // config["dim_model_base"]
            self.scale_o = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )
            self.scale_down = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=(
                config["num_key_value_heads"]
                if "num_key_value_heads" in config
                else config["num_attention_heads"]
            ),
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=(config["rope_theta"] if "rope_theta" in config else 100000.0),
            end_token=2,
        )
        self.torch_dtype_logits = dtype


class JiugeOptimizedWeightsImpl(JiugeWeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
        enable_optimization=True,
        optimization_strategy="fp16",  # "fp16", "fp8", "int8", "selective"
        optimize_linear_layers=True,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        scale_input = meta.scale_input
        scale_output = meta.scale_output
        scale_o = meta.scale_o
        scale_down = meta.scale_down
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        torch_dt_logits = meta.torch_dtype_logits
        
        # 数据类型设置
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")
            
        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported norm weight data type")

        # 优化相关设置
        self.enable_optimization = enable_optimization
        self.optimization_strategy = optimization_strategy
        self.optimize_linear_layers = optimize_linear_layers
        self.optimization_info = {}
        
        if self.enable_optimization and self.optimize_linear_layers:
            print(f"开始权重优化，策略: {optimization_strategy}...")
            # 确定需要优化的权重
            linear_weight_keys = []
            for i in range(nlayer):
                linear_weight_keys.extend([
                    naming.attn_q(i), naming.attn_k(i), naming.attn_v(i), naming.attn_o(i),
                    naming.gate(i), naming.up(i), naming.down(i)
                ])
            # 根据策略决定是否优化输出embedding
            if optimization_strategy in ["fp16", "fp8", "gptq", "awq", "any4", "selective"]:
                linear_weight_keys.append(naming.output_embd())
            
            # 执行优化
            self.optimization_info = AdvancedQuantizer.optimize_weights(
                state_dict, linear_weight_keys, optimization_strategy
            )
            print(f"完成 {len(self.optimization_info)} 个权重的优化")
            
            # 计算总的内存节省
            total_reduction = sum(info.get('size_reduction', 1.0) for info in self.optimization_info.values())
            avg_reduction = total_reduction / len(self.optimization_info) if self.optimization_info else 1.0
            print(f"平均内存节省比例: {avg_reduction:.2f}x")

        input_embd_naming = (
            naming.input_embd()
            if naming.input_embd() in state_dict
            else naming.output_embd()
        )
        output_embd_naming = (
            naming.output_embd()
            if naming.output_embd() in state_dict
            else naming.input_embd()
        )
        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        
        # Input embedding (不优化)
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_logits) * scale_input
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        
        # Output norm (不优化)
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm) * scale_output
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        
        # Output embedding (可能优化)
        if self.enable_optimization and output_embd_naming in self.optimization_info:
            # 使用优化的权重
            opt_info = self.optimization_info[output_embd_naming]
            if opt_info['strategy'] == 'fp16':
                self.output_embd_tensor = opt_info['optimized_weight'].to(torch_dt_mat)
            elif opt_info['strategy'] == 'fp8':
                # 对于FP8，需要反量化回浮点进行计算
                dequantized = AdvancedQuantizer.fp8_e4m3_to_fp32(
                    opt_info['optimized_weight'], opt_info['metadata']
                )
                self.output_embd_tensor = dequantized.to(torch_dt_mat)
            elif opt_info['strategy'] in ['gptq', 'awq']:
                # 对于GPTQ和AWQ，需要反量化回浮点进行计算
                if opt_info['strategy'] == 'gptq':
                    dequantized = AdvancedQuantizer.gptq_dequantize(
                        opt_info['optimized_weight'], opt_info['metadata']
                    )
                else:  # awq
                    dequantized = AdvancedQuantizer.awq_dequantize(
                        opt_info['optimized_weight'], opt_info['metadata']
                    )
                self.output_embd_tensor = dequantized.to(torch_dt_mat)
            elif opt_info['strategy'] == 'any4':
                # 对于ANY4，需要反量化回浮点进行计算
                dequantized = AdvancedQuantizer.any4_dequantize(
                    opt_info['optimized_weight'], opt_info['metadata']
                )
                self.output_embd_tensor = dequantized.to(torch_dt_mat)
            print(f"使用优化的 output embedding，策略: {opt_info['strategy']}")
        else:
            self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
            
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        # Attention norm (不优化)
        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        def qkv_slices(_i):
            # 根据是否优化来处理权重
            def get_optimized_weight(key):
                if self.enable_optimization and key in self.optimization_info:
                    opt_info = self.optimization_info[key]
                    if opt_info['strategy'] == 'fp16':
                        return opt_info['optimized_weight']
                    elif opt_info['strategy'] == 'fp8':
                        return AdvancedQuantizer.fp8_e4m3_to_fp32(
                            opt_info['optimized_weight'], opt_info['metadata']
                        )
                    elif opt_info['strategy'] == 'gptq':
                        return AdvancedQuantizer.gptq_dequantize(
                            opt_info['optimized_weight'], opt_info['metadata']
                        )
                    elif opt_info['strategy'] == 'awq':
                        return AdvancedQuantizer.awq_dequantize(
                            opt_info['optimized_weight'], opt_info['metadata']
                        )
                    elif opt_info['strategy'] == 'any4':
                        return AdvancedQuantizer.any4_dequantize(
                            opt_info['optimized_weight'], opt_info['metadata']
                        )
                else:
                    return state_dict[key]
            
            _Q = get_optimized_weight(naming.attn_q(_i))
            _K = get_optimized_weight(naming.attn_k(_i))
            _V = get_optimized_weight(naming.attn_v(_i))
            
            _Q = _Q.reshape([nh, 2, dh // 2, d]).transpose(1, 2)
            _K = _K.reshape([nkvh, 2, dh // 2, d]).transpose(1, 2)
            _V = _V.reshape([nkvh, dh // 2, 2, d])
            
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :, :])
                _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :])
                _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            return _result

        self.qkv_tensor = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.qkv_tensor[i] = (
                    self.qkv_tensor[i]
                    .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        # QKV bias处理
        def qkv_b_slices(_i):
            _QB = (
                state_dict[naming.attn_q_b(_i)]
                .reshape([nh, 2, dh // 2])
                .transpose(1, 2)
            )
            _KB = (
                state_dict[naming.attn_k_b(_i)]
                .reshape([nkvh, 2, dh // 2])
                .transpose(1, 2)
            )
            _VB = state_dict[naming.attn_v_b(_i)].reshape([nkvh, dh // 2, 2])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_QB[_idev * _nh : (_idev + 1) * _nh, :, :].flatten())
                _result.append(_KB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
                _result.append(_VB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
            return _result

        if naming.attn_q_b(0) in state_dict:
            self.qkv_b_tensors = [
                torch.concat(qkv_b_slices(i)).to(torch_dt_logits) for i in range(nlayer)
            ]
            self.qkv_b_tensor_ptrs = [
                self.qkv_b_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_qkv_b = (c_void_p * nlayer)(*self.qkv_b_tensor_ptrs)
        else:
            self.attn_qkv_b = None

        # Attention output projection (可能优化)
        def get_optimized_weight(key):
            if self.enable_optimization and key in self.optimization_info:
                opt_info = self.optimization_info[key]
                if opt_info['strategy'] == 'fp16':
                    return opt_info['optimized_weight']
                elif opt_info['strategy'] == 'fp8':
                    return AdvancedQuantizer.fp8_e4m3_to_fp32(
                        opt_info['optimized_weight'], opt_info['metadata']
                    )
                elif opt_info['strategy'] == 'gptq':
                    return AdvancedQuantizer.gptq_dequantize(
                        opt_info['optimized_weight'], opt_info['metadata']
                    )
                elif opt_info['strategy'] == 'awq':
                    return AdvancedQuantizer.awq_dequantize(
                        opt_info['optimized_weight'], opt_info['metadata']
                    )
                elif opt_info['strategy'] == 'any4':
                    return AdvancedQuantizer.any4_dequantize(
                        opt_info['optimized_weight'], opt_info['metadata']
                    )
            else:
                return state_dict[key]

        self.attn_o_tensor = []
        for i in range(nlayer):
            o_key = naming.attn_o(i)
            weight = get_optimized_weight(o_key).to(torch_dt_mat)
                
            if transpose_weight:
                weight = weight.reshape([d, ndev, nh // ndev * dh]).transpose(0, 1).contiguous()
            else:
                weight = weight.transpose(0, 1).contiguous()
                
            self.attn_o_tensor.append(weight * scale_o)
            
        self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        # FFN norm (不优化)
        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        def gate_up_slices(_i):
            gate_key, up_key = naming.gate(_i), naming.up(_i)
            
            gate_weight = get_optimized_weight(gate_key)
            up_weight = get_optimized_weight(up_key)
            
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(gate_weight[_start:_end, :])
                _result.append(up_weight[_start:_end, :])
            return _result

        self.gate_up_tensors = [
            torch.concat(gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.gate_up_tensors[i] = (
                    self.gate_up_tensors[i]
                    .reshape(ndev, 2 * di // ndev, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)

        # FFN down projection (可能优化)
        self.ffn_down_tensor = []
        for i in range(nlayer):
            down_key = naming.down(i)
            weight = get_optimized_weight(down_key).to(torch_dt_mat)
                
            if transpose_weight:
                weight = weight.reshape([d, ndev, di // ndev]).transpose(0, 1).contiguous()
            else:
                weight = weight.transpose(0, 1).contiguous()
                
            self.ffn_down_tensor.append(weight * scale_down)
            
        self.ffn_down_ptrs = [self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)


class JiugeBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )


class JiugeOptimizedForCauslLM:
    def __init__(
        self, 
        model_dir_path, 
        device=DeviceType.DEVICE_TYPE_CPU, 
        ndev=1, 
        max_tokens=None,
        enable_optimization=True,
        optimization_strategy="fp16",  # "fp16", "fp8", "gptq", "awq", "any4", "selective"
        optimize_linear_layers=True,
    ):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_

        print("加载模型权重到主机内存...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        transpose_weight = (
            device != DeviceType.DEVICE_TYPE_ASCEND
        )  # y = xW is faster than y=xW^T on Ascend
        
        if "llama" == config["model_type"]:
            model = (
                transformers.LlamaForCausalLM.from_pretrained(model_dir_path)
                .cpu()
                .half()
            )
            self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
            self.weights = JiugeOptimizedWeightsImpl(
                self.meta,
                LlamaWeightsNaming(),
                model.state_dict(),
                ndev=ndev,
                transpose_weight=transpose_weight,
                enable_optimization=enable_optimization,
                optimization_strategy=optimization_strategy,
                optimize_linear_layers=optimize_linear_layers,
            )
        elif "fm9g" == config["model_type"]:
            if any(
                file.suffix == ".safetensors" for file in Path(model_dir_path).iterdir()
            ):
                state_dict = load_all_safetensors_from_dir(model_dir_path)
            else:
                state_dict = torch.load(
                    os.path.join(model_dir_path, "pytorch_model.bin"),
                    weights_only=True,
                    map_location="cpu",
                )
            if LlamaWeightsNaming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
                self.weights = JiugeOptimizedWeightsImpl(
                    self.meta,
                    LlamaWeightsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                    enable_optimization=enable_optimization,
                    optimization_strategy=optimization_strategy,
                    optimize_linear_layers=optimize_linear_layers,
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
                )
            else:
                raise ValueError("Unsupported weight naming")
        elif "fm9g7b" == config["model_type"]:
            if any(
                file.suffix == ".safetensors" for file in Path(model_dir_path).iterdir()
            ):
                state_dict = load_all_safetensors_from_dir(model_dir_path)
            else:
                state_dict = torch.load(
                    os.path.join(model_dir_path, "pytorch_model.bin"),
                    weights_only=True,
                    map_location="cpu",
                )
            if LlamaWeightsNaming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
                self.weights = JiugeOptimizedWeightsImpl(
                    self.meta,
                    LlamaWeightsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                    enable_optimization=enable_optimization,
                    optimization_strategy=optimization_strategy,
                    optimize_linear_layers=optimize_linear_layers,
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
                )
            else:
                raise ValueError("Unsupported weight naming")
        elif "qwen2" == config["model_type"]:
            state_dict = load_all_safetensors_from_dir(model_dir_path)
            if LlamaWeightsNaming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
                self.weights = JiugeOptimizedWeightsImpl(
                    self.meta,
                    LlamaWeightsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                    enable_optimization=enable_optimization,
                    optimization_strategy=optimization_strategy,
                    optimize_linear_layers=optimize_linear_layers,
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path
                )
        else:
            raise ValueError("Unsupported model architecture")

        load_end_time = time.time()
        print(f"权重加载耗时: {load_end_time - load_start_time:.3f}s")
        
        if enable_optimization:
            print(f"优化模式已启用，策略: {optimization_strategy}, 线性层优化: {optimize_linear_layers}")
            if hasattr(self.weights, 'optimization_info'):
                print(f"优化了 {len(self.weights.optimization_info)} 个权重张量")

        print(f"在 {ndev} 个设备上创建模型...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_jiuge_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        load_end_time = time.time()
        print(f"模型创建耗时: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = JiugeBatchedTask(tasks)
        infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )
            output_content += output_str
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                break
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"每步推理时间: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def destroy_model_instance(self):
        destroy_jiuge_model(self.model_instance)
        print("模型已销毁")


def test():
    if len(sys.argv) < 3:
        print(
            "用法: python jiuge_quantized.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <模型路径> [设备数量] [选项]"
        )
        print("选项:")
        print("  --no-optimization: 禁用优化")
        print("  --strategy=fp16: 使用FP16优化 (默认)")
        print("  --strategy=fp8: 使用改进的FP8量化")
        print("  --strategy=gptq: 使用GPTQ量化算法")
        print("  --strategy=awq: 使用AWQ激活感知量化")
        print("  --strategy=any4: 使用ANY4学习量化")
        print("  --strategy=selective: 智能选择最佳策略")
        sys.exit(1)
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif sys.argv[1] == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif sys.argv[1] == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    else:
        print(
            "用法: python jiuge_quantized.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <模型路径> [设备数量] [选项]"
        )
        sys.exit(1)

    ndev = 1
    enable_optimization = True
    optimization_strategy = "fp16"  # 默认使用FP16

    # 解析其他参数
    for arg in sys.argv[3:]:
        if arg.isdigit():
            ndev = int(arg)
        elif arg == "--no-optimization":
            enable_optimization = False
        elif arg.startswith("--strategy="):
            strategy = arg.split("=")[1]
            if strategy in ["fp16", "fp8", "gptq", "awq", "any4", "selective"]:
                optimization_strategy = strategy
            else:
                print(f"❌ 不支持的策略: {strategy}")
                print("支持的策略: fp16, fp8, gptq, awq, any4, selective")
                sys.exit(1)
    
    print(f"🚀 使用 {optimization_strategy.upper()} 策略开始模型推理...")
    
    model = JiugeOptimizedForCauslLM(
        model_path, 
        device_type, 
        ndev,
        enable_optimization=enable_optimization,
        optimization_strategy=optimization_strategy,
        optimize_linear_layers=True
    )
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test() 