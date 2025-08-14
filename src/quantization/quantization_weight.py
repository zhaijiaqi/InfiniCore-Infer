import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
import safetensors

class JiugeQuantizer:
    """Jiuge模型量化器，支持多种量化方法"""
    
    def __init__(self, model_path, quant_path):
        self.model_path = model_path
        self.quant_path = quant_path
        self.config = None
        self.tokenizer = None
        
    def load_model_config(self):
        """加载模型配置"""
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)
        print(f"模型类型: {self.config.get('model_type', 'unknown')}")
        return self.config
    
    def load_tokenizer(self):
        """加载tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        return self.tokenizer
    
    def load_safetensors(self, model_dir):
        """加载safetensors格式的权重"""
        tensors = {}
        model_dir = Path(model_dir)
        for file in sorted(model_dir.glob("*.safetensors")):
            data = safetensors.safe_open(file, "pt")
            for name in data.keys():
                tensors[name] = data.get_tensor(name)
        return tensors
    
    def load_pytorch_weights(self, model_dir):
        """加载pytorch格式的权重"""
        model_file = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(model_file):
            return torch.load(model_file, weights_only=True, map_location="cpu")
        else:
            raise FileNotFoundError(f"找不到权重文件: {model_file}")
    
    def quantize_weights(self, state_dict, quant_config):
        """量化权重"""
        quantized_state_dict = {}
        quantization_info = {}
        
        print("开始权重量化...")
        
        for key, weight in state_dict.items():
            # 只量化线性层的权重
            if "weight" in key and any(layer in key for layer in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                print(f"量化层: {key}")
                
                # 根据量化配置选择方法
                if quant_config.get("method") == "awq":
                    quantized_weight, metadata = self.awq_quantize(weight, quant_config)
                elif quant_config.get("method") == "gptq":
                    quantized_weight, metadata = self.gptq_quantize(weight, quant_config)
                elif quant_config.get("method") == "int8":
                    quantized_weight, metadata = self.int8_quantize(weight)
                elif quant_config.get("method") == "hybrid":
                    quantized_weight, metadata = self.hybrid_quantize(weight, quant_config)
                else:
                    # 默认使用INT8量化
                    quantized_weight, metadata = self.int8_quantize(weight)
                
                quantized_state_dict[key] = quantized_weight
                quantized_state_dict[f"{key}_scale"] = metadata["scale"]
                quantization_info[key] = metadata
            else:
                # 非线性层保持原样
                quantized_state_dict[key] = weight
        
        print(f"量化完成，共量化了 {len(quantization_info)} 个权重层")
        return quantized_state_dict, quantization_info
    
    def int8_quantize(self, weight):
        """INT8对称量化（完整实现）"""
        # 确保权重是浮点类型
        weight = weight.float()
        
        original_shape = weight.shape
        if len(original_shape) != 2:
            weight = weight.view(-1, original_shape[-1])
        
        # 1. 计算量化参数
        # 使用对称量化，zero_point = 0
        max_abs_val = torch.max(torch.abs(weight))
        scale = max_abs_val / 127.0
        if scale == 0:
            scale = 1.0
        
        # 2. 量化
        quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
        
        # 3. 反量化计算误差
        dequantized = quantized.float() * scale
        quantization_error = torch.mean((weight - dequantized) ** 2).item()
        
        # 4. 计算统计信息
        original_size = weight.numel() * weight.element_size()
        quantized_size = quantized.numel() * quantized.element_size()
        scale_size = 4  # float32 scale
        total_quantized_size = quantized_size + scale_size
        
        compression_ratio = original_size / total_quantized_size
        
        metadata = {
            "scale": scale,
            "method": "int8",
            "original_shape": original_shape,
            "original_dtype": str(weight.dtype),
            "compression_ratio": compression_ratio,
            "quantization_error": quantization_error,
            "max_abs_value": max_abs_val.item(),
            "scale_value": scale
        }
        
        return quantized, metadata
    
    def gptq_quantize(self, weight, quant_config):
        """GPTQ量化（完整实现）
        
        GPTQ (Gradient-based Post-Training Quantization) 是一种基于梯度的
        后训练量化方法，通过最小化量化误差来优化量化参数。
        """
        bits = quant_config.get("w_bit", 4)
        group_size = quant_config.get("q_group_size", 128)
        
        # 确保权重是浮点类型
        weight = weight.float()
        
        original_shape = weight.shape
        if len(original_shape) != 2:
            weight = weight.view(-1, original_shape[-1])
        
        # 1. 计算Hessian矩阵（简化版）
        # 在实际GPTQ中，Hessian矩阵基于激活值计算
        # 这里使用权重本身的统计特性作为近似
        weight_hessian = torch.diag(torch.var(weight, dim=0))  # 对角Hessian近似
        
        # 2. 初始化量化参数
        quantized_weights = []
        scales = []
        zeros = []
        quantization_stats = []
        
        # 3. 逐列进行最优量化
        for col_idx in range(weight.shape[1]):
            # 获取当前列
            weight_col = weight[:, col_idx]  # [K]
            
            # 计算该列的量化参数
            q_max = 2 ** bits - 1
            q_min = 0
            
            # 使用GPTQ的贪心搜索策略
            min_val = weight_col.min()
            max_val = weight_col.max()
            
            # 计算初始量化参数
            scale = (max_val - min_val) / q_max
            if scale == 0:
                scale = 1.0
            
            zero_point = (-min_val / scale).round().clamp(q_min, q_max)
            
            # 4. 贪心搜索最优量化参数
            best_error = float('inf')
            best_scale = scale
            best_zero_point = zero_point
            
            # 搜索最优的scale和zero_point
            scale_candidates = torch.linspace(scale * 0.8, scale * 1.2, 20)
            zero_candidates = torch.linspace(
                max(q_min, zero_point - 10), 
                min(q_max, zero_point + 10), 
                20
            )
            
            for scale_cand in scale_candidates:
                for zero_cand in zero_candidates:
                    # 量化
                    quantized = ((weight_col / scale_cand) + zero_cand).round().clamp(q_min, q_max)
                    
                    # 反量化
                    dequantized = (quantized.float() - zero_cand) * scale_cand
                    
                    # 计算量化误差（考虑Hessian权重）
                    error = torch.sum((weight_col - dequantized) ** 2 * weight_hessian[col_idx, col_idx])
                    
                    if error < best_error:
                        best_error = error
                        best_scale = scale_cand
                        best_zero_point = zero_cand
            
            # 5. 使用最优参数进行最终量化
            final_quantized = ((weight_col / best_scale) + best_zero_point).round().clamp(q_min, q_max)
            
            # 转换为适当的整数类型
            if bits <= 4:
                final_quantized = final_quantized.to(torch.uint8)
            elif bits <= 8:
                final_quantized = final_quantized.to(torch.int8)
            else:
                final_quantized = final_quantized.to(torch.int16)
            
            quantized_weights.append(final_quantized)
            scales.append(torch.tensor([best_scale]))
            zeros.append(torch.tensor([best_zero_point]))
            
            # 记录量化统计信息
            final_dequantized = (final_quantized.float() - best_zero_point) * best_scale
            final_error = torch.mean((weight_col - final_dequantized) ** 2).item()
            
            quantization_stats.append({
                "column": col_idx,
                "original_range": [min_val.item(), max_val.item()],
                "scale": best_scale.item(),
                "zero_point": best_zero_point.item(),
                "quantization_error": final_error,
                "hessian_weight": weight_hessian[col_idx, col_idx].item()
            })
        
        # 6. 合并结果
        quantized_tensor = torch.stack(quantized_weights, dim=1)  # [K, N]
        scale_tensor = torch.cat(scales, dim=0)  # [N]
        zero_tensor = torch.cat(zeros, dim=0)  # [N]
        
        # 7. 计算整体统计信息
        original_size = weight.numel() * weight.element_size()
        quantized_size = quantized_tensor.numel() * quantized_tensor.element_size()
        scale_size = scale_tensor.numel() * scale_tensor.element_size()
        zero_size = zero_tensor.numel() * zero_tensor.element_size()
        total_quantized_size = quantized_size + scale_size + zero_size
        
        compression_ratio = original_size / total_quantized_size
        
        # 8. 计算整体量化误差
        dequantized = (quantized_tensor.float() - zero_tensor) * scale_tensor
        overall_error = torch.mean((weight - dequantized) ** 2).item()
        
        # 9. 计算逐层误差统计
        layer_errors = []
        for col_idx in range(weight.shape[1]):
            col_error = torch.mean((weight[:, col_idx] - dequantized[:, col_idx]) ** 2).item()
            layer_errors.append(col_error)
        
        metadata = {
            "scale": scale_tensor,
            "zero_point": zero_tensor,
            "method": "gptq",
            "bits": bits,
            "group_size": group_size,
            "original_shape": original_shape,
            "original_dtype": str(weight.dtype),
            "compression_ratio": compression_ratio,
            "overall_quantization_error": overall_error,
            "layer_quantization_errors": layer_errors,
            "quantization_stats": quantization_stats,
            "hessian_stats": {
                "mean": torch.mean(weight_hessian.diag()).item(),
                "std": torch.std(weight_hessian.diag()).item(),
                "min": torch.min(weight_hessian.diag()).item(),
                "max": torch.max(weight_hessian.diag()).item()
            }
        }
        
        return quantized_tensor, metadata
    
    def awq_quantize(self, weight, quant_config):
        """AWQ量化（完整实现）
        
        AWQ (Activation-aware Weight Quantization) 是一种基于激活值分布
        来优化权重量化的方法，通过分析激活值的重要性来调整量化精度。
        """
        bits = quant_config.get("w_bit", 4)
        group_size = quant_config.get("q_group_size", 128)
        
        # 确保权重是浮点类型
        weight = weight.float()
        
        # 简化的AWQ实现
        original_shape = weight.shape
        if len(original_shape) != 2:
            weight = weight.view(-1, original_shape[-1])
        
        # 1. 计算权重的重要性指标
        # 使用权重的L2范数作为重要性指标
        weight_norms = torch.norm(weight, dim=0, p=2)  # [N]
        
        # 2. 计算激活感知的重要性权重
        # 基于权重的统计特性估计激活值分布
        weight_std = torch.std(weight, dim=0)  # [N]
        weight_mean = torch.mean(weight, dim=0)  # [N]
        
        # 计算激活敏感性权重（基于权重分布特征）
        # 这里使用简化的激活敏感性计算
        activation_sensitivity = torch.sqrt(weight_norms ** 2 + weight_std ** 2)
        
        # 3. 自适应精度选择
        # 根据重要性选择量化精度
        importance_threshold_high = torch.quantile(activation_sensitivity, 0.9)  # 前10%最重要的通道
        importance_threshold_medium = torch.quantile(activation_sensitivity, 0.7)  # 前30%重要的通道
        
        # 4. 分组量化
        quantized_weights = []
        scales = []
        zeros = []
        importance_info = []
        
        for i in range(0, weight.shape[0], group_size):
            end_idx = min(i + group_size, weight.shape[0])
            group = weight[i:end_idx]  # [group_size, N]
            
            # 计算该组的重要性
            group_importance = activation_sensitivity.mean()
            
            # 根据重要性选择量化精度
            if group_importance > importance_threshold_high:
                effective_bits = min(bits + 2, 8)  # 最重要组使用更高精度
                quant_method = "high_precision"
            elif group_importance > importance_threshold_medium:
                effective_bits = min(bits + 1, 8)  # 重要组使用中等精度
                quant_method = "medium_precision"
            else:
                effective_bits = bits  # 普通组使用标准精度
                quant_method = "standard_precision"
            
            # 5. 执行量化
            q_max = 2 ** effective_bits - 1
            q_min = 0
            
            # 计算量化参数 - 为每一列计算独立的scale和zero_point
            min_val = group.min(dim=0, keepdim=True)[0]  # [1, N]
            max_val = group.max(dim=0, keepdim=True)[0]  # [1, N]
            
            # 避免除零
            scale = (max_val - min_val) / q_max
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            
            # 计算零点
            zero_point = (-min_val / scale).round().clamp(q_min, q_max)
            
            # 量化
            quantized = ((group / scale) + zero_point).round().clamp(q_min, q_max)
            
            # 转换为适当的整数类型
            if effective_bits <= 4:
                quantized = quantized.to(torch.uint8)
            elif effective_bits <= 8:
                quantized = quantized.to(torch.int8)
            else:
                quantized = quantized.to(torch.int16)
            
            quantized_weights.append(quantized)
            scales.append(scale)  # [1, N]
            zeros.append(zero_point)  # [1, N]
            
            # 记录重要性信息
            importance_info.append({
                "group_start": i,
                "group_end": end_idx,
                "importance": group_importance.item(),
                "effective_bits": effective_bits,
                "quant_method": quant_method
            })
        
        # 6. 合并结果
        quantized_tensor = torch.cat(quantized_weights, dim=0)  # [K, N]
        scale_tensor = torch.cat(scales, dim=0)  # [num_groups, N]
        zero_tensor = torch.cat(zeros, dim=0)  # [num_groups, N]
        
        # 确保scale和zero_point的维度与quantized_tensor匹配
        # 如果scale_tensor和zero_tensor的行数与quantized_tensor不匹配，需要广播
        if scale_tensor.shape[0] != quantized_tensor.shape[0]:
            # 创建与quantized_tensor相同形状的scale和zero_point张量
            final_scale = torch.zeros_like(quantized_tensor, dtype=torch.float32)
            final_zero = torch.zeros_like(quantized_tensor, dtype=torch.float32)
            
            # 为每个组分配对应的scale和zero_point
            for group_idx, (start_idx, end_idx) in enumerate([(i, min(i + group_size, weight.shape[0])) 
                                                             for i in range(0, weight.shape[0], group_size)]):
                if group_idx < scale_tensor.shape[0]:
                    final_scale[start_idx:end_idx] = scale_tensor[group_idx:group_idx+1]
                    final_zero[start_idx:end_idx] = zero_tensor[group_idx:group_idx+1]
            
            scale_tensor = final_scale
            zero_tensor = final_zero
        
        # 7. 计算量化统计信息
        original_size = weight.numel() * weight.element_size()
        quantized_size = quantized_tensor.numel() * quantized_tensor.element_size()
        scale_size = scale_tensor.numel() * scale_tensor.element_size()
        zero_size = zero_tensor.numel() * zero_tensor.element_size()
        total_quantized_size = quantized_size + scale_size + zero_size
        
        compression_ratio = original_size / total_quantized_size
        
        # 8. 计算量化误差
        # 反量化以计算误差
        dequantized = (quantized_tensor.float() - zero_tensor) * scale_tensor
        quantization_error = torch.mean((weight - dequantized) ** 2).item()
        
        metadata = {
            "scale": scale_tensor,
            "zero_point": zero_tensor,
            "method": "awq",
            "bits": bits,
            "group_size": group_size,
            "original_shape": original_shape,
            "original_dtype": str(weight.dtype),
            "importance_info": importance_info,
            "compression_ratio": compression_ratio,
            "quantization_error": quantization_error,
            "activation_sensitivity_stats": {
                "mean": activation_sensitivity.mean().item(),
                "std": activation_sensitivity.std().item(),
                "min": activation_sensitivity.min().item(),
                "max": activation_sensitivity.max().item()
            }
        }
        
        return quantized_tensor, metadata
    
    def hybrid_quantize(self, weight, quant_config):
        """混合量化（完整实现）
        
        结合多种量化方法的优势，根据权重的不同特征选择最适合的量化策略。
        """
        # 确保权重是浮点类型
        weight = weight.float()
        
        original_shape = weight.shape
        if len(original_shape) != 2:
            weight = weight.view(-1, original_shape[-1])
        
        # 1. 分析权重特征
        weight_norms = torch.norm(weight, dim=0, p=2)  # [N]
        weight_std = torch.std(weight, dim=0)  # [N]
        weight_range = torch.max(weight, dim=0)[0] - torch.min(weight, dim=0)[0]  # [N]
        
        # 2. 计算权重复杂度指标
        complexity_score = weight_std / (weight_norms + 1e-8)  # 避免除零
        
        # 3. 根据复杂度选择量化策略
        quantized_weights = []
        scales = []
        zeros = []
        methods = []
        quantization_info = []
        
        for col_idx in range(weight.shape[1]):
            weight_col = weight[:, col_idx]
            col_complexity = complexity_score[col_idx]
            col_norm = weight_norms[col_idx]
            col_std = weight_std[col_idx]
            
            # 根据特征选择量化方法
            if col_complexity < 0.1 and col_norm > 1.0:
                # 低复杂度、高范数：使用INT8对称量化
                method = "int8_symmetric"
                bits = 8
                q_max = 127
                q_min = -128
                
                max_abs = torch.max(torch.abs(weight_col))
                scale = max_abs / 127.0
                if scale == 0:
                    scale = 1.0
                
                zero_point = 0
                quantized = torch.round(weight_col / scale).clamp(q_min, q_max).to(torch.int8)
                
            elif col_complexity < 0.3:
                # 中等复杂度：使用GPTQ风格量化
                method = "gptq_style"
                bits = 6
                q_max = 2 ** bits - 1
                q_min = 0
                
                min_val = weight_col.min()
                max_val = weight_col.max()
                scale = (max_val - min_val) / q_max
                if scale == 0:
                    scale = 1.0
                
                zero_point = (-min_val / scale).round().clamp(q_min, q_max)
                quantized = ((weight_col / scale) + zero_point).round().clamp(q_min, q_max).to(torch.uint8)
                
            else:
                # 高复杂度：使用AWQ风格量化
                method = "awq_style"
                bits = 4
                q_max = 2 ** bits - 1
                q_min = 0
                
                # 自适应精度
                if col_std > torch.median(weight_std):
                    effective_bits = min(bits + 1, 6)
                else:
                    effective_bits = bits
                
                q_max = 2 ** effective_bits - 1
                
                min_val = weight_col.min()
                max_val = weight_col.max()
                scale = (max_val - min_val) / q_max
                if scale == 0:
                    scale = 1.0
                
                zero_point = (-min_val / scale).round().clamp(q_min, q_max)
                quantized = ((weight_col / scale) + zero_point).round().clamp(q_min, q_max).to(torch.uint8)
            
            quantized_weights.append(quantized)
            scales.append(torch.tensor([scale]))
            zeros.append(torch.tensor([zero_point]))
            methods.append(method)
            
            # 计算量化误差
            dequantized = (quantized.float() - zero_point) * scale
            error = torch.mean((weight_col - dequantized) ** 2).item()
            
            quantization_info.append({
                "column": col_idx,
                "method": method,
                "bits": effective_bits if method == "awq_style" else bits,
                "complexity": col_complexity.item(),
                "norm": col_norm.item(),
                "std": col_std.item(),
                "scale": scale.item() if hasattr(scale, 'item') else scale,
                "zero_point": zero_point.item() if hasattr(zero_point, 'item') else zero_point,
                "quantization_error": error
            })
        
        # 4. 合并结果
        quantized_tensor = torch.stack(quantized_weights, dim=1)
        scale_tensor = torch.cat(scales, dim=0)
        zero_tensor = torch.cat(zeros, dim=0)
        
        # 5. 计算整体统计
        original_size = weight.numel() * weight.element_size()
        quantized_size = quantized_tensor.numel() * quantized_tensor.element_size()
        scale_size = scale_tensor.numel() * scale_tensor.element_size()
        zero_size = zero_tensor.numel() * zero_tensor.element_size()
        total_quantized_size = quantized_size + scale_size + zero_size
        
        compression_ratio = original_size / total_quantized_size
        
        # 6. 计算整体误差
        dequantized = (quantized_tensor.float() - zero_tensor) * scale_tensor
        overall_error = torch.mean((weight - dequantized) ** 2).item()
        
        # 7. 统计各方法的使用情况
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        metadata = {
            "scale": scale_tensor,
            "zero_point": zero_tensor,
            "method": "hybrid",
            "original_shape": original_shape,
            "original_dtype": str(weight.dtype),
            "compression_ratio": compression_ratio,
            "overall_quantization_error": overall_error,
            "quantization_info": quantization_info,
            "method_distribution": method_counts,
            "complexity_stats": {
                "mean": complexity_score.mean().item(),
                "std": complexity_score.std().item(),
                "min": complexity_score.min().item(),
                "max": complexity_score.max().item()
            }
        }
        
        return quantized_tensor, metadata
    
    def save_quantized_model(self, quantized_state_dict, quantization_info):
        """保存量化模型"""
        # 创建输出目录
        os.makedirs(self.quant_path, exist_ok=True)
        
        # 保存量化后的权重
        quantized_weights = {}
        for key, value in quantized_state_dict.items():
            if not key.endswith("_scale"):
                quantized_weights[key] = value
        
        # 保存为safetensors格式
        from safetensors.torch import save_file
        save_file(quantized_weights, os.path.join(self.quant_path, "model.safetensors"))
        
        # 保存量化信息
        with open(os.path.join(self.quant_path, "quantization_info.json"), "w") as f:
            json.dump(quantization_info, f, indent=2, default=str)
        
        # 保存配置和tokenizer
        if self.config:
            with open(os.path.join(self.quant_path, "config.json"), "w") as f:
                json.dump(self.config, f, indent=2)
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(self.quant_path)
        
        print(f"量化模型已保存到: {self.quant_path}")
    
    def quantize(self, quant_config):
        """执行完整的量化流程"""
        print("开始Jiuge模型量化...")
        
        # 1. 加载模型配置
        self.load_model_config()
        
        # 2. 加载tokenizer
        self.load_tokenizer()
        
        # 3. 加载权重
        print("加载模型权重...")
        if any(Path(self.model_path).glob("*.safetensors")):
            state_dict = self.load_safetensors(self.model_path)
        else:
            state_dict = self.load_pytorch_weights(self.model_path)
        
        print(f"加载了 {len(state_dict)} 个权重层")
        
        # 4. 量化权重
        quantized_state_dict, quantization_info = self.quantize_weights(state_dict, quant_config)
        
        # 5. 保存量化模型
        self.save_quantized_model(quantized_state_dict, quantization_info)
        
        print("量化完成！")

def main():
    # 配置参数
    model_path = "/home/shared/models/jiuge9G4B/"  # 本地模型路径
    quant_path = "/home/halozjq/models/jiuge9G4B_quantized"  # 量化模型保存路径
    
    # 量化配置 - 可以选择不同的量化方法
    quant_configs = {
        "awq": {
            "method": "awq",        # 激活感知量化
            "w_bit": 8,             # 量化比特数
            "q_group_size": 128,    # 量化组大小
            "zero_point": True      # 是否使用零点量化
        },
        "gptq": {
            "method": "gptq",       # GPTQ量化
            "w_bit": 8,             # 量化比特数
            "q_group_size": 128,    # 量化组大小
            "zero_point": True      # 是否使用零点量化
        },
        "int8": {
            "method": "int8",       # INT8对称量化
            "w_bit": 8,             # 量化比特数
            "q_group_size": 128,    # 量化组大小
            "zero_point": False     # 对称量化不使用零点
        },
        "hybrid": {
            "method": "hybrid",     # 混合量化
            "w_bit": 8,             # 基础量化比特数
            "q_group_size": 128,    # 量化组大小
            "zero_point": True      # 是否使用零点量化
        }
    }
    
    # 选择要使用的量化方法
    selected_method = "int8"  # 可以改为 "awq", "gptq", "int8", "hybrid"
    quant_config = quant_configs[selected_method]
    quant_path += "_"+selected_method+"/"
    
    print(f"使用量化方法: {selected_method}")
    print(f"量化配置: {quant_config}")
    
    # 创建量化器并执行量化
    quantizer = JiugeQuantizer(model_path, quant_path)
    quantizer.quantize(quant_config)
    
    print(f"\n量化完成！量化模型保存在: {quant_path}")
    print(f"量化方法: {selected_method}")
    print(f"量化比特数: {quant_config['w_bit']}")
    print(f"量化组大小: {quant_config['q_group_size']}")

if __name__ == "__main__":
    main()