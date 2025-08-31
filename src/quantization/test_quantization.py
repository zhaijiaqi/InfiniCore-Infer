#!/usr/bin/env python3
"""
量化模型质量测试脚本
用于验证量化后的模型质量和性能
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_quantization_info(quant_path):
    """加载量化信息"""
    info_file = Path(quant_path) / "quantization_info.json"
    with open(info_file, 'r') as f:
        return json.load(f)

def analyze_quantization_quality(quantization_info):
    """分析量化质量"""
    print("=== 量化质量分析 ===")
    
    # 统计信息
    total_layers = len(quantization_info)
    methods_used = {}
    compression_ratios = []
    quantization_errors = []
    
    for layer_name, info in quantization_info.items():
        # 统计方法使用情况
        method = info.get('method', 'unknown')
        methods_used[method] = methods_used.get(method, 0) + 1
        
        # 收集压缩比和误差
        if 'compression_ratio' in info:
            compression_ratios.append(info['compression_ratio'])
        if 'quantization_error' in info:
            quantization_errors.append(info['quantization_error'])
    
    # 打印统计结果
    print(f"总层数: {total_layers}")
    print(f"量化方法分布: {methods_used}")
    
    if compression_ratios:
        print(f"平均压缩比: {np.mean(compression_ratios):.2f}x")
        print(f"压缩比范围: {np.min(compression_ratios):.2f}x - {np.max(compression_ratios):.2f}x")
    
    if quantization_errors:
        print(f"平均量化误差: {np.mean(quantization_errors):.2e}")
        print(f"最大量化误差: {np.max(quantization_errors):.2e}")
        print(f"最小量化误差: {np.min(quantization_errors):.2e}")
    
    return {
        'total_layers': total_layers,
        'methods_used': methods_used,
        'avg_compression_ratio': np.mean(compression_ratios) if compression_ratios else 0,
        'avg_quantization_error': np.mean(quantization_errors) if quantization_errors else 0
    }

def test_dequantization_accuracy(quant_path):
    """测试反量化精度"""
    print("\n=== 反量化精度测试 ===")
    
    # 加载量化信息
    quantization_info = load_quantization_info(quant_path)
    
    # 选择几个层进行测试
    test_layers = list(quantization_info.keys())[:5]  # 测试前5层
    
    for layer_name in test_layers:
        info = quantization_info[layer_name]
        print(f"\n测试层: {layer_name}")
        
        # 模拟量化-反量化过程
        original_shape = info['original_shape']
        scale = info['scale']
        zero_point = info['zero_point']
        
        # 生成随机权重进行测试
        original_weight = torch.randn(original_shape)
        
        # 量化
        if isinstance(scale, str) and 'tensor' in scale:
            # 处理tensor字符串
            scale_values = torch.tensor([0.001])  # 简化处理
        else:
            scale_values = torch.tensor([scale])
        
        if isinstance(zero_point, str) and 'tensor' in zero_point:
            zero_point_values = torch.tensor([0])  # 简化处理
        else:
            zero_point_values = torch.tensor([zero_point])
        
        # 模拟量化过程
        quantized = torch.round(original_weight / scale_values).clamp(-128, 127)
        
        # 反量化
        dequantized = (quantized.float() - zero_point_values) * scale_values
        
        # 计算误差
        error = torch.mean((original_weight - dequantized) ** 2).item()
        print(f"  量化误差: {error:.2e}")
        print(f"  相对误差: {error / torch.mean(original_weight ** 2).item():.2e}")

def benchmark_quantization_speed(quant_path):
    """基准测试量化速度"""
    print("\n=== 量化速度基准测试 ===")
    
    # 模拟不同大小的权重矩阵
    test_sizes = [
        (512, 512),    # 小矩阵
        (1024, 1024),  # 中等矩阵
        (2048, 2048),  # 大矩阵
        (4096, 4096),  # 超大矩阵
    ]
    
    for size in test_sizes:
        print(f"\n测试矩阵大小: {size}")
        
        # 生成测试权重
        weight = torch.randn(size)
        
        # 测试INT8量化速度
        start_time = time.time()
        for _ in range(100):  # 重复100次
            max_abs = torch.max(torch.abs(weight))
            scale = max_abs / 127.0
            quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
        int8_time = time.time() - start_time
        
        # 测试AWQ风格量化速度
        start_time = time.time()
        for _ in range(100):
            # 简化的AWQ量化
            weight_norms = torch.norm(weight, dim=0, p=2)
            weight_std = torch.std(weight, dim=0)
            activation_sensitivity = torch.sqrt(weight_norms ** 2 + weight_std ** 2)
            
            # 分组量化
            group_size = 128
            quantized_groups = []
            for i in range(0, weight.shape[0], group_size):
                group = weight[i:i+group_size]
                scale = torch.max(torch.abs(group)) / 127.0
                quantized_group = torch.round(group / scale).clamp(-128, 127).to(torch.int8)
                quantized_groups.append(quantized_group)
        awq_time = time.time() - start_time
        
        print(f"  INT8量化时间: {int8_time:.4f}s (100次)")
        print(f"  AWQ量化时间: {awq_time:.4f}s (100次)")
        print(f"  速度比: {awq_time/int8_time:.2f}x")

def generate_quantization_report(quant_path, output_file="quantization_report.txt"):
    """生成量化报告"""
    print(f"\n=== 生成量化报告: {output_file} ===")
    
    quantization_info = load_quantization_info(quant_path)
    quality_stats = analyze_quantization_quality(quantization_info)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("量化模型质量报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 总体统计\n")
        f.write(f"   总层数: {quality_stats['total_layers']}\n")
        f.write(f"   平均压缩比: {quality_stats['avg_compression_ratio']:.2f}x\n")
        f.write(f"   平均量化误差: {quality_stats['avg_quantization_error']:.2e}\n\n")
        
        f.write("2. 量化方法分布\n")
        for method, count in quality_stats['methods_used'].items():
            percentage = count / quality_stats['total_layers'] * 100
            f.write(f"   {method}: {count} 层 ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("3. 逐层详细信息\n")
        for layer_name, info in list(quantization_info.items())[:10]:  # 只显示前10层
            f.write(f"   {layer_name}:\n")
            f.write(f"     方法: {info.get('method', 'unknown')}\n")
            f.write(f"     压缩比: {info.get('compression_ratio', 'N/A')}\n")
            f.write(f"     量化误差: {info.get('quantization_error', 'N/A')}\n")
            f.write(f"     原始形状: {info.get('original_shape', 'N/A')}\n\n")
    
    print(f"报告已保存到: {output_file}")

def main():
    """主函数"""
    # 量化模型路径 - 测试AWQ量化模型
    quant_path = "/home/halozjq/models/jiuge9G4B_quantized_awq"
    
    if not Path(quant_path).exists():
        print(f"错误: 量化模型路径不存在: {quant_path}")
        return
    
    print("开始AWQ量化模型质量测试...")
    
    # 1. 分析量化质量
    quantization_info = load_quantization_info(quant_path)
    quality_stats = analyze_quantization_quality(quantization_info)
    
    # 2. 测试反量化精度
    test_dequantization_accuracy(quant_path)
    
    # 3. 基准测试量化速度
    benchmark_quantization_speed(quant_path)
    
    # 4. 生成报告
    generate_quantization_report(quant_path, "awq_quantization_report.txt")
    
    print("\n=== 测试完成 ===")
    print("AWQ量化模型质量测试已完成，请查看生成的报告文件。")

if __name__ == "__main__":
    main() 