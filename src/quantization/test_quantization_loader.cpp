#include "awq_quantization_params.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::string model_path = "/home/halozjq/models/jiuge9G4B_quantized_int8";
    
    std::cout << "正在加载量化参数..." << std::endl;
    
    // 初始化量化参数管理器
    bool success = awq_gemm::initializeQuantizationManager(model_path);
    
    if (!success) {
        std::cerr << "Failed to initialize quantization manager!" << std::endl;
        return -1;
    }
    
    // 获取量化参数管理器
    awq_gemm::QuantizationParamsManager* manager = awq_gemm::getQuantizationManager();
    if (!manager) {
        std::cerr << "Failed to get quantization manager!" << std::endl;
        return -1;
    }
    
    // 获取所有权重名称
    std::vector<std::string> weight_names = manager->getAllWeightNames();
    std::cout << "找到 " << weight_names.size() << " 个量化权重" << std::endl;
    
    // 显示前几个权重的详细信息
    std::cout << "\n前5个权重的量化信息:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (int i = 0; i < std::min(5, (int)weight_names.size()); ++i) {
        const std::string& weight_name = weight_names[i];
        const awq_gemm::QuantizationParams* params = manager->getParams(weight_name);
        
        if (params) {
            std::cout << "权重名称: " << weight_name << std::endl;
            std::cout << "量化方法: " << params->method << std::endl;
            std::cout << "量化位数: " << params->bits << std::endl;
            std::cout << "原始形状: [";
            for (size_t j = 0; j < params->original_shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << params->original_shape[j];
            }
            std::cout << "]" << std::endl;
            std::cout << "原始数据类型: " << params->original_dtype << std::endl;
            
            if (!params->scale.empty()) {
                std::cout << "缩放因子: " << std::fixed << std::setprecision(6) << params->scale[0] << std::endl;
            }
            
            std::cout << "压缩比: " << std::fixed << std::setprecision(2) << params->compression_ratio << "x" << std::endl;
            std::cout << "量化误差: " << std::scientific << std::setprecision(6) << params->quantization_error << std::endl;
            std::cout << "最大绝对值: " << std::fixed << std::setprecision(6) << params->max_abs_value << std::endl;
            std::cout << std::string(80, '-') << std::endl;
        }
    }
    
    // 测试AWQ参数创建
    if (!weight_names.empty()) {
        std::cout << "\n测试AWQ参数创建:" << std::endl;
        const std::string& test_weight = weight_names[0];
        awq_gemm::AWQQuantizeParams awq_params = manager->createAWQParams(test_weight);
        
        std::cout << "权重: " << test_weight << std::endl;
        std::cout << "AWQ缩放因子: " << awq_params.scale << std::endl;
        std::cout << "AWQ分组大小: " << awq_params.group_size << std::endl;
        std::cout << "AWQ通道量化: " << (awq_params.per_channel ? "是" : "否") << std::endl;
    }
    
    // 统计信息
    std::cout << "\n量化统计信息:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    int int8_count = 0;
    float total_compression = 0.0f;
    float total_error = 0.0f;
    
    for (const auto& weight_name : weight_names) {
        const awq_gemm::QuantizationParams* params = manager->getParams(weight_name);
        if (params) {
            if (params->method == "int8") {
                int8_count++;
                total_compression += params->compression_ratio;
                total_error += params->quantization_error;
            }
        }
    }
    
    std::cout << "INT8量化权重数量: " << int8_count << std::endl;
    if (int8_count > 0) {
        std::cout << "平均压缩比: " << std::fixed << std::setprecision(2) 
                  << (total_compression / int8_count) << "x" << std::endl;
        std::cout << "平均量化误差: " << std::scientific << std::setprecision(6) 
                  << (total_error / int8_count) << std::endl;
    }
    
    std::cout << "\n量化参数加载测试完成!" << std::endl;
    return 0;
} 