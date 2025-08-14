#ifndef INFINICORE_INFER_AWQ_QUANTIZATION_PARAMS_HPP
#define INFINICORE_INFER_AWQ_QUANTIZATION_PARAMS_HPP

#include "awq_gemm.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace awq_gemm {

// 量化参数结构
struct QuantizationParams {
    std::vector<float> scale;           // 缩放因子
    std::vector<float> zero_point;      // 零点
    std::string method;                 // 量化方法 (awq, gptq等)
    int bits;                           // 量化位数
    int group_size;                     // 分组大小
    bool per_channel;                   // 是否使用通道量化
    std::vector<int> original_shape;    // 原始形状
    std::string original_dtype;         // 原始数据类型
    
    // 量化统计信息
    float compression_ratio;            // 压缩比
    float quantization_error;           // 量化误差
    float max_abs_value;                // 最大绝对值
    
    // 重要性信息
    struct ImportanceInfo {
        int group_start;
        int group_end;
        float importance;
        int effective_bits;
        std::string quant_method;
    };
    std::vector<ImportanceInfo> importance_info;
};

// 量化参数管理器
class QuantizationParamsManager {
private:
    std::unordered_map<std::string, QuantizationParams> params_map;
    std::string model_path;

public:
    // 从JSON文件加载量化参数
    bool loadFromFile(const std::string& file_path);
    
    // 获取指定权重的量化参数
    const QuantizationParams* getParams(const std::string& weight_name) const;
    
    // 检查是否包含指定权重的量化参数
    bool hasParams(const std::string& weight_name) const;
    
    // 获取所有权重名称
    std::vector<std::string> getAllWeightNames() const;
    
    // 创建AWQ量化参数
    AWQQuantizeParams createAWQParams(const std::string& weight_name) const;
    
    // 解析tensor字符串 (如 "tensor([[0.0011, 0.0012, ...]])")
    static std::vector<float> parseTensorString(const std::string& tensor_str);
    
    // 简单的JSON解析函数
    static bool parseJsonFile(const std::string& file_path, std::unordered_map<std::string, QuantizationParams>& params_map);
    static QuantizationParams parseWeightParams(const std::string& json_str);
};

// 全局量化参数管理器实例
extern std::unique_ptr<QuantizationParamsManager> g_quantization_manager;

// 初始化全局量化参数管理器
bool initializeQuantizationManager(const std::string& model_path);

// 获取全局量化参数管理器
QuantizationParamsManager* getQuantizationManager();

} // namespace awq_gemm

#endif // INFINICORE_INFER_AWQ_QUANTIZATION_PARAMS_HPP 