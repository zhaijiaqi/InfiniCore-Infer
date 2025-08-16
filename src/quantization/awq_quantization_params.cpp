#include "awq_quantization_params.hpp"
#include "awq_gemm.hpp"
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

namespace awq_gemm {

// 全局量化参数管理器实例
std::unique_ptr<QuantizationParamsManager> g_quantization_manager;

// 解析tensor字符串 (如 "tensor(0.0052)" 或 "tensor([[0.0011, 0.0012, ...]])")
std::vector<float> QuantizationParamsManager::parseTensorString(const std::string& tensor_str) {
    std::vector<float> result;
    
    // 移除"tensor("和")"包装
    std::string content = tensor_str;
    if (content.find("tensor(") == 0) {
        content = content.substr(7); // 移除"tensor("
        if (content.back() == ')') {
            content.pop_back(); // 移除最后的")"
        }
    }
    
    // 使用正则表达式提取数字
    std::regex number_regex(R"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)");
    std::sregex_iterator iter(content.begin(), content.end(), number_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        try {
            float value = std::stof(iter->str());
            result.push_back(value);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse number: " << iter->str() << std::endl;
        }
    }
    
    return result;
}

// 解析权重参数 - 匹配实际的JSON格式
QuantizationParams QuantizationParamsManager::parseWeightParams(const std::string& json_str) {
    QuantizationParams params;
    
    // 解析scale（同时支持数值 or 字符串形式）
    {
        // 优先尝试解析数值型: "scale": 0.0052
        std::regex scale_num_regex("\\\"scale\\\":\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)");
        std::smatch scale_num_match;
        if (std::regex_search(json_str, scale_num_match, scale_num_regex)) {
            try {
                params.scale.push_back(std::stof(scale_num_match[1].str()));
            } catch (...) {}
        } else {
            // 其次尝试字符串型: "scale": "tensor(0.0052)" 或 "scale": "[ ... ]"
            std::regex scale_str_regex("\\\"scale\\\":\\s*\\\"([^\\\"]*)\\\"");
            std::smatch scale_str_match;
            if (std::regex_search(json_str, scale_str_match, scale_str_regex)) {
                auto parsed = parseTensorString(scale_str_match[1].str());
                if (!parsed.empty()) params.scale = std::move(parsed);
            }
        }
    }
    
    // 解析method
    std::regex method_regex("\\\"method\\\":\\s*\\\"([^\\\"]*)\\\"");
    std::smatch method_match;
    if (std::regex_search(json_str, method_match, method_regex)) {
        params.method = method_match[1].str();
    }
    
    // 解析original_shape
    std::regex shape_regex("\\\"original_shape\\\":\\s*\\[([^\\\\]]*)\\]");
    std::smatch shape_match;
    if (std::regex_search(json_str, shape_match, shape_regex)) {
        std::string shape_str = shape_match[1].str();
        std::regex num_regex(R"(\d+)");
        std::sregex_iterator iter(shape_str.begin(), shape_str.end(), num_regex);
        std::sregex_iterator end;
        for (; iter != end; ++iter) {
            params.original_shape.push_back(std::stoi(iter->str()));
        }
    }
    
    // 解析original_dtype
    std::regex dtype_regex("\\\"original_dtype\\\":\\s*\\\"([^\\\"]*)\\\"");
    std::smatch dtype_match;
    if (std::regex_search(json_str, dtype_match, dtype_regex)) {
        params.original_dtype = dtype_match[1].str();
    }
    
    // 解析compression_ratio
    std::regex comp_regex("\\\"compression_ratio\\\":\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)");
    std::smatch comp_match;
    if (std::regex_search(json_str, comp_match, comp_regex)) {
        params.compression_ratio = std::stof(comp_match[1].str());
    }
    
    // 解析quantization_error
    std::regex error_regex("\\\"quantization_error\\\":\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)");
    std::smatch error_match;
    if (std::regex_search(json_str, error_match, error_regex)) {
        params.quantization_error = std::stof(error_match[1].str());
    }
    
    // 解析max_abs_value
    std::regex max_regex("\\\"max_abs_value\\\":\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)");
    std::smatch max_match;
    if (std::regex_search(json_str, max_match, max_regex)) {
        params.max_abs_value = std::stof(max_match[1].str());
    }
    
    // 解析scale_value（同时支持数值 or 字符串形式，作为兜底）
    if (params.scale.empty()) {
        std::regex scale_val_num_regex("\\\"scale_value\\\":\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)");
        std::smatch scale_val_num_match;
        if (std::regex_search(json_str, scale_val_num_match, scale_val_num_regex)) {
            try {
                params.scale.push_back(std::stof(scale_val_num_match[1].str()));
            } catch (...) {}
        }
        if (params.scale.empty()) {
            std::regex scale_val_str_regex("\\\"scale_value\\\":\\s*\\\"([^\\\"]*)\\\"");
            std::smatch scale_val_str_match;
            if (std::regex_search(json_str, scale_val_str_match, scale_val_str_regex)) {
                auto parsed = parseTensorString(scale_val_str_match[1].str());
                if (!parsed.empty()) params.scale = std::move(parsed);
            }
        }
    }
    
    // 设置默认值
    if (params.method == "int8") {
        params.bits = 8;
        params.group_size = 128;
        params.per_channel = false;
        // INT8对称量化不使用zero_point
        params.zero_point = std::vector<float>(params.scale.size(), 0.0f);
        // 如果依然未解析到scale，依据对称量化公式做一个安全兜底（不会影响AWQ/gptq）
        if (params.scale.empty()) {
            // 无法从JSON取到，退化为一个保守的默认scale，避免用1.0导致数值爆炸
            params.scale.push_back(1.0f / 127.0f);
        }
    }
    
    return params;
}

// 简单的JSON文件解析
bool QuantizationParamsManager::parseJsonFile(const std::string& file_path, std::unordered_map<std::string, QuantizationParams>& params_map) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open quantization file: " << file_path << std::endl;
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    // 简单的JSON解析 - 查找权重名称和对应的参数
    std::regex weight_regex("\\\"([^\\\"]+\\.weight)\\\":\\s*\\{([^}]+(?:\\{[^}]*\\}[^}]*)*)\\}");
    std::sregex_iterator iter(content.begin(), content.end(), weight_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::string weight_name = iter->str(1);
        std::string weight_params = iter->str(2);
        QuantizationParams params = parseWeightParams(weight_params);
        params_map[weight_name] = params;
    }
    
    return true;
}

// 从JSON文件加载量化参数
bool QuantizationParamsManager::loadFromFile(const std::string& file_path) {
    try {
        bool success = parseJsonFile(file_path, params_map);
        
        if (success) {
            std::cout << "Loaded quantization parameters for " << params_map.size() << " weights from " << file_path << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading quantization parameters: " << e.what() << std::endl;
        return false;
    }
}

// 获取指定权重的量化参数
const QuantizationParams* QuantizationParamsManager::getParams(const std::string& weight_name) const {
    auto it = params_map.find(weight_name);
    if (it != params_map.end()) {
        return &(it->second);
    }
    return nullptr;
}

// 检查是否包含指定权重的量化参数
bool QuantizationParamsManager::hasParams(const std::string& weight_name) const {
    return params_map.find(weight_name) != params_map.end();
}

// 获取所有权重名称
std::vector<std::string> QuantizationParamsManager::getAllWeightNames() const {
    std::vector<std::string> names;
    names.reserve(params_map.size());
    for (const auto& pair : params_map) {
        names.push_back(pair.first);
    }
    return names;
}

// 创建AWQ量化参数
AWQQuantizeParams QuantizationParamsManager::createAWQParams(const std::string& weight_name) const {
    AWQQuantizeParams awq_params;
    
    const QuantizationParams* params = getParams(weight_name);
    if (params) {
        // 使用第一个scale值作为全局scale
        if (!params->scale.empty()) {
            awq_params.scale = params->scale[0];
        } else {
            awq_params.scale = 1.0f / 127.0f; // 更安全的默认值
        }
        
        awq_params.group_size = params->group_size;
        awq_params.per_channel = params->scale.size() > 1;
        
        // 如果有多个scale值，使用它们作为分组scale
        if (params->per_channel) {
            awq_params.group_scales = params->scale;
        }
    } else {
        // 默认参数
        awq_params.scale = 1.0f / 127.0f;
        awq_params.group_size = 128;
        awq_params.per_channel = false;
    }
    
    return awq_params;
}

// 初始化全局量化参数管理器
bool initializeQuantizationManager(const std::string& model_path) {
    std::string quant_file_path = model_path + "/quantization_info.json";
    
    g_quantization_manager = std::make_unique<QuantizationParamsManager>();
    bool success = g_quantization_manager->loadFromFile(quant_file_path);
    
    if (success) {
        std::cout << "Quantization parameters loaded successfully from: " << quant_file_path << std::endl;
    } else {
        std::cout << "Warning: Failed to load quantization parameters from: " << quant_file_path << std::endl;
        std::cout << "Model will run without quantization." << std::endl;
    }
    
    return success;
}

// 获取全局量化参数管理器
QuantizationParamsManager* getQuantizationManager() {
    return g_quantization_manager.get();
}

} // namespace awq_gemm 