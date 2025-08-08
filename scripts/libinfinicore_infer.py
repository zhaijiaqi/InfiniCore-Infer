import ctypes
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER
import os


class DataType(ctypes.c_int):
    INFINI_DTYPE_INVALID = 0
    INFINI_DTYPE_BYTE = 1
    INFINI_DTYPE_BOOL = 2
    INFINI_DTYPE_I8 = 3
    INFINI_DTYPE_I16 = 4
    INFINI_DTYPE_I32 = 5
    INFINI_DTYPE_I64 = 6
    INFINI_DTYPE_U8 = 7
    INFINI_DTYPE_U16 = 8
    INFINI_DTYPE_U32 = 9
    INFINI_DTYPE_U64 = 10
    INFINI_DTYPE_F8 = 11
    INFINI_DTYPE_F16 = 12
    INFINI_DTYPE_F32 = 13
    INFINI_DTYPE_F64 = 14
    INFINI_DTYPE_C16 = 15
    INFINI_DTYPE_C32 = 16
    INFINI_DTYPE_C64 = 17
    INFINI_DTYPE_C128 = 18
    INFINI_DTYPE_BF16 = 19


class DeviceType(ctypes.c_int):
    DEVICE_TYPE_CPU = 0
    DEVICE_TYPE_NVIDIA = 1
    DEVICE_TYPE_CAMBRICON = 2
    DEVICE_TYPE_ASCEND = 3
    DEVICE_TYPE_METAX = 4
    DEVICE_TYPE_MOORE = 5
    DEVICE_TYPE_ILUVATAR = 6


class JiugeMetaCStruct(ctypes.Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
    ]


# Define the JiugeWeights struct
class JiugeWeightsCStruct(ctypes.Structure):
    _fields_ = [
        ("nlayer", c_size_t),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("transpose_linear_weights", c_int),
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)),
        ("attn_qkv", POINTER(c_void_p)),
        ("attn_qkv_b", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]


class JiugeModelCSruct(ctypes.Structure):
    pass


class KVCacheCStruct(ctypes.Structure):
    pass

# 这段代码的作用是加载和配置C++推理库的Python接口
def __open_library__():
    """
    加载libinfinicore_infer.so动态链接库并配置C函数的类型签名
    
    这个函数负责：
    1. 从环境变量INFINI_ROOT找到动态库文件路径
    2. 使用ctypes加载C++编译的推理库
    3. 为库中的C函数定义参数类型和返回值类型，确保Python能正确调用C++函数
    """
    # 构建动态库文件的完整路径
    lib_path = os.path.join(
        os.environ.get("INFINI_ROOT"), "lib", "libinfinicore_infer.so"
    )
    # 加载C++动态库
    lib = ctypes.CDLL(lib_path)
    
    # 配置createJiugeModel函数的类型签名
    # 这个函数用于创建大语言模型推理实例
    lib.createJiugeModel.restype = POINTER(JiugeModelCSruct)  # 返回模型指针
    lib.createJiugeModel.argtypes = [
        POINTER(JiugeMetaCStruct),  # 模型元数据（层数、维度等）
        POINTER(JiugeWeightsCStruct),  # 模型权重数据
        DeviceType,  # 设备类型（CPU/GPU等）
        c_int,  # 设备数量
        POINTER(c_int),  # 设备ID数组
    ]
    
    # 配置destroyJiugeModel函数 - 销毁模型实例，释放内存
    lib.destroyJiugeModel.argtypes = [POINTER(JiugeModelCSruct)]
    
    # 配置createKVCache函数 - 创建KV缓存（attention机制需要）
    lib.createKVCache.argtypes = [POINTER(JiugeModelCSruct)]
    lib.createKVCache.restype = POINTER(KVCacheCStruct)
    
    # 配置dropKVCache函数 - 释放KV缓存
    lib.dropKVCache.argtypes = [POINTER(JiugeModelCSruct), POINTER(KVCacheCStruct)]
    
    # 配置inferBatch函数 - 批量推理的核心函数
    lib.inferBatch.restype = None  # 无返回值，结果通过output参数返回
    lib.inferBatch.argtypes = [
        POINTER(JiugeModelCSruct),  # 模型实例
        POINTER(c_uint),  # 输入token序列
        c_uint,  # token总数
        POINTER(c_uint),  # 每个请求的长度
        c_uint,  # 请求数量
        POINTER(c_uint),  # 每个请求的位置信息
        POINTER(POINTER(KVCacheCStruct)),  # KV缓存数组
        POINTER(c_float),  # 温度参数（控制随机性）
        POINTER(c_uint),  # top-k参数（采样范围）
        POINTER(c_float),  # top-p参数（核采样）
        POINTER(c_uint),  # 输出结果数组
    ]

    return lib


# 加载库并创建全局实例
LIB = __open_library__()

# 创建Python函数别名，方便调用
create_jiuge_model = LIB.createJiugeModel      # 创建模型
destroy_jiuge_model = LIB.destroyJiugeModel    # 销毁模型
create_kv_cache = LIB.createKVCache            # 创建KV缓存
drop_kv_cache = LIB.dropKVCache                # 释放KV缓存
infer_batch = LIB.inferBatch                   # 批量推理
