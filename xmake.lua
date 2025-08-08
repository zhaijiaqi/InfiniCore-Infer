-- 设置 INFINI_ROOT 环境变量，如果未设置则使用默认路径
-- 在 Windows 系统使用 HOMEPATH，其他系统使用 HOME
local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

-- 定义 infinicore_infer 目标，这是一个用于大语言模型推理的核心库
-- 编译后会生成 libinfinicore_infer.so 动态链接库文件，供 Python 脚本调用
target("infinicore_infer")
    -- 设置为共享库类型，在 Linux 下生成 libinfinicore_infer.so，在 Windows 下生成 infinicore_infer.dll
    set_kind("shared")

    -- 添加本地头文件目录，仅对当前目标可见
    add_includedirs("include", { public = false })
    -- 添加 INFINI 框架的头文件目录，对依赖此目标的其他目标也可见
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    -- 添加量化相关源文件
    add_files("src/quantization/*.cpp")
    add_includedirs("include")

    -- 安装配置：将编译好的 libinfinicore_infer.so 和头文件安装到 INFINI_ROOT 目录
    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()
