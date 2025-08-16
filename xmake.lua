local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

target("infinicore_infer")
    set_kind("shared")

    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_files("src/quantization/*.cpp")
    add_includedirs("include")

    -- 添加CUDA支持（如果可用）
    if os.getenv("CUDA_PATH") or has_config("cuda") then
        add_files("src/quantization/*.cu")
        add_includedirs("/usr/local/cuda/include")
        add_linkdirs("/usr/local/cuda/lib64")
        add_links("cuda", "cudart", "cublas")
        set_languages("cxx17", "cuda")
        add_defines("CUDA_ENABLED")
        -- 修复CUDA编译标志
        add_cuflags("-arch=sm_75", "-std=c++17", {force = true})
        set_policy("check.auto_ignore_flags", false)
    end

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()
