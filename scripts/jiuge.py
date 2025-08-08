from typing import List
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


class JiugeWeightsImpl(JiugeWeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
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
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_logits) * scale_input
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm) * scale_output
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        def qkv_slices(_i):
            _Q = (
                state_dict[naming.attn_q(_i)]
                .reshape([nh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _K = (
                state_dict[naming.attn_k(_i)]
                .reshape([nkvh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
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

        self.attn_o_tensor = [
            (
                state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.attn_o(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_o
            for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[naming.up(_i)][_start:_end, :])
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

        self.ffn_down_tensor = [
            (
                state_dict[naming.down(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, di // ndev])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.down(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_down
            for i in range(nlayer)
        ]
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


class JiugeForCauslLM:
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_

        print("Loading model weights to host...")
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
            self.weights = JiugeWeightsImpl(
                self.meta,
                LlamaWeightsNaming(),
                model.state_dict(),
                ndev=ndev,
                transpose_weight=transpose_weight,
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
                self.weights = JiugeWeightsImpl(
                    self.meta,
                    LlamaWeightsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
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
                self.weights = JiugeWeightsImpl(
                    self.meta,
                    LlamaWeightsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
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
                self.weights = JiugeWeightsImpl(
                    self.meta,
                    LlamaWeightsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path
                )
        else:
            raise ValueError("Unsupported model architecture")

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

        print(f"Creating model on {ndev} devices...")
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
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        """
        执行一轮批量推理
        
        参数:
            tasks: 推理任务列表，每个任务包含输入tokens和采样参数
            
        返回:
            list: 每个任务生成的下一个token ID列表
        """
        # 创建用于存储输出结果的C数组，长度等于任务数量
        output = (c_uint * len(tasks))()
        
        # 将多个推理任务打包成批量输入格式
        batch_inputs = JiugeBatchedTask(tasks)
        
        # 调用底层C接口执行批量推理
        # model_instance: 模型实例
        # *batch_inputs.input_args(): 展开批量输入参数（tokens、长度、位置等）
        # output: 存储每个任务生成的token结果
        infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        
        # 将C数组转换为Python列表并返回
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        """
        生成文本的主函数
        
        参数:
            input_content: 用户输入的文本内容
            max_steps: 最大生成token数量，即模型最多生成多少个新token
            topp_: top-p采样参数，控制候选词的概率累积阈值
            topk_: top-k采样参数，选择概率最高的k个词
            temperature_: 采样温度，控制生成的随机性
        
        返回:
            (output_content, avg_time): 生成的文本内容和平均每步用时
        """
        # 使用tokenizer将用户输入格式化为对话模板
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,  # 添加生成提示符
            tokenize=False,  # 不进行tokenize，返回字符串
        )
        print(input_content, end="", flush=True)  # 打印输入内容
        
        # 将格式化后的文本编码为token序列
        tokens = self.tokenizer.encode(input_content)
        
        # 创建推理任务，包含所有必要参数
        infer_task = InferTask(
            0,  # 任务ID
            tokens,  # 输入token序列
            self.max_context_len(),  # 最大上下文长度
            temperature_,  # 采样温度
            topk_,  # top-k参数
            topp_,  # top-p参数
            self.eos_token_id,  # 结束token的ID
        )
        # 为推理任务绑定KV缓存（用于attention计算的优化）
        infer_task.bind_kvcache(KVCache(self))

        # 初始化计数器和累积变量
        steps = 0  # 实际执行的步数
        total_time = 0  # 总用时（不包括第一步预填充）
        output_content = ""  # 累积的输出内容

        # 逐步生成token
        for step_i in range(max_steps):
            start_time = time.time()
            # 执行一轮批量推理，获取下一个token
            output_tokens = self.batch_infer_one_round([infer_task]) # batch_infer_one_round 支持 batch 推理，但这里只传入一个任务
            end_time = time.time()
            steps += 1
            
            # 将输出token转换为可读文本
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace("▁", " ")  # 替换特殊空格符号
                .replace("<0x0A>", "\n")  # 替换换行符
            )
            output_content += output_str  # 累积输出内容
            print(output_str, end="", flush=True)  # 实时打印生成的文本
            
            # 检查是否生成了结束token
            if output_tokens[0] in self.eos_token_id:
                break
            
            # 将生成的token添加到推理任务中，为下一步推理做准备
            infer_task.next(output_tokens[0])

            # 统计用时（跳过第一步，因为第一步包含预填充时间）
            if step_i > 0:
                total_time += end_time - start_time

        print("\n")  # 生成完成后换行
        # 计算平均每步用时（毫秒）
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        # 释放KV缓存资源
        infer_task._kv_cache.drop(self)
        return output_content, avg_time
    def destroy_model_instance(self):
        destroy_jiuge_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
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
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = JiugeForCauslLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
