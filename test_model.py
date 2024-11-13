from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn as nn
import time

def get_model_size(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model_size = sum(param.numel() for param in model.parameters()) * 4 / (1024 ** 2)  # 4 bytes per float32
    del model  # 卸载模型以释放内存
    torch.cuda.empty_cache()
    return model_size  # 返回模型大小（以MB为单位）

def get_gpu_memory():
    gpu_id = 0  # 如果有多张 GPU，可以选择对应的 GPU ID
    gpu_properties = torch.cuda.get_device_properties(gpu_id)
    total_memory = gpu_properties.total_memory / (1024 ** 2)  # 以MB为单位
    reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024 ** 2)
    available_memory = total_memory - reserved_memory
    return available_memory

# '猴子补丁'方法替换原本的torch.to()
# 备份原始的 `to` 方法
original_to = torch.nn.Module.to

# 自定义的 `to` 方法
def custom_to(self, device, *args, **kwargs):
    start_time = time.time()
    print(f"Transferring model to {device}...")

    # 调用原始的 `to` 方法，将模型转移到指定设备
    result = original_to(self, device, *args, **kwargs)

    # 记录并打印转移所需的时间
    end_time = time.time()
    print(f"Model transferred to {device} in {end_time - start_time:.2f} seconds.")

    return result

# 替换 `torch.nn.Module.to` 方法
torch.nn.Module.to = custom_to

# 本地模型路径
model_path = "/public/home/shenzhaoyan/zhu/llama3.1_instruct/"  # 替换为实际本地路径
# model_size = get_model_size(model_path)
# available_memory = get_gpu_memory()
# print(f"Model size: {model_size} MB")
# print(f"Available GPU memory: {available_memory} MB")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

model=model.to("cuda")

# 测试输入文本
input_text = "Hello, how are you?"

# 将文本转换为 BERT 的输入格式
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,        # 启用采样模式以获得更随机的输出
    top_k=50,              # 限制候选词的个数，提升生成速度
    temperature=1.0,        # 设置温度参数以控制生成的多样性
    top_p=0.9
)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Decoded output:", decoded)

