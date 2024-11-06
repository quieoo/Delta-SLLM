from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

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



# 本地模型路径
model_path = "/mnt/n0/models/falcon_1b"  # 替换为实际本地路径
model_size = get_model_size(model_path)
available_memory = get_gpu_memory()
print(f"Model size: {model_size} MB")
print(f"Available GPU memory: {available_memory} MB")

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

