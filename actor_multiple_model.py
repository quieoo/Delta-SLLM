import os
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from fastapi import FastAPI
import uvicorn
import time
import psutil
from pydantic import BaseModel

ray.init(namespace="sllm")

# 定义Ray Actor，用于加载多个模型并支持动态模型选择
@ray.remote(num_gpus=1)
class LLMModel:
    def __init__(self, model_dir="/mnt/n0/models/"):
        # 检查模型目录
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist.")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 遍历目录并加载所有模型到主机内存
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            if os.path.isdir(model_path):
                print(f"Loading model: {model_name}")
                
                # 记录加载前的内存使用
                mem_before = process.memory_info().rss / (1024 ** 2)  # 转换为MB
                
                start_time = time.time()
                
                # 加载tokenizer和模型到主机内存
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to("cpu").eval()
                
                end_time = time.time()
                
                # 记录加载后的内存使用
                mem_after = process.memory_info().rss / (1024 ** 2)  # 转换为MB
                mem_used = mem_after - mem_before
                
                print(f"Model {model_name} loaded to CPU in {end_time - start_time:.2f} seconds.")
                print(f"Memory used by model {model_name}: {mem_used:.2f} MB")
                
                # 保存tokenizer和模型
                self.models[model_name] = {"tokenizer": tokenizer, "model": model}
        
        print(f"Initialize done. Loaded {len(self.models)} models.")

    def generate(self, model_name, prompt):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} is not loaded.")
        
        print(f"Generating response using model: {model_name}")
        
        # 将模型和tokenizer加载到GPU
        prepare_time = time.time()
        tokenizer = self.models[model_name]["tokenizer"]
        model = self.models[model_name]["model"].to(self.device)
        
        # 记录生成时间
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            top_p=0.9
        )
        
        # 将生成结果解码
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        print(f"Generate response, prepare time {start_time - prepare_time} seconds, inference time {end_time - start_time} seconds.")
        # print(f"Generated response in {end_time - start_time} seconds.")
        
        # 生成完毕后将模型移回主机内存，以节省GPU资源
        del model
        torch.cuda.empty_cache()
        
        return decoded

# 初始化FastAPI应用
app = FastAPI()

class GenerateRequest(BaseModel):
    model_name: str
    prompt: str

# 使用Ray中的Actor初始化所有模型
llm_actor = LLMModel.options(name="llm_actor", lifetime="detached").remote()

@app.post("/generate")
async def generate(request: GenerateRequest):
    # 调用Actor生成方法并指定模型和输入
    response = await llm_actor.generate.remote(request.model_name, request.prompt)
    return {"response": response}

# 启动FastAPI应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
