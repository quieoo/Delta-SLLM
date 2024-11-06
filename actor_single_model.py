import os
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from fastapi import FastAPI
import uvicorn
import time
from pydantic import BaseModel

@ray.remote(num_gpus=1)
class LLMModel:
    def __init__(self, model_path):
        # 检查模型路径
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")

        # 记录加载模型的时间
        start_time = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载 tokenizer 和模型到 GPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        load_model_time=time.time()
        model=AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

        to_time=time.time()
        self.model=model.to(self.device)

        # self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
        end_time = time.time()

        print(f"Prepare model, loading from disk takes {to_time - load_model_time} seconds, load into GPU takes {end_time - to_time} seconds.")

    def generate(self, prompt):
        # 开始生成的时间记录
        start_time = time.time()
        
        # 将输入加载到模型的设备上
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_time = time.time()
        
        # 使用模型生成文本
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            do_sample=True,        # 启用采样模式以获得更随机的输出
            top_k=50,              # 限制候选词的个数，提升生成速度
            temperature=1.0,        # 设置温度参数以控制生成的多样性
            top_p=0.9
        )
        generate_time = time.time()
        
        # 解码生成的输出
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        decode_time = time.time()
        
        # 打印时间记录
        # print("Tokenization time:", token_time - start_time)
        print("Generation time:", generate_time - token_time)
        # print("Decoding time:", decode_time - generate_time)

        return decoded

# 使用本地模型路径初始化Actor
model_path = "/mnt/n0/models/llama_3.2_3B/"
llm_actor = LLMModel.options(name="llm_actor", lifetime="detached").remote(model_path)

# 初始化 FastAPI
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    # 调用 Actor 的生成方法
    response = await llm_actor.generate.remote(request.prompt)
    return {"response": response}

# 启动 FastAPI 应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
