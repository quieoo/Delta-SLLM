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

@ray.remote(num_gpus=1)
class LLMModel:
    def __init__(self, model_dir="/mnt/n0/models/"):
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist.")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        
        process = psutil.Process(os.getpid())
        
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            if os.path.isdir(model_path):
                print(f"Loading model: {model_name}")
                
                mem_before = process.memory_info().rss / (1024 ** 2)  
                
                start_time = time.time()
                
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to("cpu").eval()
                
                end_time = time.time()
                
                mem_after = process.memory_info().rss / (1024 ** 2) 
                mem_used = mem_after - mem_before
                
                print(f"Model {model_name} loaded to CPU in {end_time - start_time:.2f} seconds.")
                print(f"Memory used by model {model_name}: {mem_used:.2f} MB")
                
                self.models[model_name] = {"tokenizer": tokenizer, "model": model}
        
        print(f"Initialize done. Loaded {len(self.models)} models.")

    def generate(self, model_name, prompt):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} is not loaded.")
        
        print(f"Generating response using model: {model_name}")
        
        prepare_time = time.time()
        tokenizer = self.models[model_name]["tokenizer"]
        model = self.models[model_name]["model"].to(self.device)
        
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = model.generate(
            **inputs,
            max_length=50,       # 设为较小值
            do_sample=False,     # 不采样，生成更确定的输出
            top_k=10,            # 降低候选词数量
            temperature=0.7,     # 降低随机性
            top_p=0.95           # 增加筛选阈值
        )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        print(f"Generate response, prepare time {start_time - prepare_time} seconds, inference time {end_time - start_time} seconds.")
        # print(f"Generated response in {end_time - start_time} seconds.")
        
        del model
        torch.cuda.empty_cache()
        
        return decoded

app = FastAPI()

class GenerateRequest(BaseModel):
    model_name: str
    prompt: str

llm_actor = LLMModel.options(name="llm_actor", lifetime="detached").remote()

@app.post("/generate")
async def generate(request: GenerateRequest):
    response = await llm_actor.generate.remote(request.model_name, request.prompt)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
