from diffusers import StableDiffusionPipeline
import torch

# 指定模型名称或路径
# model_name = "/mnt/n0/models/path_to_saved_model_rafa"
model_name = "/mnt/n0/models/stable-diffusion-v1-4"

# 加载模型
original_pipe = StableDiffusionPipeline.from_pretrained(model_name, use_safetensors=False)
original_pipe = original_pipe.to("cuda:1")  # 如果有GPU，将模型加载到CUDA设备上


def model_infer():
    # 定义文本提示
    prompt = "A sks dog"

    # 生成图像
    with torch.no_grad():  # 禁用梯度计算以加速推理
        image = original_pipe(prompt).images[0]  # 获取生成的图像

    # 显示图像
    image.show()  # 在默认图像查看器中显示图像

    # 保存图像到文件
    image.save("output_image.png")

def modules():
    # 定义需要统计的模块列表
    modules = {
        "U-Net": original_pipe.unet,
        "VAE": original_pipe.vae,
        "Text Encoder": original_pipe.text_encoder,
    }

    # 统计每个模块的参数数量
    for module_name, module in modules.items():
        total_params = sum(p.numel() for p in module.parameters())
        print(f"{module_name} - Total parameters: {total_params}")
    # output: 
        # U-Net - Total parameters: 859520964
        # VAE - Total parameters: 83653863
        # Text Encoder - Total parameters: 123060480

def compare():
    # mode_name_="/mnt/n0/models/path_to_saved_model_rafa"
    # mode_name_="/mnt/n0/models/japanese-stable-diffusion/"
    # mode_name_="/mnt/n0/models/output_test/"
    mode_name_="/mnt/n0/models/stable-diffusion/"
    
    
    
    finetuned_pipe = StableDiffusionPipeline.from_pretrained(mode_name_, use_safetensors=False)
    finetuned_pipe = finetuned_pipe.to("cuda:1")
    total_params = 0
    changed_params = 0
    # 定义需要比较的模块列表
    modules_to_compare = {
        "U-Net": (original_pipe.unet, finetuned_pipe.unet),
        "VAE": (original_pipe.vae, finetuned_pipe.vae),
        "Text Encoder": (original_pipe.text_encoder, finetuned_pipe.text_encoder)
    }
    
        # 逐个模块进行比较
    for module_name, (original_module, finetuned_module) in modules_to_compare.items():
        module_total_params = 0
        module_changed_params = 0
        
        # 逐层比较参数
        for original_param, finetuned_param in zip(original_module.parameters(), finetuned_module.parameters()):
            if original_param.shape != finetuned_param.shape:
                print(f"Skipping comparison for {module_name} due to shape mismatch: {original_param.shape} vs {finetuned_param.shape}")
                continue  # 跳过此参数的比较

            # 计算当前模块的总参数数量
            module_total_params += original_param.numel()
            total_params += original_param.numel()
            
            # 将参数转换为相同的精度（float32），确保精度一致
            original_param = original_param.float()
            finetuned_param = finetuned_param.float()

            unequal_elements = torch.sum(original_param != finetuned_param).item()
            module_changed_params += unequal_elements
            changed_params += unequal_elements

        # 输出当前模块的比较结果
        print(f"{module_name} - Total parameters: {module_total_params}")
        print(f"{module_name} - Changed parameters after fine-tuning: {module_changed_params}")
        print(f"{module_name} - Percentage of changed parameters: {module_changed_params / module_total_params * 100:.5f}%\n")

    # 输出整个模型的比较结果
    print(f"Overall Model - Total parameters: {total_params}")
    print(f"Overall Model - Changed parameters after fine-tuning: {changed_params}")
    print(f"Overall Model - Percentage of changed parameters: {changed_params / total_params * 100:.5f}%")

def compare_with_error():

    # 设置容差
    ATOL = 1e-6  # 绝对容差
    RTOL = 1e-5  # 相对容差
    mode_name_="/mnt/n0/models/output_test"
    
    # mode_name_="/mnt/n0/models/japanese-stable-diffusion/"
    
    finetuned_pipe = StableDiffusionPipeline.from_pretrained(mode_name_, use_safetensors=False)
    # finetuned_pipe = finetuned_pipe.to("cuda")
    total_params = 0
    changed_params = 0
    # 定义需要比较的模块列表
    modules_to_compare = {
        "U-Net": (original_pipe.unet, finetuned_pipe.unet),
        "VAE": (original_pipe.vae, finetuned_pipe.vae),
        "Text Encoder": (original_pipe.text_encoder, finetuned_pipe.text_encoder)
    }
    
        # 逐个模块进行比较
    for module_name, (original_module, finetuned_module) in modules_to_compare.items():
        module_total_params = 0
        module_changed_params = 0
        
        # 逐层比较参数
        for original_param, finetuned_param in zip(original_module.parameters(), finetuned_module.parameters()):
            # 计算当前模块的总参数数量
            module_total_params += original_param.numel()
            total_params += original_param.numel()
            
            # 检查该参数是否发生了更改
            # if not torch.equal(original_param, finetuned_param):
            if not torch.allclose(original_param, finetuned_param, atol=ATOL, rtol=RTOL):
                module_changed_params += original_param.numel()
                changed_params += original_param.numel()

        # 输出当前模块的比较结果
        print(f"{module_name} - Total parameters: {module_total_params}")
        print(f"{module_name} - Changed parameters after fine-tuning: {module_changed_params}")
        print(f"{module_name} - Percentage of changed parameters: {module_changed_params / module_total_params * 100:.2f}%\n")

    # 输出整个模型的比较结果
    print(f"Overall Model - Total parameters: {total_params}")
    print(f"Overall Model - Changed parameters after fine-tuning: {changed_params}")
    print(f"Overall Model - Percentage of changed parameters: {changed_params / total_params * 100:.2f}%")


compare()