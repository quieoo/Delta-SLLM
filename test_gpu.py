import ray

# 初始化 Ray，如果 Ray 集群已经在外部启动，则不需要执行此步
ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1)
def check_cuda():
    import torch
    return torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0)

# 调用远程函数并打印结果
print(ray.get(check_cuda.remote()))

# 关闭 Ray
ray.shutdown()
