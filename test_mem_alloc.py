import numpy as np

def test_available_memory():
    allocated_memory = 0
    step_size = 100  # 每次申请100MB的内存
    allocated_blocks = []

    try:
        while True:
            # 尝试申请内存块（100MB）
            block = np.zeros((step_size * 1024 * 1024 // 8,), dtype=np.float64)  # float64占8字节
            allocated_blocks.append(block)
            allocated_memory += step_size
            print(f"Allocated {allocated_memory} MB")
    except MemoryError:
        print("Memory allocation failed.")
        print(f"Total allocated memory: {allocated_memory} MB")
    finally:
        # 释放内存
        del allocated_blocks

test_available_memory()
