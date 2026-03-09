import torch
import numpy as np
from wandb.sdk.internal.profiler import torch_trace_handler

def init_tensor():
    # 创建 tensor
    ## 小写：根据数据创建
    ## 大写：创建指定维度的tensor
    t1 = torch.tensor(10)
    print(f't1: {t1}, type: {type(t1)}')


    # list 转 tensor
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2: {t2}, type: {type(t2)}')

    # numpy nd数组 转 tensor
    data = np.random.randint(0, 10, size=(2,3))
    t3 = torch.tensor(data)
    t3 = torch.tensor(data, dtype=torch.float)
    print(f't3: {t3}, type: {type(t3)}')

    # 直接创建 指定维度 (两行三列) tensor
    t4 = torch.Tensor(2, 3) # 可以指定形状，但是不能指定类型
    t5 = torch.IntTensor(2, 3)
    t6 = torch.FloatTensor(2, 3)

    # 输出时不显示type，是默认 float32，原因是省空间（所以不用 float64）

def init_ones_zeros():
    # 创建指定 0 1 tensor
    t1 = torch.ones(2, 3)  # 创建2行3列tensor
    t2 = torch.ones_like(t1)  # 传入tensor，创建与其相同size的tensor

    t3 = torch.zero(2, 3)
    t4 = torch.zeros_like(t3)

    # 填充指定值的 tensor
    t5 = torch.full((2,3),255)
    t6 = torch.full_like(t3, 255)

def init_linear_tensor():
    # 创建指定范围的线性tensor
    t1 = torch.arange(0, 10, 2)  # 包左不包右，第3个参数是步长
    print(f't1: {t1}, type: {type(t1)}')

    # 线性等差数列
    t2 = torch.linspace(1, 10, 5)  # 这里第3个参数是元素个数
    print(f't2: {t2}, type: {type(t2)}')

def init_random_tensor():
    torch.initial_seed()  # 默认采用当前系统的时间戳作为随机种子，每时每刻都不同
    torch.manual_seed(42)  # 设定随机种子

    # 随机 tensor，均匀分布在(0,1)
    t1 = torch.rand((2,3)) # 创建2行3列随机矩阵
    print(f't1: {t1}, type: {type(t1)}')
    # 随机 tensor，正态分布
    t2 = torch.randn((2,3))
    # 随机 tensor，正态分布
    t3 = torch.randint(1, 10, (2,3))
    print(f't3: {t3}, type: {type(t3)}')






