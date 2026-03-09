# Numpy和Tensor互相转换
import torch


# tensor -> numpy
def tensor_to_numpy():
    t1 = torch.tensor([1, 2, 3])
    n1 = t1.numpy()  # 浅拷贝，会共享内存！！！
    print(f'n1 = {n1}, type(n1) = {type(n1)}')

# torch -> numpy
def numpy_to_tensor():
    n1 = [1, 2, 3]
    t2 = torch.from_numpy(n1)  # 整数转tensor后默认 int64，也会共享内存！！！
    t3 = t2.type(torch.float32)  # 一般都需要 float32，需要转一下
    t3 = torch.from_numpy(n1).type(torch.float32)  # 链式编程

    t4 = torch.tensor(n1)  # 这种方式不共享内存

    n1[0] = 100
    print(f'n1 {n1}')  # 100, 2, 3
    print(f't2 {t2}')  # 100, 2, 3
    print(f't4 {t4}')  # 1, 2, 3


# 提取torch
def get_v_from_tensor():
    t1 = torch.tensor(1)
    n2 = t1.item()  # 变成标量tensor (Scalar), 要求tensor里必须是1个数值
    print(f'n2 {n2}, type(n2) = {type(n2)}')

# 乘法
def multiply():
    t1 = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    
    t3 = t1 * t2 # 点乘 hadamard
    t4 = t1 @ t2 # 矩阵乘法

# 计算api
def compute_api():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]]).type(torch.float32)
    print(f'sum {t1.sum(dim=0)}') # 按列求和
    print(f'sum {t1.sum()}')

    print(f'max {t1.max(dim=0)}') # 按列求最大
    print(f'max {t1.max()}')

    print(f'mean {t1.mean(dim=0)}') # 按列求平均，需要是float类型，int会报错
    print(f'mean {t1.mean()}')

    print(t1 ** 3)  # pow 求n次幂
    print(t1.sqrt())  # 平方根
    print(t1.exp())  # e^x次方，x是矩阵中的数
    print(t1.log())
    print(t1.log2())
    print(t1.log10())


# 索引
def index():
    torch.manual_seed(42)
    t1 = torch.randint(1, 10, (5,5))
    print(f't1: {t1}')

    # ------- 1. 值索引-----------------
    # tensor[行，列] 进行索引
    print(t1[1])  # 也是所有列的第2行数据
    print(t1[1,:])  # 所有列的第2行数据
    print(t1[:,2])  # 所有行的第3列数据

    # ------- 2. 列表索引 -------------
    # tensor[[行值], [列值]]
    print(t1[[1,3], [2,4]])  # 返回(1,2), (3,4) 位置元素
    print(t1[[[0],[1]], [1,2]])  # 0对后面1，2；1也对后面1，2

    # -------- 3. 范围索引---------------
    print(t1[:3,:2])  # 前3行，前2列
    print(t1[1::2, ::2])  # 所有奇数行，偶数列

    # -------- 4. 多维索引---------------
    t2 = torch.randint(1, 10, (2, 3, 4))

    print(t2[0,:,:])  # 获取0轴上的第1个数据
    print(t2[:,0,:])  # 获取1轴上的第1个数据

