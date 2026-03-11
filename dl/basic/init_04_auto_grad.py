import torch

# 只有标量张量才能求导，且大多数都是 float

# 更新一次梯度
def update_once():
    # 1. 设定参数
    w = torch.tensor(10, requires_grad=True, dtype=torch.float)

    # 2. 定义loss函数
    loss = 2 * w ** 2
    print(type(loss.grad_fn))

    # 3. 计算梯度
    loss.sum().backward()  # .sum保证loss是标量
    # 计算完毕后，会记录到 w.grad属性中

    # 4.代入权重更新公式: w_new = w_old - learningRate * grad
    w.data = w.data - 0.01 * w.grad

    print(f'updated weight {w.data}')

# 循环更新梯度
def rotate_grad():
    w = torch.tensor(10, requires_grad=True, dtype=torch.float)
    loss = w**2 + 20

    # 1. 迭代100次，求最优解
    for i in range(1, 101):
        loss = w**2 + 20  # 正向计算
        loss.backward()  # 反向传播
        w.data = w.data - 0.01 * w.grad
        w.grad.zero_()  # 梯度清零，因为默认梯度累加
        print(f'updated weight {w.data}')


# tensor设置自动微分后，不可以转numpy，需要detach
# detach 的tensor们共享空间
def grad_to_numpy():
    t1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)

    # 通过detach拷贝一份，然后转换，但是 detach后的tensor和原tensor共享空间
    t2 = t1.detach()
    n2 = t2.numpy()  # 可以转numpy
