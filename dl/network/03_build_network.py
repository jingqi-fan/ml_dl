import torch
import torch.nn as nn
from torchsummary import summary

# 搭建模型
class ModelDemo(nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层 1，输入特征3，输出特征3
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层 2，输入特征 3 输出特征2
        self.linear2 = nn.Linear(3, 2)
        # 输出层, in 2 out 2
        self.output = nn.Linear(2, 2)

        # 可选，对隐藏层参数初始化
        # 隐藏层 1
        nn.init.xavier_normal(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 隐藏层 2
        nn.init.kaiming_normal(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # 前向传播: 输入层 -> 隐藏层 -> 输出层
    def forward(self, x):
        # 分解版写法
        # x = self.linear1(x)  # 加权求和
        # x = torch.sigmoid(x)  # 激活函数
        # ...

        # 合并版
        x = torch.sigmoid(self.linear1(x))  # 隐 1
        x = torch.relu(self.linear2(x))  # 隐 2
        x = torch.softmax(self.output(x), dim=-1)  # output, dim=-1意味按行计算

        return x

# 模型训练
def train():
    my_model = ModelDemo()
    data = torch.randn(size=(5,3))
    output = my_model(data)

    # 计算模型参数：参1 模型对象，惨2 输入数据维度
    summary(my_model, input_size=(5,3))





# 模型测试
















