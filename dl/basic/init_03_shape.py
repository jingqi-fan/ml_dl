# tensor 形状操作
import torch

# reshape: 在保证 tensor 数据不变的情况下，改变数据维度，将其转换成指定的
# squeeze: 删除形状为 1 的维度 (降维)
# unsqueeze: 添加形状为 1 的维度 (生维)
# transpose: 一次只能交换两个维度
# permute: 一次可以交换多个维度
# view: 修改连续的 tensor 的形状
    ## 连续？ 存储顺序 与 tensor 逻辑顺序是否一致。比如经过permute，就不能review
# continuous: 把不连续的 tensor 变成连续的 (改变存储)
# is_continuous: 判断是否连续

# cat:  在指定的维度拼接 tensor，不改变维度数。拼接tensor。除了拼接的维度外，其他维度需要保持一致
# stack: 在一个新的维度上拼接一系列 tensor，会改变维度数，所有维度必须保持一致



torch.manual_seed(42)
t1 = torch.randint(1, 10, (2,3))
print(t1)
print(f'shape {t1.shape}, row {t1.shape[0]}, column {t1.shape[1]}, last dim {t1.shape[-1]}')  # 查看形状

# reshape
t2 = t1.reshape(3, 2)  # 转成 3 行 3 列，数据是挨个塞进去的
# t3 = t1.reshape(2, 5)  # 如果转后丢失数据，会报错


# unsqueeze
t4 = t1.unsqueeze(0)  # 在0维上，增加一个维度 (2,3) 变成 (1,2,3)
t5 = t1.unsqueeze(2)  # 变成 (2, 3, 1)
# t6 = t1.unsqueeze(3)  # 跳出维度(越界) 会报错


# squeeze
t7 = t4.squeeze(0)  # 在0维上，减少一个维度 (2,3) 变成 (1,2,3) 变成 (2,3)
t8 = torch.randint(1, 10, (2,1,3,1,1))
t8 = t8.squeeze()  # 删掉所有维度为一的维度


# transpose
t9 = torch.randint(1, 10, (2,3,4))

t10 = t9.transpose(0, 1)  # 交换两个维度的数据
t11 = t9.permute(2, 0 ,1)  # 改变维度 (2,3,4) -> (4,3,2)


# view
t12 = torch.randint(1, 10, (2,3))
print(t12)
print(t12.is_contiguous())
t13 = t12.view(3,2)  # shape (2,3) -> (3,2)
print(t13)
print(t13.is_contiguous())  # view后仍然连续，但是transpose后不是连续。因为view是挨个放元素，但是transpose是换位置，所以不连续了
t14 = t12.transpose(0, 1).contiguous().view(3,2)  # 先用continuous方函数变成连续的



# 拼接
## 1.cat
t15 = torch.randint(1, 10, (2,3))
t16 = torch.randint(1, 10, (5,3))

t17 = torch.cat([t15,t16], dim=0)
 # (2,3) + (5,3) = (7,3)  在第0维拼接。除了拼接维度，其他需维度要保持一致
print(f't17 {t17}, shape {t17.shape}')



## 2.stack
t15 = torch.randint(1, 10, (2,3))
t16 = torch.randint(1, 10, (2,3))

t18 = torch.stack([t15,t16], dim=0)  # (2,3) + (2,3) = (2, 2, 3)
