# 欠拟合
- 原因：模型太简单
- 表现：在训练和测试上表现都不好
- 解决：增加特征，添加多项式特征

# 过拟合
- 原因：模型太复杂，原始特征过多，存在嘈杂特征
- 表现：在训练上好，在测试上表现不好
- 解决：
  - 重新清洗数据
  - 增加数据量
  - L_1, L_2正则化
  - 减少特征维度

## L_1 正则化

$$J(w) = MSE(w) + \alpha \sum_{i=1}^n |w_i|$$

$\alpha$ 惩罚系数，L1可以让权重直接变成0

使用L1正则化的线性回归模型时Lasso回归

from sklearn.linear_model import Lasso

## L_2 正则化

$$J(w) = MSE(w) + \alpha \sum_{i=1}^n {w_i}^2$$

$\alpha$ 惩罚系数

L2正则化使得权重趋向于0，一般不等于0

使用L2正则化的线性回归模型时Ridge回归

from sklearn.linear_model import Ridge


