# 损失函数 loss function
比较网络输出和真实输出的差异

## 多分类任务 损失函数

多分类交叉熵损失 = softmax() + 损失计算

$$ L = -\sum_{i=1}^n y_i \log[ S(f_\theta (x_i)) ] $$
- y: x属于某个类别的真实概率 (one-hot)
- f(x): x属于某个类别的预测分数
- S: softmax激活函数，将属于某类的预测分数转换成概率
- L 衡量真实值y和预测值f(x)之间的差异


因为 多分类 交叉熵损失函数 的公式中包含 softmax(), 所以在 多分类+交叉熵损失函数 的情况下，输出层可以不用softmax()激活函数


## 二分类任务 损失函数
二分类用sigmoid激活函数，使用二分类的交叉熵损失函数

$$L = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

nn.BCRLoss()

## 回归任务 损失函数 MAE
mean absolute loss, i.e., L1 loss

$$ L = \frac{1}{n} \sum_{i=1}^n |y_i - f_\theta(x_i)| $$

- 用绝对误差作为距离 
- L1 loss因为具有稀疏性，可以作为正则项添加到其他loss中 
- 最大问题是梯度在零点不平滑，会跳过极小值


## 回归任务 损失函数 MSE
mean squared loss, 也叫L2 loss, 或欧式距离

$$ L = \frac{1}{n} \sum_{i=1}^n [y_i - f_\theta(x_i)]^2 $$

- 考虑平方和
- L2 loss也常作为正则项
- 当预测值和目标值相差很大时，梯度容易爆炸
- 更容易受到异常值的影响

nn.MSELoss()

## Smooth L1 loss

$$ L = 0.5x^2, if |x| < 1 $$
$$ L = |x| - 0.5, otherwise $$

- 在[-1, 1]之间是L2 loss，解决不光滑问题
- 在[-1, 1]之外是L1 loss, 解决离群点梯度爆炸的问题






