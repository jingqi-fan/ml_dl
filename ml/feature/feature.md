# 特征工程

归一化，标准化

## 归一化
把数据映射到[min, max] (默认0到1) 之间

### 计算

$$x^\prime = \frac{x - \min}{\max - \min}$$

(当前值-该列最小值) / (该列最大值-该列最小值)

如果不是[0,1]而是[min,max]: 

$$x^{\prime\prime} = X^\prime * (\max-\min) + \min$$

容易受到极端值影响，适合小数据集

### code

1. sklearn.preprocess.MinMaxScaler()
2. fit_transform() 第一次训练+处理
3. transform  后续再处理

```python
from sklearn.preprocessing import MinMaxScaler

x_train = [[90,2,10,40], [60,4,15,45], [75,3,13,46]]
transfer = MinMaxScaler(feature_range=(0,1))  # 0到1是默认的，可不写
x_train_new = transfer.fit_transform(x_train)



```


## 标准化

考虑均值和标准差，映射到N(0,1) 正太分布

出现少量异常点，对于平均值影响不大

$$x^\prime = \frac{x-\mu}{\sigma}$$

sklearn.preprocessing.StandardScaler()

```python
from sklearn.preprocessing import StandardScaler

x_train = [[90,2,10,40], [60,4,15,45], [75,3,13,46]]
transfer = StandardScaler() 
x_train_new = transfer.fit_transform(x_train)
```












