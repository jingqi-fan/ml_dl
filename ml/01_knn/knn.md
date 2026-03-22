# KNN 算法 / K近邻算法

K nearest neighbor，距离最近的k个样本。

注意：是算离test的样本最近的train的，找出这个train的标签作为结果

## 距离
### 欧式距离：
两点在空间中距离，对应维度差值的平方和
- 二维 $d_{12} = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$
- n维：$d_{12} = \sqrt{\sum_{i=1}^n (x_{1i} - x_{2i})^2}$
- n是特征数量

### 曼哈顿距离：

又叫城市街区距离。对应维度差值的绝对值之和

- 二维：$d_{12} = |x_1-x_2| + |y_1-y_2|$
- n维：$d_{12} = \sum_{k=1}^n |x_{1k} - x_{2k}|$


### 其他
#### 切比雪夫距离
国际象棋中国王可以直行、斜行

对应维度的差值的绝对值，求最大值

- 二维：$d_{12} = max( |x_1-x_2|,  |y_1-y_2| )$
- n维：$d_{12} = \sum_{k=1}^n |x_{1k} - x_{2k}|$




## KNN分类
有监督学习，标签不连续

### 算法流程：（投票）
1. 输入k
2. 算距离
3. 取前k个
4. 进行多数表决，哪个最多就选哪个
5. 未知的样本就是那个类别

classification

### code
sklearn.neighbors.KNeighborClassifier

导包，准备数据集，创建模型对象，训练，预测

```python
from sklearn.neighbors import KNeighborsClassifier

x_train = [[0], [1], [2], [3]]  # 训练集特征，因为可以有多个特征，所以必须是二维
y_train = [0, 0, 1, 1]  # 训练集标签
x_test = [[5]]

estimator = KNeighborsClassifier(n_neighbors=2)

estimator.fit(x_train, y_train)

y_pred = estimator.predict(x_test)

print(f'预测值：{y_pred}')

```



## KNN回归 
有监督学习，标签连续

### 算法流程 （平均值）
1. 输入k
2. 算距离
3. 取前k个
4. 计算这k个样本的标签(y)的平均值
5. 作为未知样本的预测的值

### code

sklearn.ne
```python
from sklearn.neighbors import KNeighborsRegressor

x_train = [[0,0,1], [1,1,0], [3,10,10], [4,11,12]]
y_train = [0.1, 0.2, 0.3, 0.4]
x_test = [[3,11,10]]

estimator = KNeighborsRegressor(n_neighbors=3)
estimator.fit(x_train, y_train)

y_test = estimator.predict(y_train)



```


## 拟合 fit 问题：
- k小，模型过于复杂。过拟合。
- k大，模型过于简单，欠拟合。















