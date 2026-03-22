# 线性回归
包括一元和多元线性回归

## 一元线性回归
$$y=kx+b$$

```python
from sklearn.linear_model import LinearRegression
x_train = [[160], [166], [172], [180], [176]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [[176]]

estimator = LinearRegression()
estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)
print(y_pred)
```

## 多元线性回归
$$y=w^Tx+b$$



## Loss 损失函数
预测值 - 真实值

### 均方误差 MSE
Mean-square Error, 
$$J(w,b) = \frac{1}{m} \sum_{i=0}^m [h(x^{(i)} - y^{(i)}]^2$$

每个样本点误差的平方和，除以样本总数

### 平均绝对误差 MAE
Mean Absolute Error
$$J(w,b) = \frac{1}{m} \sum_{i=0}^m |h(x^{(i)} - y^{(i)}|$$

每个样本点误差的绝对值，除以样本总数

## 优化方法

### 正规方程
就是直接算，kx+b，分别对k和b求偏导

sklearn.linear_model.LinearRegression(fit_intercept=True)

- fit_intercept=True：计算偏置 b

### 梯度下降
sklearn.linear_model.SGDRegressor(loss='squared_loss', fit_intercept=True, learning_rate='constant', eta0=0.01)

- learning_rate: 学习率策略
- eta0: 学习率

梯度
- 单变量函数中，切线斜率
- 多变量函数中，某点的偏导数

$$\theta_{i+1} = \theta_i - \alpha \frac{\partial}{\partial \theta_i} J(\theta)$$

其中 $J(\theta)$ 为损失函数

#### 全梯度下降 FGD
full gradient descent

每次迭代时，使用全部样本的梯度值

用了全部的m样本，训练慢

#### SGD
每次迭代时，随机选择并使用一个样本梯度值
sklearn.linear_model import SGDRegressor

训练快，不稳定

#### Mini-batch 小批量梯度下降

每次迭代时，随机选择并使用**小批量的样本**梯度值

每次从m个样本中，选择x个样本进行迭代 (1<x<m)

#### 随机平均梯度下降 SAG

每次迭代时，随机选择一个样本的梯度值和以往样本的梯度值的均值

初期表现不佳，因为初始梯度往往设置为0



## 回归评估方法

### 均方误差 MSE
mean squared error
$$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_predict)


### 均方根误差 RMSE
root mean squared error

$$RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 }$$

from sklearn.metrics import root_mean_squared_error

### 平均绝对误差 MAE
mean absolute error

$$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

from sklearn.metrics import mean_absolute_error









