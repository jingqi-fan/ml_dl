# 逻辑回归 Logistic Regression

有监督，有特征，标签(离散)，适用于 **二分类**

原理：数据，经过线性回归 -> 预测值 -> Sigmoid激活函数 -> 映射为概率 -> 基于写的阈值，决定是A还是B

## Sigmoid函数

$$f(x) = \frac{1}{1 + e^{-x}}$$

把 $(-\infty,+\infty)$ 映射到 (0,1)

## 极大似然估计
根据观测到的结果来估计模型算法中未知的参数

```angular2html
例题：有一个不均匀硬币，抛6次得到 D={正，反，反，正，正，正}。根据D估计参数theta（正面向上概率）是？
```

1. 列公式 $P(D) = \theta (1-\theta)^2 \theta^3$
2. $f(\theta)$最大时，$\theta$是多少
3. 对 $f(\theta)$ 求导找最大值

## 概念
一种分类模型，把线性回归的输出，作为逻辑回归的输入

1. 利用线性模型 $f(x) = w^Tx + b$ 根据特征和权重算出一个值
2. 利用 sigmoid 函数把它映射为概率值
3. 设置阈值，如果大于0.5，输出为1类，否则0类

### 损失函数

$$Loss(L) = -\sum_{i=1}^m [y_i\log(p_i) + (1-y_i)\log(1-p_i)] ,$$
where $p_i = sigmoid (w^Tx + b)$

这个损失函数是从极大似然估计推导来的，中间有一个取对数的，变成了找概率最大时的权重参数

逻辑回归损失函数 = - 极大似然估计函数

### API
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='12', C=1.0)

- solver：损失函数优化方法
  - liblinear对小数据场景训练更快，sag和saga对大数据更块
  - sag，saga支持L2正则或没有正则
  - liblinear和saga支持L1正则化
- penalty：正则化种类，l1或l2
- 默认将类别少的当作正例


## 评估
true/false决定真实结果，positive/negative是结果是否正确
- true positive (TP)
- true negative (TN)
- false positive (FP)
- false negative (FN)

精确率 precision = TP/ (TP+FP)
召回率 recall = TP/ (TP+FN)
F1 = [2* 精确率* 召回率] / [精确率 + 召回率] 

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [0,1,1,0,1,0]
y_pred = [0,1,0,0,1,1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

输出
```angular2html
              precision    recall  f1-score   support
0             0.67        0.67      0.67        3
1             0.67        0.67      0.67        3
```



