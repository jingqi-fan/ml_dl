# 找最优超参


## 交叉验证
cv, cross-validation, 几折，就是把数据分成多少份，分别验证


折数越大，越慢


## 网格搜索

```python
from sklearn.model_selection import train_test_split, GridSearchCV

# 交叉验证+网格搜索 找超参
param_dict = {'n_neighbors': [i for i in range(1, 11)]}  # 超参范围
estimator = GridSearchCV(estimator, param_dict, cv=4)  # 4是cross verification的数，返回处理后的模型对象
estimator.fit(x_train, y_train)
```















