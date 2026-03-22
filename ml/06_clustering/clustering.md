# Clustering 聚类算法

- 是无监督学习，没有标签
- 根据样本之间的相似性，将样本划分到不同的类别中
- K-means 是一个聚类算法

## API

sklearn.cluster.KMeans(n_clusters=8)
- estimator.fit(x)
- estimator.predict(x)
- estimator.fit_predict(x)

## 实现流程

1. 确定常数 K (几个簇)，为最终类别
2. 随机选 K 个样本点为初始聚类中心
3. 计算每个样本到 K 个中心的距离，先择最近的聚类中心点作为标记类别
4. 根据每个类别中的样本点，重新算聚类中心点（平均值）
5. 如果新的中心点与原来的中心点一样，则停止，否则3

## 评估
SSE, SC, CH 三种评价指标

### SSE 误差平方和
the sum of squares due to error

$$ SSE = \sum_{i=1}^k\sum_{p\in C_i} |p-m_i|^2 $$

1. $C_i$ 表示簇
2. k 聚类中心个数
3. p 某个簇内的样本
4. m 质心点

真实值 - 质心 = 簇内距离 = 误差，+平方

#### Elbow method
肘方法确定 K

1. 算K从1到n的SSN
2. SSN逐渐变小
3. SSN中途会突然出现一个拐点
4. 这个拐点对应的K就是

### SC 轮廓系数法
silhouette coefficient

$$ S = (b-a) / max(a,b) $$
- a: 样本i到本簇内其他点的距离平均值
- b: 样本i到其他簇的点的距离平均值的最小值

### CH 轮廓系数法
Calinski-Harabasz index

$$ CH(k) = [SSB/SSW] * [(m-k)/k-1] $$

- SSW 相当于SSE
- SSB 每个簇中心点之间距离
- m 样本数量

$$SSB = \sum_{j=1}^k n_j |C_j - \bar{X}|^2$$
- $C_i$ 质心
- $\bar{X}$ 质心们的均值
- $n_j$ 样本个数