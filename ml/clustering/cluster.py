import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

# 1. 生成示例数据
x, y_true = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.6,
    random_state=42
)

plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()

# 2. 创建 K-means 模型
model = KMeans(
    n_clusters=3,
    random_state=42
)

# 3. 训练模型 + 预测
y_pred = model.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

print(calinski_harabasz_score(x, y_pred))