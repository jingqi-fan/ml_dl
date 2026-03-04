# 1. 导入库
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 2. 准备数据
data = load_breast_cancer()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. 创建, 训练模型
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)

# 6. 预测
y_pred = model.predict(x_test)

# 7. 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



