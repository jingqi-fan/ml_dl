from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1) 训练数据
X_train = [
    "free money offer",
    "free lottery win",
    "meeting tomorrow",
    "project discussion meeting",
    "schedule a meeting",
    "win money now"
]
y_train = ["spam", "spam", "normal", "normal", "normal", "spam"]

# 2) 模型：分词/计数 -> 朴素贝叶斯
model = Pipeline([
    ("vect", CountVectorizer()),   # 把文本转成词频向量（bag-of-words）
    ("nb", MultinomialNB())        # 多项式朴素贝叶斯
])

# 3) 训练
model.fit(X_train, y_train)

# 4) 预测
tests = [
    "free win money",
    "project meeting tomorrow",
    "lottery offer now",
    "discussion schedule"
]

pred = model.predict(tests)
proba = model.predict_proba(tests)

print("classes:", model.classes_)
for t, p, pb in zip(tests, pred, proba):
    print(f"text={t!r} -> pred={p}, proba={pb}")