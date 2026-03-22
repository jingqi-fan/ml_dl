# 鸢尾花有三类，数据离散，属于分类

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def loadiris():
    iris_data = load_iris()
    # print(f'all data {iris_data}')
    # print(f'data type {type(iris_data)}')
    print(f'keys {iris_data.keys()}')
    print(f'data {iris_data.data[:5]}')  # 只看前5个
    print(f'target {iris_data.target[:5]}')

    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)  # 第一次需要先训练，训练的是这个Scaler，所以要先 fit_transform
    x_test = transfer.fit_transform(x_test)
    estimator = KNeighborsClassifier()

    # 交叉验证+网格搜索 找超参
    param_dict = {'n_neighbors': [i for i in range(1, 11)]}  # 超参范围
    estimator = GridSearchCV(estimator, param_dict, cv=4)  # 4是cross verification的数，返回处理后的模型对象
    estimator.fit(x_train, y_train)


    print(f'best score: {estimator.best_score_}')
    print(f'best params: {estimator.best_params_}')



    # 测试与评估
    y_pre = estimator.predict(x_test)
    y_pre_prob = estimator.predict_proba(x_test)

    # 直接评分，基于 训练集特征 和 训练集标签
    print(f'正确率(准确率): {estimator.score(x_train, y_train)}')

    # 基于 测试集的标签 和 预测结果
    print(f'正确率(准确率): {accuracy_score(y_test, y_pre)}')


if __name__ == '__main__':
    loadiris()



