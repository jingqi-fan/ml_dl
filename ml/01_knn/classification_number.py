import joblib
import torch
import torchvision
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from torchvision import datasets, transforms

def minist_tr():
    transform = transforms.Compose([
        transforms.ToTensor(),   # (1, 28, 28), 值域 [0,1]
    ])

    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    x_train, y_train = to_numpy(train_ds, max_n=200)
    x_test, y_test = to_numpy(test_ds, max_n=200)

    estimator = KNeighborsClassifier()

    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'before, accuracy = {accuracy}')

    joblib.dump(estimator, 'model/model_number.pkl')  # pickle文件



def to_numpy(dataset, max_n=None):
    X, y = [], []
    for i, (img, label) in enumerate(dataset):
        if max_n and i >= max_n:
            break
        X.append(img.view(-1).numpy())  # (784,)
        y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    minist_tr()