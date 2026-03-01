from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def housing():
    data = fetch_california_housing(as_frame=True)
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression()),
    ])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f'Linear Regression MAE = {mean_absolute_error(y_test, y_pred)}')
    print(f'Linear Regression MSE = {mean_squared_error(y_test, y_pred)}')
    print(f'Linear Regression R2 = {r2_score(y_test, y_pred)}')

    print('--------------------------------')

    model_2 = Pipeline([
        ('scaler', StandardScaler()),
        ("sgd", SGDRegressor(
            loss="squared_error",  # 线性回归
            penalty=None,  # 不加正则（先对齐 LR）
            max_iter=2000,
            tol=1e-3,
            random_state=42
        ))
    ])

    model_3 = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(
            alpha=1.0,  # 正则强度（最常用的起点）
            fit_intercept=True,
            random_state=42
        ))
    ])

    model_4 = Pipeline([
        ('scaler', StandardScaler()),
        'lasso', Lasso(
            alpha=0.01,  # Lasso 通常需要更小的 alpha
            max_iter=5000,
            random_state=42
        )
    ])

    model_2.fit(x_train, y_train)
    y_pred = model_2.predict(x_test)

    print(f'SGD Regression MAE = {mean_absolute_error(y_test, y_pred)}')
    print(f'SGD Regression MSE = {mean_squared_error(y_test, y_pred)}')
    print(f'SGD Regression R2 = {r2_score(y_test, y_pred)}')



if __name__ == '__main__':
    housing()
