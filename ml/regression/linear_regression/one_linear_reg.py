from sklearn.linear_model import LinearRegression

def reg():
    x_train = [[160], [166], [172], [180], [176]]
    y_train = [56.3, 60.6, 65.1, 68.5, 75]
    x_test = [[176]]

    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    print(f'weight: {estimator.coef_}')
    print(f'bias: {estimator.intercept_}')
    y_pred = estimator.predict(x_test)
    print(f'y_pred: {y_pred}')




if __name__=='__main__':
    reg()