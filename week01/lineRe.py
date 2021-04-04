"""
线性回归代码实现
"""

import numpy as np
import matplotlib.pyplot as plt


def loaddata(filepath):
    """
    加载数据集
    :param filepath: 数据集的路径
    :return: 对应的X 和 y
    """
    data = np.loadtxt(filepath, delimiter=',')
    print(data.shape)
    n = data.shape[1] - 1  # 特征的数量

    X = data[:, 0:n]  # 数据
    y = data[:, -1].reshape(-1, 1)
    return X, y


def featureNormalize(X):
    """
    特征归一化
    :param X:
    :return:
    """
    mean = np.average(X, axis=0)  # 计算均值
    sigma = np.std(X, axis=0, ddof=1)  # 计算方差

    X = (X - mean) / sigma

    return X, mean, sigma


def gradientDenscent(X, y, theta, iterations, lr):
    """
    梯度下降
    :param X:
    :param y:
    :param theta:
    :param iterations:
    :param lr:
    :return:
    """
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X, 0, values=c, axis=1)  # 原始数据插入一个全为1的列
    m = X.shape[0]  # 样本的数据量
    n = X.shape[1]  # 特征的个数

    costs = np.zeros(iterations)

    for num in range(iterations):
        for j in range(n):
            theta[j] = theta[j] + (lr / m) * (np.sum(y - np.dot(X, theta) * X[:, j].reshape(-1, 1)))
        costs[num] = computerCost(X, y, theta)  # 计算损失函数
        print("第{0}次迭代 \t theta:{1}\t Costs:{2}".format(num + 1, theta, costs[num]))
    return theta, costs


def computerCost(X, y, theta):
    """
    计算损失函数
    :param X:
    :param y:
    :param theta:
    :return:
    """
    m = X.shape[0]
    return np.sum(np.power((np.dot(X, theta) - y), 2)) / (2 * m)


def predict(X):
    X = (X - mean) / sigma  # 待预测数据的特征归一化

    c = np.ones(X.shape[0]).transpose()

    X=np.insert(X,0,values=c,axis=1)
    return np.dot(X,theta)



if __name__ == "__main__":
    data_path = "./data/data1.txt"
    X, y = loaddata(data_path)

    # 定义对应的值
    theta = np.zeros(X.shape[1] + 1).reshape(-1, 1)
    iterations = 400
    lr = 0.01

    # 进行特征归一化
    X, mean, sigma = featureNormalize(X)

    theta, costs = gradientDenscent(X, y, theta, iterations, lr)

    # 绘制图像
    plt.scatter(X, y)
    h_theta = theta[0] + theta[1] * X
    plt.plot(X, h_theta)

    plt.show()

    # 绘制损失函数的图
    x_axis = np.linspace(1, iterations, iterations)
    plt.plot(x_axis, costs[0:iterations])
    plt.show()


    #预测
    print("预测结果：")
    print(predict([[20.341]]))