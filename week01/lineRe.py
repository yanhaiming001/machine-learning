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
    return None


def gradientDenscent(X, y, theta, iterations, lr):
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X, 0, values=c, axis=1)  # 原始数据插入一个全为1的列
    m=X.shape[0]  # 样本的数据量
    n=X.shape[1]  # 特征的个数

    for num in range(iterations):
        for j in range(n):
            theta[j]=theta[j]+(lr/m)*(np.sum(y-np.dot(X,theta)*X[:,j].reshape(-1,1)))
    return theta



if __name__ == "__main__":
    data_path = "./data/data1.txt"
    X, y = loaddata(data_path)

    #定义对应的值
    theta=np.zeros(X.shape[1]+1).reshape(-1,1)
    iterations=400
    lr=0.01


    theta=gradientDenscent(X,y,theta,iterations,lr)

    #绘制图像
    plt.scatter(X,y)
    h_theta=theta[0]+theta[1]*X
    plt.plot(X,h_theta)

    plt.show()
