import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def doit(num, w1, b1, w2, b2):  # 输入迭代次数
    number = num
    train_number = 0
    totalE = []
    totalN = []
    while (number > 0):
        number -= 1
        train_number += 1
        E = 0
        for index, i in enumerate(train_x):
            # 对x中的每一个样本进行正`向传播
            # 中间层的输入
            a1 = np.dot(i, w1)  # [1,4]
            # 隐藏层的输出
            y1 = sigmoid(a1 + b1)  # [1,4]
            # 输出层的输入
            # print(w2)
            # print(y1)
            a2 = np.dot(y1, w2)  # [1,1]
            # 输出层的输出
            y2 = sigmoid(a2 + b2)  # [1,1]
            # print(y2)
            # 反向传播
            g = np.multiply((train_y[0][index] - y2), np.multiply(y2, (1 - y2)))  # 1*19
            # print(g)
            # 更新输出层的权值w和偏置值b
            w2 = w2 + lr * np.dot(y1.T, g)
            b2 = b2 + lr * g
            # print("第"+str(index+1)+"个样本")
            # print("更新后的输出层权重w2",w2)
            # print("更新后的输出层b2",b2)
            # print("下面开始更新中间层权重w1和偏置值b1")
            w1 = w1 + lr * i.reshape(2, 1) * np.multiply(np.multiply(y1, (1 - y1)), np.dot(w2, g.T).T)
            b1 = b1 + lr * np.multiply(np.multiply(y1, (1 - y1)), np.dot(w2, g.T).T)
            # print("更新后的输出层权重w1", w1)
            # print("更新后的输出层b1", b1)
            E += (1 / 2.0 * (y2[0][0] - train_y[0][index]) ** 2)
        # print(E)
        totalE.append(E)
        totalN.append(train_number)
    print("训练完毕，输出权重和偏置值:")
    print("更新后的输出层权重w2", w2)
    print("更新后的输出层b2", b2)
    print("下面开始更新中间层权重w1和偏置值b1")
    print("更新后的输出层权重w1", w1)
    print("更新后的输出层b1", b1)

    # 画图
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax1.set_title('Result Analysis')
    # 设置横坐标名称
    ax1.set_xlabel('times')
    # 设置纵坐标名称
    ax1.set_ylabel('Error')
    # 画散点图
    ax1.scatter(totalN, totalE, s=20, c='#DC143C', marker='.')
    plt.show()

    # 预测
    output = []
    output1 = []
    for index, i in enumerate(test_x):
        # 对x中的每一个样本进行正`向传播
        # 中间层的输入
        a1 = np.dot(i, w1)  # [1,4]
        # 隐藏层的输出
        y1 = sigmoid(a1 + b1)  # [1,4]
        # 输出层的输入
        # print(w2)
        # print(y1)
        a2 = np.dot(y1, w2)  # [1,1]
        # 输出层的输出
        y2 = sigmoid(a2 + b2)  # [1,1]
        # print(y2)
        if y2[0][0] > 0.5:
            output.append(1)
        else:
            output.append(0)
    print("预测值:", output)
    trueOutPut = test_y.tolist()
    print("真实值:", trueOutPut[0])
    successNumber = 0
    for index, i in enumerate(trueOutPut[0]):
        if output[index] == trueOutPut[0][index]:
            successNumber += 1
    print("成功率:", successNumber / test_y.shape[1])


if __name__ == '__main__':
    train_data = np.loadtxt("train.txt", delimiter=',')
    test_data = np.loadtxt("test.txt", delimiter=',')
    train_x = train_data[:, :2]
    train_y = train_data[:, 2:].reshape(1, -1)
    test_x = test_data[:, :2]
    test_y = test_data[:, 2:].reshape(1, -1)
    w1 = np.random.rand(2, 4)  # 第一层四个神经元，2个特征
    b1 = np.random.rand(1, 4)

    w2 = np.random.rand(4, 1)  # 输出层有一个神经元，上一层每个神经元生成一个特征，共四个特征
    b2 = np.random.rand(1, 1)

    lr = 0.5  # 学习率
    doit(1000, w1, b1, w2, b2)