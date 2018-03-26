from mxnet import ndarray as nd
from mxnet import autograd as ag
import matplotlib.pyplot as plt
import random

# 获取训练数据
NumInputs = 2
NumExamples = 1000

TrueW = [2, -3.4]
TrueB = 4.2

X = nd.random_normal(shape=(NumExamples, NumInputs))
Y = TrueW[0] * X[:, 0] + TrueW[1] * X[:, 1] + TrueB
Y += .01 * nd.random_normal(shape=Y.shape)

# 原始数据的可视化
plt.scatter(X[:, 1].asnumpy(), Y.asnumpy())
plt.show()

# 模型参数
W = nd.random_normal(shape=(NumInputs, 1))
B = nd.zeros((1,))
Params = [W, B]
print(Params)
for Param in Params:
    Param.attach_grad()


def data_iter(batch_size):
    # 产生一个随机索引
    idx = list(range(NumExamples))
    random.shuffle(idx)
    for i in range(0, NumExamples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, NumExamples)])
        yield nd.take(X, j), nd.take(Y, j)


def net(x):
    return nd.dot(x, W) + B


def square_loss(y_hat, y):
    # 注意这里我们把y变形成yhat的形状来避免矩阵形状的自动转换
    # 行向量和列向量的问题
    return (y_hat - y.reshape(y_hat.shape)) ** 2


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


# 模型函数
def real_fn(x):
    return 2 * x[:, 0] - 3.4 * x[:, 1] + 4.2


# 绘制损失随训练次数降低的折线图，以及预测值和真实值的散点图
def plot(losses, x, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             net(x[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             real_fn(x[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()


if __name__ == '__main__':
    # 可调参数
    BatchSize = 10
    Epochs = 5  # 扫描全部数据多少遍
    LearningRate = .001

    # 用于描述损失
    NIter = 0  # 记录当前迭代到达的次数，读取一个数据算一次迭代
    Losses = []
    MovingLoss = 0
    SmoothingConstant = .01

    # 训练
    for E in range(Epochs):
        TotalLoss = 0

        for Data, Label in data_iter(BatchSize):
            with ag.record():
                Output = net(Data)
                Loss = square_loss(Output, Label)
            Loss.backward()
            SGD(Params, LearningRate)
            TotalLoss += nd.sum(Loss).asscalar()

            # 记录每读取一个数据点后，损失的移动平均值的变化；
            NIter += 1
            CurrLoss = nd.mean(Loss).asscalar()
            MovingLoss = (1 - SmoothingConstant) * MovingLoss + SmoothingConstant * CurrLoss

            # correct the bias from the moving averages
            EstLoss = MovingLoss / (1 - (1 - SmoothingConstant) ** NIter)

            if (NIter + 1) % 100 == 0:
                Losses.append(EstLoss)
                print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" %
                      (E, NIter, EstLoss, TotalLoss / NumExamples))
                # plot(Losses, X)
