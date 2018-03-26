from mxnet import gluon
from utils import SGD
from mxnet import autograd
from mxnet import nd


def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')


# 获取训练数据
BatchSize = 256
MnistTrain = gluon.data.vision.FashionMNIST(train=True, transform=transform)
MnistTest = gluon.data.vision.FashionMNIST(train=False, transform=transform)
TrainData = gluon.data.DataLoader(MnistTrain, BatchSize, shuffle=True)
TestData = gluon.data.DataLoader(MnistTest, BatchSize, shuffle=False)


# 初始化模型参数
NumInputs = 784
NumOutputs = 10
W = nd.random_normal(shape=(NumInputs, NumOutputs))
B = nd.random_normal(shape=NumOutputs)
Params = [W, B]
for Param in Params:
    Param.attach_grad()


def softmax(x):
    exp = nd.exp(x)
    # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
    # 就是返回 (nrows, 1) 形状的矩阵
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition


def net(x):
    return softmax(nd.dot(x.reshape((-1, NumInputs)), W) + B)


# 损失函数
def cross_entropy(y_hat, y):
    return - nd.pick(nd.log(y_hat), y)


def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iterator, net_func):
    acc = 0.
    for data, label in data_iterator:
        output = net_func(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)


LearningRate = .1

for Epoch in range(5):
    TrainLoss = 0.
    TrainAcc = 0.
    for Data, Label in TrainData:
        with autograd.record():
            Output = net(Data)
            Loss = cross_entropy(Output, Label)
        Loss.backward()
        # 将梯度做平均，这样学习率会对batch size不那么敏感
        SGD(Params, LearningRate / BatchSize)

        TrainLoss += nd.mean(Loss).asscalar()
        TrainAcc += accuracy(Output, Label)

    TestAcc = evaluate_accuracy(TestData, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        Epoch, TrainLoss / len(TrainData), TrainAcc / len(TrainData), TestAcc))
