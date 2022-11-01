import numpy as np

def rand(a, b):     #随机数
    return (b - a) * np.random.random() + a

def sigmod(x): #s函数
    return 1.0 / (1.0 + np.exp(-x))

def sigmod_derivative(x):   #s函数的导数
    return x * (1 - x)

class BP:
    def __init__(self, layer, iter, max_error):
        self.input_n = layer[0]     #输入层节点数 d
        self.hidden_n = layer[1]    #隐藏层节点数 q
        self.output_n = layer[2]     #输出层节点数 l
        self.gj = []    #输出层神经元梯度项
        self.eh = []    #隐层神经元梯度项
        self.input_weights = []     #输入层与隐藏层的权值矩阵
        self.output_weights = []    #隐藏层与输出层的权值矩阵
        self.iter = iter    #最大迭代次数
        self.max_error = max_error

        #初始化一个（d+1）*q的矩阵，多加的1是隐藏层的阈值
        self.input_weights = np.random.random((self.input_n+1, self.hidden_n))

        #初始化一个（q+1）*l的矩阵，多加的1是输出层的阈值
        self.output_weights = np.random.random((self.hidden_n+1, self.output_n))

        self.gj = np.zeros(layer[2])
        self.eh = np.zeros(layer[1])

    #前向传播，反向更新
    def forward_backward(self, xi, y, learning_rate=0.1): #输入，输出，学习率
        xi = np.array(xi)
        y = np.array(y)
        input_x = np.ones((1, self.input_n+1)) #记得多一个阈值,用input_x当一个中间变量,有点繁琐了
        input_x[:, :-1] = xi
        xi = input_x
        ah = np.dot(xi, self.input_weights) #计算隐藏层的输入值
        bh = sigmod(ah) #隐藏层的输出值

        #下面接着算输出层的输入和输出,和上面一样
        input_x = np.ones((1, self.hidden_n+1))
        input_x[:, :-1] = bh
        bh = input_x
        bj = np.dot(bh, self.output_weights)
        yj = sigmod(bj)

        Ek = np.sum((yj - y)**2) * 0.5  #均方误差
        self.gj = (y - yj) * sigmod_derivative(yj) #gj

        #算eh
        whj_gj = np.dot(self.output_weights, self.gj.T)  #见公式
        whj_gj = whj_gj.T
        self.eh = bh * (1 - bh) * whj_gj
        self.eh = self.eh[:, :-1]

        #更新隐藏层和输出层连接权值,最后一层是阈值，不用动
        for i in range(self.output_weights.shape[0] - 1):
            for j in range(self.output_weights.shape[1]):
                self.output_weights[i][j] += learning_rate * bh[0][i] * self.gj[0][j]

        #更新输出层阈值
        for j in range(self.output_weights.shape[1]):
            self.output_weights[-1][j] += -1.0 * learning_rate * self.gj[0][j]

        #更新输入层和隐藏层连接权值
        for i in range(self.input_weights.shape[0] - 1):
            for j in range(self.input_weights.shape[1]):
                self.input_weights[i][j] += learning_rate * xi[0][i] * self.eh[0][j]

        #更新隐藏层阈值
        for j in range(self.input_weights.shape[1]):
            self.input_weights[-1][j] += -1.0 * learning_rate * self.eh[0][j]

        return Ek

    #迭代
    def fit(self, X, Y):
        for i in range(self.iter):
            error = 0.0
            for j in range(len(X)):
                error += self.forward_backward(X[j], Y[j])
            error = error.sum()
            if abs(error) <= self.max_error:
                break

    def predict(self, x_test):
        x_test = np.array(x_test)
        tmp = np.ones((x_test.shape[0], self.input_n + 1))
        tmp[:, :-1] = x_test
        x_test = tmp
        ah = np.dot(x_test, self.input_weights)
        bh = sigmod(ah)

        tmp = np.ones((bh.shape[0], self.hidden_n + 1))
        tmp[:, :-1] = bh
        bh = tmp
        bj = np.dot(bh, self.output_weights)
        yj = sigmod(bj)
        print(yj)
        return yj

if __name__ == '__main__':
    layer = [2, 4, 1]
    X = [
            [1, 1],
            [2, 2],
            [1, 2],
            [1, -1],
            [2, 0],
            [2, -1]
        ]
    Y = [[0], [0], [0], [1], [1], [1]]

    bp = BP(layer, 10000, 0.00001)
    bp.fit(X, Y)
    bp.predict(X)
