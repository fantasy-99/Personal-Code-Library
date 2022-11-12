from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

class DE:
    def __init__(self, dim, size, iter_num, x_min, x_max, get_best_fitness, F = 0.5, CR = 0.8):
        self.F = F  #缩放因子
        self.CR = CR    #交叉概率
        self.dim = dim  #维数
        self.size = size    #种群个数
        self.iter_num = iter_num    #迭代次数
        self.cur_iter_num = 0
        self.x_min = x_min  #x下界
        self.x_max = x_max  #x上界
        self.get_best_fitness = get_best_fitness
        self.mutant = None
        #种群初始化
        self.individuality = [np.array([random.uniform(self.x_min, self.x_max) for s in range(self.dim)])
                              for tmp in range(self.size)]
        self.best_fitness = [self.get_best_fitness(v) for v in self.individuality]   #每次迭代最优适应值

    def mutate(self):
        self.mutant = []
        for i in range(self.size):
            r0, r1, r2 = 0, 0, 0
            while r0 == r1 or r1 == r2 or r0 == r2 or r0 ==i:
                r0 = random.randint(0, self.size - 1)
                r1 = random.randint(0, self.size - 1)
                r2 = random.randint(0, self.size - 1)
            #计算变异值
            tmp = self.individuality[r0] + (self.individuality[r1] - self.individuality[r2]) * self.F
            #不能超出上下界
            for t in range(self.dim):
                if tmp[t] > self.x_max or tmp[t] < self.x_min:
                    tmp[t] = random.uniform(self.x_min, self.x_max)
            self.mutant.append(tmp)

    def crossover_select(self):
        for i in range(self.size):
            #生成一个随机数，保证至少一个位置会选择变异后的值
            Jrand = random.randint(0, self.dim)
            for j in range(self.dim):
                if random.random() > self.CR and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]
            #比较当前的适应值与之前的最优适应值
            tmp = self.get_best_fitness(self.mutant[i])
            if tmp < self.best_fitness[i]:
                self.individuality[i] = self.mutant[i]
                self.best_fitness[i] = tmp

    def print_best(self):
        m = min(self.best_fitness)
        i = self.best_fitness.index(m)
        print("轮数：" + str(self.cur_iter_num))
        print("最佳个体：" + str(self.individuality[i]))
        print("目标函数值：" + str(m))

    def evolution(self):
        while self.cur_iter_num < self.iter_num:
            self.mutate()
            self.crossover_select()
            self.print_best()
            self.cur_iter_num += 1

if __name__ == "__main__":

    figure = plot.figure()
    axes = Axes3D(figure)
    X = np.arange(-513, 513, 2)
    Y = np.arange(-513, 513, 2)  # 前两个参数为自变量取值范围
    X, Y = np.meshgrid(X, Y)
    Z = -(Y + 47) * np.sin(np.sqrt(np.abs(Y + (X / 2) + 47))) - X * np.sin(
            np.sqrt(np.abs(X - Y - 47)))
    axes.plot_surface(X, Y, Z, cmap='rainbow')
    plot.show()

    def f(v):
        return -(v[1] + 47) * np.sin(np.sqrt(np.abs(v[1] + (v[0] / 2) + 47))) - v[0] * np.sin(
            np.sqrt(np.abs(v[0] - v[1] - 47)))
    p = DE(dim=2, iter_num=100, size=100, x_min=-513, x_max=513, get_best_fitness=f)
    p.evolution()

