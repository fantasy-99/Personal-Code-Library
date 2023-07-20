import numpy as np
import itertools
import tqdm

# 该文件是处理原始数据

class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getTrainData(self):
        trainqus = np.array([])
        trainans = np.array([])
        with open(self.path, 'r') as train:
            # mininterval是更新时间，2s     *[]表示把它拆成独立的对象    zip_longest是打包，已最长的序列为标准  itertools创建高效迭代器   tqdm是一个智能进度条
            for len, ques, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 3), desc='loading train data:    ', mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                # 要满足整除
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                trainqus = np.append(trainqus, ques).astype(int)
                trainans = np.append(trainans, ans).astype(int)
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep])


    def getTestData(self):
        testqus = np.array([])
        testans = np.array([])
        with open(self.path, 'r') as test:
            for len, ques, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 3), desc='loading test data:    ', mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                # 要满足被max_step整除的条件，不够要补-1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(int)
                testans = np.append(testans, ans).astype(int)
                #返回 总数/max_step
        return testqus.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])