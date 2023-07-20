import numpy as np
from torch.utils.data.dataset import Dataset
from Constant import Constants as C
import torch

# 在pytorch框架下创建自定义数据集,即单个输入样例是如何定义的。

class DKTDataSet(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        onehot = self.onehot(questions, answers)
        # 以元组形式返回特征和标签
        return torch.FloatTensor(onehot.tolist())

    def onehot(self, questions, answers):
        result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS])
        for i in range(C.MAX_STEP):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                result[i][questions[i] + C.NUM_OF_QUESTIONS] = 1
        return result