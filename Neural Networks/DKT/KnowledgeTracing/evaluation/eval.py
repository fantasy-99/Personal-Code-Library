import sys
from sklearn import metrics
# sys.path.append('../')
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from Constant import Constants as C

def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().numpy(), prediction.detach().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())
    precision = metrics.precision_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())

    print('auc:' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + '\n')

class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
        loss = torch.Tensor([0.0])
        for student in range(pred.shape[0]):
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
            # 得到的是一个MAX_STEP-1 * MAX_STEP-1的矩阵
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]])
            # 通过索引取出数据
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]
            for i in range(len(p)):
                if p[i] > 0:
                    loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
        return loss

def train_epoch(model, trainLoader, optimizer, loss_func):
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        # print(batch.size())
        pred = model(batch)
        loss = loss_func(pred, batch)
        # 将梯度归0
        optimizer.zero_grad()
        # 反向传播计算每个参数的梯度值
        loss.backward()
        # 通过梯度下降执行一步参数更新
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader):
    gold_epoch = torch.Tensor([])
    pred_epoch = torch.Tensor([])
    for batch in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        pred = model(batch)
        # print(batch.size())
        for student in range(pred.shape[0]):
            # print(pred.shape[0])
            # print(pred.shape[1])
            # print(pred.size())
            temp_pred = torch.Tensor([])
            temp_gold = torch.Tensor([])
            # mm是矩阵相乘，t是转置, delta是为了确定题目的知识点标签
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
            # 生成长整型的张量
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]])
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]
            for i in range(len(p)):
                if p[i] > 0:
                    temp_pred = torch.cat([temp_pred, p[i:i+1]])
                    temp_gold = torch.cat([temp_gold, a[i:i+1]])
            pred_epoch = torch.cat([pred_epoch, temp_pred])
            gold_epoch = torch.cat([gold_epoch, temp_gold])
    return pred_epoch, gold_epoch


# trainLoaders实际上就是一个epoch。。len为1，这样搞是为了能在tqdm下显示进度，其实有其它方法实现。
def train(trainLoaders, model, optimizer, lossFunc):
    for i in range(len(trainLoaders)):
        # print(len(trainLoaders))
        # print(i)
        model, optimizer = train_epoch(model, trainLoaders[i], optimizer, lossFunc)
    return model, optimizer

def test(testLoaders, model):
    ground_truth = torch.Tensor([])
    prediction = torch.Tensor([])
    for i in range(len(testLoaders)):
        pred_epoch, gold_epoch = test_epoch(model, testLoaders[i])
        prediction = torch.cat([prediction, pred_epoch])
        ground_truth = torch.cat([ground_truth, gold_epoch])
    performance(ground_truth, prediction)