import pandas as pd
import numpy as np

#计算信息熵
def cal_information_ent(data):
    data_label = data.iloc[:, -1]
    label_class = data_label.value_counts()
    Ent = 0
    for k in label_class.keys():
        pk = label_class[k] / len(data_label)
        Ent += -pk * np.log2(pk)
    return Ent

#计算属性a的信息增益
def cal_information_gain(data, a):
    Ent = cal_information_ent(data) #先算该点的信息熵
    feature_class = data[a].value_counts() #属性可能取值的数量
    gain = 0
    for v in feature_class.keys():
        weight = feature_class[v] / data.shape[0]
        Ent_v = cal_information_ent(data.loc[data[a] == v])
        gain += weight * Ent_v
    return Ent - gain

#选出样本数量最多的类
def get_most_label(data):
    data_label = data.iloc[:, -1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

#计算信息增益，挑选最优属性
def get_best_feature(data):
    #取出当前所有属性
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = cal_information_gain(data, a)
        res[a] = temp
    res = sorted(res.items(), key=lambda x: x[1], reverse=True) #items表示把字典转换成列表， x[1]表示按value排序，而且保留key和value
    return res[0][0]

#将数据转化为（属性值：数据表）的元组形式返回，并删去之前的属性列
def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data

#创建决策树，递归
def create_tree(data):
    data_label = data.iloc[:, -1]
    #只有一类了，叶节点
    if len(data_label.value_counts()) == 1:
        return data_label.values[0]

    #所有样本所有属性的取值相同，选样本数量最多的类为叶节点
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:, :-1].columns):
        return get_most_label(data)

    #根据信息增益得到的最优属性划分
    best_feature = get_best_feature(data)
    Tree = {best_feature: {}}  #字典形式存储
    exist_vals = pd.unique(data[best_feature]) #最优属性的取值

    #如果属性的取值减少了，即某个取值的样本集合为空，则标记为叶节点，类别为样本集中样本数最多的类
    if len(exist_vals) != len(column_count[best_feature]):
        no_exist_attr = set(column_count[best_feature]) - set(exist_vals)
        for no_feat in no_exist_attr:
            Tree[best_feature][no_feat] = get_most_label(data)

    #根据属性值创建决策子树
    for item in drop_exist_feature(data, best_feature): #item[1]其实就是data
        Tree[best_feature][item[0]] = create_tree(item[1])
    return Tree

#预测
def predict(Tree, test_data):
    #从决策树根节点开始
    first_feature = list(Tree.keys())[0] #属性
    second_dict = Tree[first_feature] #当前节点的子树
    input_first = test_data.get(first_feature) #属性值
    input_value = second_dict[input_first] #选择对应子节点的子树
    if isinstance(input_value, dict): #判断是否是字典，即是不是非叶节点
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label


if __name__ == '__main__':
    data = pd.read_csv('data_word.csv') #github有，17个样本
    #统计每个特征的取值情况作为全局变量,以字典形式存储
    column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])

    #创建决策树
    decision_tree = create_tree(data)
    print(decision_tree)
    test_data = {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '稍糊', '脐部': '凹陷', '触感': '硬滑'}
    result = predict(decision_tree, test_data)
    print('好瓜' if result == 1 else '坏瓜')