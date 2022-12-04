import pandas as pd
import numpy as np

# 注意，未设计剪枝
# 首先CART是一棵二叉树，所以不能按照之前的决策树一样将每个特征的取值列举出来往下衍生，需要挑选一个基尼指数最小的取值作为最优切分点

# 计算基尼指数
def gini(data):
    data_label = data.iloc[:, -1]
    label_class = data_label.value_counts()
    res = 0
    for k in label_class.keys():
        pk = label_class[k] / len(data_label)
        res += pk ** 2
    return 1 - res

# 计算属性a的基尼指数, 选择最优划分点
def gini_index(data, a):
    feature_class = data[a].value_counts()
    res = []
    for feature in feature_class.keys():
        weight = feature_class[feature] / len(data)
        gini_value = gini(data.loc[data[a] == feature])
        res.append([feature, weight * gini_value])
    res = sorted(res, key=lambda x: x[-1])
    return res[0]

# 选出样本数量最多的类
def get_most_label(data):
    data_label = data.iloc[:, -1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

# 挑选最优属性，基尼指数最小的属性
def get_best_feature(data):
    #取出当前所有属性
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = gini_index(data, a) #temp是一个列表
        res[a] = temp
    res = sorted(res.items(), key=lambda x: x[1][1])  #按基尼指数排序，从小到大
    return res[0][0], res[0][1][0]

# 将数据转化为（属性值：数据表）的元组形式返回，并删去之前的属性列
def drop_exist_feature(data, best_feature, value, type):
    attr = pd.unique(data[best_feature]) # 表示属性的所有取值
    if type == 1:  #使用属性值==value的进行划分
        new_data = [[value], data.loc[data[best_feature] == value]]
    else:
        new_data = [attr, data.loc[data[best_feature] != value]]
    new_data[1] = new_data[1].drop([best_feature], axis=1)  #删除该属性
    return new_data

# 创建决策树，递归
def create_tree(data):
    data_label = data.iloc[:, -1]
    #只有一类了，叶节点
    if len(data_label.value_counts()) == 1:
        return data_label.values[0]

    #所有样本所有属性的取值相同，选样本数量最多的类为叶节点
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:, :-1].columns):
        return get_most_label(data)

    #根据信息增益得到的最优属性划分
    best_feature, best_feature_value = get_best_feature(data)
    Tree = {best_feature: {}}  #字典形式存储
    exist_vals = pd.unique(data[best_feature]) #最优属性的取值

    Tree[best_feature][best_feature_value] = create_tree(drop_exist_feature(data, best_feature, best_feature_value, 1)[1])
    Tree[best_feature]['Others'] = create_tree(drop_exist_feature(data, best_feature, best_feature_value, 2)[1])
    return Tree

# 预测
def predict(Tree, test_data):
    #从决策树根节点开始
    first_feature = list(Tree.keys())[0] #属性
    second_dict = Tree[first_feature] #当前节点的子树
    input_first = test_data.get(first_feature) #预测输入的第一个属性值
    input_value = second_dict[input_first] if input_first == first_feature else second_dict['Others'] #选择对应子节点的子树
    if isinstance(input_value, dict): #判断是否是字典，即是不是非叶节点
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label


if __name__ == '__main__':
    data = pd.read_csv('data_word.csv') #github有，17个样本
    # 统计每个特征的取值情况作为全局变量,以字典形式存储
    column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])

    # 创建决策树
    decision_tree = create_tree(data)
    print(decision_tree)
    test_data = {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '稍糊', '脐部': '凹陷', '触感': '硬滑'}
    result = predict(decision_tree, test_data)
    print('好瓜' if result == 1 else '坏瓜')
