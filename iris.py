from sklearn.datasets import load_iris
import numpy as np
import math
from collections import Counter
import treePlotter
import matplotlib.pyplot as plt


class decisionnode:
    def __init__(self, dimension=None, value=None,results=None, mul=None,
                                lb=None, rb=None, most_label=None):
        self.dimension = dimension   # dimension表示维度
        self.value = value  # value表示二分时的比较值，将样本集分为2类
        self.results = results  # 最后的叶节点代表的类别
        self.mul = mul  # mul存储各节点的样本量与经验熵的乘积，便于剪枝时使用
        self.lb = lb  # desision node,对应于样本在d维的数据小于value时，树上相对于当前节点的子树上的节点
        self.rb = rb  # desision node,对应于样本在d维的数据大于value时，树上相对于当前节点的子树上的节点
        self.most_label = most_label

def entropy(labels):# 计算信息熵

    if labels.size > 1:

        category = list(set(labels))
    else:

        category = [labels.item()]

    entropy = 0

    for label in category:
        p = len([label_ for label_ in labels if label_ == label]) / len(labels)
        entropy += -p * math.log(p, 2)

    return entropy

def maxGainEntropy(data, labels, dimension):# 二分法计算各属性最大增益

    entropyX = entropy(labels)
    attribution = data[:, dimension]
    attribution = list(set(attribution))
    attribution = sorted(attribution)# 将值排序
    gain = 0
    value = 0
    for i in range(len(attribution) - 1):
        value_temp = (attribution[i] + attribution[i + 1]) / 2 #取每两个的中值进行计算
        small_index = [j for j in range(
            len(data[:, dimension])) if data[j, dimension] <= value_temp]
        big_index = [j for j in range(
            len(data[:, dimension])) if data[j, dimension] > value_temp]
        small = labels[small_index]
        big = labels[big_index]

        gain_temp = entropyX - (len(small) / len(labels)) * \
            entropy(small) - (len(big) / len(labels)) * entropy(big)


        if gain < gain_temp:# 计算最大信息增益与相对应的值
            gain = gain_temp
            value = value_temp

    return gain, value

def maxAttribute(data, labels):# 基于信息增益选择最优属性

    length = np.arange(len(data[0]))
    gain_max = 0
    value_max = 0
    dimension_max = 0
    for dimension in length:
        gain, value = maxGainEntropy(data, labels, dimension)
        if gain_max < gain:
            gain_max = gain
            value_max = value
            dimension_max = dimension

    return gain_max, value_max, dimension_max

def devide_group(data, labels, value, dimension):# 分为两类

    small_index = [j for j in range(
        len(data[:, dimension])) if data[j, dimension] <= value]
    big_index = [j for j in range(
        len(data[:, dimension])) if data[j, dimension] > value]

    dataSmall = data[small_index]
    dataBig = data[big_index]
    labelsSmall = labels[small_index]
    labelsBig = labels[big_index]

    return dataSmall, labelsSmall, dataBig, labelsBig

def product(labels):# 计算熵与样本数量的乘积

    entro = entropy(labels)
    #print('ent={},y_len={},all={}'.format(entro, len(labels), entro * len(labels)))
    return entro * len(labels)

def mostLabel(labels):
    label = Counter(labels)# 以字典的形式写出列表中出现的数以及每个数的数量
    label = label.most_common(1)# 出现次数最多的标签及个数
    return label[0][0]# 取标签

def buildTree(data, labels):# 递归的方式构建决策树

    if labels.size > 1:
        gain_max, value_max, dimension_max = maxAttribute(data, labels)
        if (gain_max > 0) :
            dataSmall, labelsSmall,dataBig, labelsBig = \
                devide_group(data, labels, value_max, dimension_max)
            left_branch = buildTree(dataSmall, labelsSmall)
            right_branch = buildTree(dataBig, labelsBig)
            mul=product(labels)
            most_label = mostLabel(labels)
            return decisionnode(dimension=dimension_max, value=value_max, mul=mul,
                                lb=left_branch, rb=right_branch, most_label=most_label)
        else:
            mul=product(labels)
            most_label = mostLabel(labels)
            return decisionnode(results=labels[0], mul=mul, most_label=most_label)
    else:
        mul=product(labels)
        most_label = mostLabel(labels)
        return decisionnode(results=labels.item(), mul=mul, most_label=most_label)

def printTree(tree, indent='-', dict_tree={}, direct='L'):
    # 是否是叶节点

    if tree.results != None:
        print(tree.results)

        dict_tree = {direct: str(tree.results)}

    else:
        # 打印判断条件
        print("属性" + str(tree.dimension) + ":" + str(tree.value) + "? ")
        # 打印分支
        print(indent + "L->",)

        l = printTree(tree.lb, indent=indent + "-", direct='L')
        l2 = l.copy()# 浅复制
        print(indent + "R->",)

        r = printTree(tree.rb, indent=indent + "-", direct='R')
        r2 = r.copy()
        l2.update(r2)
        stri = str(tree.dimension) + ":" + str(tree.value) + "?"
        if indent != '-':
            dict_tree = {direct: {stri: l2}}
        else:
            dict_tree = {stri: l2}

    return dict_tree

def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.dimension]
        branch = None

        if v > tree.value:
            branch = tree.rb
        else:
            branch = tree.lb

        return classify(observation, branch)

def pruning(tree, alpha=0.1):# 剪枝操作
    if tree.lb.results == None:
        pruning(tree.lb, alpha)
    if tree.rb.results == None:
        pruning(tree.rb, alpha)

    if tree.lb.results != None and tree.rb.results != None:
        before_pruning = tree.lb.mul + tree.rb.mul + 2 * alpha
        after_pruning = tree.mul + alpha
        print('before_pruning={},after_pruning={}'.format(
            before_pruning, after_pruning))
        if after_pruning <= before_pruning:
            print('pruning--{}:{}?'.format(tree.dimension, tree.value))
            tree.lb, tree.rb = None, None
            tree.results = tree.most_label

if __name__ == '__main__':
    iris = load_iris()
    data = iris.data
    labels = iris.target
    print(maxGainEntropy(data,labels,3))
    array = np.random.permutation(data.shape[0]) #生成一个总数等于data列个数的随机排列的列表
    shuffled_data = data[array,:]
    shuffled_labels = labels[array]# 将数据完全打乱重新组合

    train_data = shuffled_data[:100, :]
    train_labels = shuffled_labels[:100]# 随机选100组为训练集

    test_data = shuffled_data[100:150, :]
    test_labels = shuffled_labels[100:150]# 其余50组为测试集

    tree = buildTree(train_data,train_labels)
    printedTree = printTree(tree=tree)

    true_num = 0
    for i in range(len(test_labels)):
        prediction = classify(test_data[i],tree)
        if prediction == test_labels[i]:
            true_num += 1
    print("ID3Tree:{}".format(true_num))

    treePlotter.createPlot(printedTree, 1)
    plt.show()