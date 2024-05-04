from queue import PriorityQueue
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.model_selection import train_test_split
from sklearn import decomposition, model_selection
import torch
import sys
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch.nn as nn
from sklearn.decomposition import KernelPCA
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy.core.umath_tests import inner1d
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.ensemble import GradientBoostingClassifier
import smote_variants
from pylmnn import LargeMarginNearestNeighbor
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler

from src.BaseSVDD import BaseSVDD
import seaborn as sns
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import tensorflow as tf
import math
from sklearn.cluster import KMeans
from sklearn.utils import stats
from scipy import stats
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=1e6)  # threshold表示输出数组的元素数目
np.set_printoptions(suppress=True) #Numpy取消numpy科学计数法
np.set_printoptions(threshold=np.inf)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time
from annoy import AnnoyIndex

class AdaCostClassifier(AdaBoostClassifier):

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        incorrect = y_predict != y

        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        estimator_weight = (-1. * self.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

        # 样本更新的公式，只需要改写这里
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)) *
                                    self._beta(y, y_predict))  # 在原来的基础上乘以self._beta(y, y_predict)，即代价调整函数
        return sample_weight, 1., estimator_error


    #  新定义的代价调整函数
    def _beta(self, y, y_hat):
        res = []

        for i in zip(y, y_hat):
            if i[0] == i[1]:
                res.append(1)   # 正确分类，系数保持不变，按原来的比例减少
            elif i[0] == 1 and i[1] == -1:
                res.append(8.5)  # 在信用卡的情景下，将好人误杀代价应该更大一些，比原来的增加比例要高
            elif i[0] == -1 and i[1] == 1:
                res.append(1)  # 将负例误判为正例，代价不变，按原来的比例增加

        return np.array(res)

#-------------------------annoy------------------------------------------------------------------

p_means=[]
q_means=[]

def means(X):
    """
    启发式的选取两个点

    参数
    ----------
    X : 特征矩阵

    返回
    ----------
    两个向量点
    """
    iteration_steps = 20
    count = X.shape[0]
    Kmeans2 = KMeans(n_clusters=2)
    kmeans_fit2 = Kmeans2.fit(X)  # 模型训练
    centers_p = kmeans_fit2.cluster_centers_  # 聚类中心

    i, j = get_density_center_index(X)
    p = centers_p[0]
    q = centers_p[1]

    # i = np.random.randint(0, count)
    # j = np.random.randint(0, count - 1)

    # 保证 i\j 不相同
    # j += (j >= i)
    # ic = 1
    # jc = 1
    # p = X[i]
    # q = X[j]
    #
    # for l in range(iteration_steps):
    #     k = np.random.randint(0, count)
    #     di = ic * distance(p, X[k])
    #     dj = jc * distance(q, X[k])
    #     if di == dj:
    #         continue
    #     if di < dj:
    #         p = (p * ic + X[k]) / (ic + 1)
    #         ic = ic + 1
    #     else:
    #         q = (q * jc + X[k]) / (jc + 1)
    #         jc = jc + 1
    p_means.append(p)
    q_means.append(q)

    return p, q

def distance(a, b):
    """
    计算距离

    参数
    ----------
    a : 向量 a

    b : 向量 b

    返回
    ----------
    向量 a 与 向量 b 直接的距离
    """
    return np.linalg.norm(a - b)

class annoynode:
    """
    Annoy 树结点
    """

    def __init__(self, index, size, w, b, left=None, right=None):
        # 结点包含的样本点下标
        self.index = index
        # 结点及其子结点包含的样本数
        self.size = size
        # 分割超平面的系数
        self.w = w
        # 分割超平面的偏移量
        self.b = b
        # 左子树
        self.left = left
        # 右子树
        self.right = right

    def __lt__(self, other):
        # 结点大小比较
        return self.size < other.size

class annoytree:
    """
    Annoy 树算法实现

    参数
    ----------
    X : 特征矩阵

    leaf_size : 叶子节点包含的最大特征向量数量，默认为 10
    """

    def __init__(self, X, leaf_size=10):
        def build_node(X_indexes):
            """
            构建结点

            参数
            ----------
            X_indexes : 特征矩阵下标
            """
            # 当特征矩阵小于等于指定的叶子结点的大小时，创建叶子结点并返回
            if len(X_indexes) <= leaf_size :
            # if len(X_indexes) <= leaf_size and len(X_indexes)>=3:
                return annoynode(X_indexes, len(X_indexes), None, None)
            # 当前特征矩阵
            _X = X[X_indexes, :]
            # 启发式的选取两点
            p, q = means(_X)
            # 超平面的系数
            w = p - q
            # 超平面的偏移量
            b = -np.dot((p + q) / 2, w)
            # 构建结点
            node = annoynode(None, len(X_indexes), w, b)
            # 在超平面“左”侧的特征矩阵下标
            left_index = (_X.dot(w) + b) > 0
            if left_index.any():
                # 递归的构建左子树
                node.left = build_node(X_indexes[left_index])
            # 在超平面“右”侧的特征矩阵下标
            right_index = ~left_index
            if right_index.any():
                # 递归的构建右子树
                node.right = build_node(X_indexes[right_index])
            return node

        # 根结点
        self.root = build_node(np.array(range(X.shape[0])))

class annoytrees:
    """
    Annoy 算法实现

    参数
    ----------
    X : 特征矩阵

    n_trees : Annoy 树的数量，默认为 10

    leaf_size : 叶子节点包含的最大特征向量数量，默认为 10
    """

    def __init__(self, X, n_trees=10, leaf_size=10):
        self._X = X
        self._trees = []
        # 循环的创建 Annoy 树
        for i in range(n_trees):
            self._trees.append(annoytree(X, leaf_size=leaf_size))

    def query(self, x, k=1, search_k=-1):
        """
        查询距离最近 k 个特征向量

        参数
        ----------
        x : 目标向量

        k : 查询邻居数量

        search_k : 最少遍历出的邻居数量，默认为 Annoy 树的数量 * 查询数量
        """

        # 创建结点优先级队列
        nodes = PriorityQueue()
        # 先将所有根结点加入到队列中
        for tree in self._trees:
            nodes.put([float("inf"), tree.root])
        if search_k == -1:
            search_k = len(self._trees) * k
        # 待查询的邻居下标数组
        nns = []
        # 循环优先级队列
        while len(nns) < search_k and not nodes.empty():
            # 获取优先级最高的结点
            (dist, node) = nodes.get()
            # 如果是叶子结点，将下标数组加入待查询的邻居中
            if node.left is None and node.right is None:
                nns.extend(node.index)
            else:
                # 计算目标向量到结点超平面的距离
                dist = min(dist, np.abs(x.dot(node.w) + node.b))
                # 将距离做为优先级的结点加入到优先级队列中
                if node.left is not None:
                    nodes.put([dist, node.left])
                if node.right is not None:
                    nodes.put([dist, node.right])
        # 对下标数组进行排序
        nns.sort()
        prev = -1
        # 优先级队列
        nns_distance = PriorityQueue()
        for idx in nns:
            # 过滤重复的特征矩阵下标
            if idx == prev:
                continue
            prev = idx
            # 计算特征向量与目标向量的距离做为优先级
            nns_distance.put([distance(x, self._X[idx]), idx])
        nearests = []
        distances = []
        # 取前 k 个
        for i in range(k):
            if nns_distance.empty():
                break
            (dist, idx) = nns_distance.get()
            nearests.append(idx)
            distances.append(dist)
        return nearests, distances
#-------------------------annoy------------------------------------------------------------------

#-------------------------play the tree----------------------------------------------------------

def layer_order(root):
    """使用FIFO队列实现层序遍历
       逐层从左到右插入队列，再从头部弹出
    """
    node_queue = list()
    value_list = list()
    node_queue.append(root)
    while node_queue:
        tree_node = node_queue.pop(0)
        value_list.append(tree_node.index)
        if tree_node.left:
            node_queue.append(tree_node.left)
        if tree_node.right:
            node_queue.append(tree_node.right)
    return value_list

def inorder(root):
    """递归实现中序遍历（中根顺序）"""
    if root == None:
        return
    inorder(root.left)
    if root.left==None and root.right==None:
        leaf.append(root.index)
    # print(root.index, end=" \n")
    inorder(root.right)

def get_num_leaf(root):
    """获取二叉树叶子节点"""
    if root==None:
        return 0 #当二叉树为空时直接返回0
    elif root.left==None and root.right==None:
        return 1 #当二叉树只有一个根，但是无左右孩子时，根节点就是一个叶子节点
    else:
        return (get_num_leaf(root.left)+get_num_leaf(root.right))  #其他情况就需要根据递归来实现

#-------------------------play the tree----------------------------------------------------------

#-------------------------play the Matrix--------------------------------------------------------

# 欧氏距离计算
def distEclud2(x,y,m):
    # return np.sqrt(np.sum((x-y)**2))  # 计算欧氏距离
    return m*m*np.sqrt(mean_squared_error(x,y))

def embedding_distance(feature_1, feature_2):
    # 计算欧式距离
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

def Matrix_division(dataArr):
    '''
    将拼接了样本权重训练集分为多数类矩阵和少数类矩阵
    Args:
        dataArr: 带标签的数据集

    Returns:
        dataArr_N：多数类矩阵
        dataArr_P：少数类矩阵

    '''

    dataArr = np.matrix(dataArr)
    p, q = np.shape(dataArr)
    # print(p, q)

    a = []  # 正类的所在行数记录
    b = []  # 负类的所在行数记录
    for i in range(p):

        c = dataArr[i, q - 1]
        if c == 1:
            a.append(i)
        else:
            b.append(i)

    # 多数类矩阵
    dataArr_N = np.delete(dataArr, a, 0)
    # 少数类矩阵
    dataArr_P = np.delete(dataArr, b, 0)

    return dataArr_N, dataArr_P

def Partitioned_subspace_matrix(at1,leaf,data):
    '''
    划分子空间矩阵
    '''
    prepare_list = locals()
    # 将原样本矩阵划分为多个子空间样本矩阵：
    for i in range(get_num_leaf(at1.root)):

        prepare_list['subspace_matrix' + str(i)] = []
        for j in range(len(leaf[i])):
            prepare_list['subspace_matrix' + str(i)].append(data[leaf[i][j],:].tolist()[0])

        np.mat(prepare_list['subspace_matrix' + str(i)])

        # print(np.shape(prepare_list['subspace_matrix' + str(i)]))
        # print(prepare_list['subspace_matrix' + str(i)])
    #将数组转换成矩阵
    for i in range(len(leaf)):
        prepare_list['subspace_matrix' + str(i)]=np.mat(prepare_list['subspace_matrix' + str(i)])

    return prepare_list

#-------------------------WGAN-------------------------------------------------------------------

def wgan(dataset, epochs,path):
    '''
        训练WGAN
    '''
    # device = 'cuda:1'
    # G = torch.load ('G4.pt', map_location=torch.device (device))
    # G.eval ()
    # random_noise = torch.randn (g_count, 100, device=device)
    # fake = G (random_noise)
    # fake = fake.detach().cpu().numpy()
    #
    # lable_0 = []
    # for i in range (len (fake)):
    #     lb_1 = 1
    #     lable_0.append (lb_1)
    # lable_0p = pd.DataFrame (lable_0)
    # lable_0p = lable_0p.T
    # print(fake.shape)
    # return fake,lable_0p

    dataset = torch.Tensor(np.array(dataset))
    print(dataset)  # 测试
    print(dataset.shape)  # 测试
    output_size = dataset.shape[1]
    Generator = nn.Sequential(
        nn.Linear(100, 256),  # 输入100个数据，输出256个数据
        nn.ReLU(),
        # nn.Linear(256, 512),  # 输入256个数据，输出512个数据
        # nn.ReLU(),
        # nn.Linear(512, output_size),  # 输入512个数据，输出output_size长度的张量
        nn.Linear (256, 512),
        nn.ReLU (),
        nn.Linear (512, output_size)
    )

    Discriminator = nn.Sequential(
        nn.Linear(output_size, 256),  # 输入一个长output_size的张量，输出512个数据
        nn.LeakyReLU(),
        nn.Linear(256, 512),  # 输入512个数据，输出256个数据
        nn.LeakyReLU(),
        # nn.Linear(256, 1),  # 输入256个数据，输出1个数据，输出一个数据是典型的判别模型
        # nn.Linear (output_size, 512),
        # nn.ReLU (),
        nn.Linear (512, 1)
    )

    '''初始化'''
    device = torch.device ("cuda:0")
    G = Generator.to(device)
    D = Discriminator.to(device)
    G_optim = torch.optim.RMSprop(G.parameters(), lr=0.00005)
    D_optim = torch.optim.RMSprop(D.parameters(), lr=0.00005)
    # D_optim = torch.optim.Adam(D.parameters(), lr=0.00005)
    # G_optim = torch.optim.Adam(G.parameters(), lr=0.00005)

    '''训练'''
    D_loss_list = []  # 用来存放每个迭代周期的损失
    G_loss_list = []
    # 用于储存生成器和辨别器得分
    for epoch in range(epochs):
        D_epoch_loss = 0
        G_epoch_loss = 0
        count = dataset.shape[0]

        for step, sample_batch in enumerate(dataset):
            sample_batch = sample_batch.to(device)
            size = sample_batch.size(0)
            random_noise = torch.randn(size, 100, device=device)

            '''wgan把D的参数截断在一个固定常数c的范围内'''
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            '''训练判别器'''
            D_loss = 0
            for _ in range(15):
                D_optim.zero_grad()
                D_real = D(sample_batch).mean()
                G_sample = G(random_noise)
                D_fake = D(G_sample.detach()).mean()
                '''损失函数'''
                D_loss = D_fake + (- D_real)
                D_loss.backward()
                D_optim.step()

            '''训练生成器'''
            G_optim.zero_grad()
            G_sample = G(random_noise)
            D_fake = D(G_sample).mean()
            '''损失函数'''
            G_loss = -D_fake
            G_loss.backward()
            G_optim.step()

            '''统计每个迭代周期的总损失'''
            with torch.no_grad():
                D_epoch_loss += D_loss.item()
                G_epoch_loss += G_loss.item()

        '''计算每个迭代周期的平均损失'''
        with torch.no_grad():
            D_epoch_loss /= count
            G_epoch_loss /= count
            D_loss_list.append(D_epoch_loss)
            G_loss_list.append(G_epoch_loss)
            if epoch % 50 == 0:
                print('Epoch:', epoch)
    # 把训练好的网络写入模型
    G.eval()
    # '''生成20个假数据'''
    # random_noise = torch.randn(g_count, 100, device=device)
    # fake = G(random_noise)
    # fake = fake.detach().cpu().numpy()
    #
    # lable_0 = []
    # for i in range (len (fake)):
    #     lb_1 = 1
    #     lable_0.append (lb_1)
    # lable_0p = pd.DataFrame (lable_0)
    # lable_0p = lable_0p.T
    # print(fake.shape)

    print(type(D_loss_list))
    print(D_loss_list)
    '''对平均损失进行绘图'''

    plt.plot((D_loss_list), label='D_loss')
    plt.plot ((G_loss_list), label='G_loss')
    plt.legend ()
    plt.show ()
    torch.save (G, path)
    # return fake,lable_0p

def wgan_Generatorfake(path,g_count):
    '''

    '''
    device = 'cuda:0'
    G = torch.load (path, map_location=torch.device (device))
    G.eval ()
    random_noise = torch.randn (g_count, 100, device=device)
    fake = G(random_noise)
    fake = fake.detach().cpu().numpy()

    lable_0 = []
    for i in range (len (fake)):
        lb_1 = 1
        lable_0.append (lb_1)
    lable_0p = pd.DataFrame (lable_0)
    lable_0p = lable_0p.T
    print(fake.shape)
    return fake,lable_0p

def wgan_GeneratorNfake(path,g_count):
    '''

    '''
    device = 'cuda:0'
    G = torch.load (path, map_location=torch.device (device))
    G.eval ()
    random_noise = torch.randn (g_count, 100, device=device)
    fake = G(random_noise)
    fake = fake.detach().cpu().numpy()

    lable_0 = []
    for i in range (len (fake)):
        lb_1 = -1
        lable_0.append (lb_1)
    lable_0p = pd.DataFrame (lable_0)
    lable_0p = lable_0p.T
    print(fake.shape)
    return fake,lable_0p

def showDataSet3(i,fake,data_p,data_n):
    """
       数据可视化
       Parameters:
           dataMat - 数据矩阵
           labelMat - 数据标签
       Returns:
           无
       """

    # for i in range (len (dataMat)):
    #     if labelMat[0][i] > 0:
    #         data_plus.append (dataMat[i])
    #     else:
    #         data_minus.append (dataMat[i])
    data_plus_np = np.array (fake)  # 转换为numpy矩阵
    data_minus_np = np.array (data_p)  # 转换为numpy矩阵
    data_maxus_np = np.array (data_n)

    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("The " + str(i) + "th" + " SubSpace:")
    plt.grid() # 添加网络
    plt.scatter (np.transpose (data_plus_np)[0], np.transpose (data_plus_np)[1],c='r',marker='*',)  # fake散点图
    plt.scatter (np.transpose (data_minus_np)[0], np.transpose (data_minus_np)[1],c='y',marker='o')  # 正样本散点图
    plt.scatter (np.transpose (data_maxus_np)[0], np.transpose (data_maxus_np)[1],c='b',marker='x')  # 负样本散点图
    plt.legend(['fakedata','data_p','data_N'],loc='upper right')
    plt.show ()

def showDataSet3_no_N(i,fake,data_p):
    """
       数据可视化
       Parameters:
           dataMat - 数据矩阵
           labelMat - 数据标签
       Returns:
           无
       """

    # for i in range (len (dataMat)):
    #     if labelMat[0][i] > 0:
    #         data_plus.append (dataMat[i])
    #     else:
    #         data_minus.append (dataMat[i])
    data_plus_np = np.array (fake)  # 转换为numpy矩阵
    data_minus_np = np.array (data_p)  # 转换为numpy矩阵
    # data_maxus_np = np.array (data_n)

    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("The " + str(i) + "th" + " SubSpace:")
    plt.grid() # 添加网络
    plt.scatter (np.transpose (data_plus_np)[0], np.transpose (data_plus_np)[1],c='r',marker='*',)  # fake散点图
    plt.scatter (np.transpose (data_minus_np)[0], np.transpose (data_minus_np)[1],c='y',marker='o')  # 正样本散点图
    # plt.scatter (np.transpose (data_maxus_np)[0], np.transpose (data_maxus_np)[1],c='b',marker='x')  # 负样本散点图
    plt.legend(['fakedata','data_p','data_N'],loc='upper right')
    plt.show ()

def showDataSet3_no_P(i,fake,data_n):
    """
       数据可视化
       Parameters:
           dataMat - 数据矩阵
           labelMat - 数据标签
       Returns:
           无
       """

    # for i in range (len (dataMat)):
    #     if labelMat[0][i] > 0:
    #         data_plus.append (dataMat[i])
    #     else:
    #         data_minus.append (dataMat[i])
    data_plus_np = np.array (fake)  # 转换为numpy矩阵
    # data_minus_np = np.array (data_p)  # 转换为numpy矩阵
    data_maxus_np = np.array (data_n)

    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("The " + str(i) + "th" + " SubSpace:")
    plt.grid() # 添加网络
    plt.scatter (np.transpose (data_plus_np)[0], np.transpose (data_plus_np)[1],c='r',marker='*',)  # fake散点图
    # plt.scatter (np.transpose (data_minus_np)[0], np.transpose (data_minus_np)[1],c='y',marker='o')  # 正样本散点图
    plt.scatter (np.transpose (data_maxus_np)[0], np.transpose (data_maxus_np)[1],c='b',marker='x')  # 负样本散点图
    plt.legend(['fakedata','data_p','data_N'],loc='upper right')
    plt.show ()

def showDataSet(featureMat, labelMat):
    #创建标签为1的样本列表
    data_one = []
    #创建标签为0的样本列表
    data_zero = []
    #遍历特征矩阵featureMat，i是特征矩阵featureMat的当前行
    #特征矩阵featureMat的两个特征列，正好是散点图的数据点的x轴坐标和y轴坐标
    for i in range(len(featureMat)):
        #如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为1
        if labelMat[i] == 1:
            #将当前特征矩阵featureMat[i]行添入data_one列表
            data_one.append(featureMat[i])
        #如果特征矩阵featureMat的当前行号i对应的标签列表labelMat[i]的值为0
        elif labelMat[i] == -1:
            #将当前特征矩阵featureMat[i]行添入data_zero列表
            data_zero.append(featureMat[i])
    #将做好的data_one列表转换为numpy数组data_one_np
    data_one_np = np.array(data_one)
    #将做好的data_zero列表转换为numpy数组data_zero_np
    data_zero_np = np.array(data_zero)
    #根据标签为1的样本的x坐标（即data_one_np的第0列）和y坐标（即data_one_np的第1列）来绘制散点图
    plt.scatter(data_one_np[:,0], data_one_np[:,1])
    #根据标签为0的样本的x坐标（即data_zero_np的第0列）和y坐标（即data_zero_np的第1列）来绘制散点图
    plt.scatter(data_zero_np[:,0], data_zero_np[:,1])
    #显示画好的散点图
    plt.show()
#-------------------------data-------------------------------------------------------------------

def load_data(path,dim,testsize):

    # Step 1. 导入训练数据
    # 划分少数类和多数类
    data_0 = pd.read_excel(path, sheet_name='major',
                           header=None)  # Read most types of data
    data_1 = pd.read_excel(path, sheet_name='minor',
                           header=None)  # Read minority data
    print(np.shape(data_0))  # 187,66
    print(np.shape(data_1))  # 15,66
    # X_0为多数类，X_1为少数类
    X_0, Y_0 = data_0.iloc[:, 1:].values, data_0.iloc[:, 0].values
    X_1, Y_1 = data_1.iloc[:, 1:].values, data_1.iloc[:, 0].values
    X_0 = pd.DataFrame(X_0)
    X_1 = pd.DataFrame(X_1)
    Y_0 = pd.DataFrame(Y_0)
    Y_1 = pd.DataFrame(Y_1)

    dataset = np.vstack((X_0, X_1))


    kpca = KernelPCA(n_components=dim, kernel="rbf", gamma=15, fit_inverse_transform=True, remove_zero_eig=True)
    dataset = kpca.fit_transform(dataset)

    Y = np.vstack((Y_0, Y_1))

    # 把负类的标签全部用-1表示
    for i in range(np.shape(Y)[0]):
        if Y[i] == 0:
            Y[i] = -1

    dataArr, testArr, LabelArr, testLabelArr = train_test_split(dataset, Y, test_size=testsize, random_state=1, stratify=Y)



    return dataArr,testArr,LabelArr,testLabelArr

def showDataSet2(data_p,data_n):
    """
       数据可视化
       Parameters:
           dataMat - 数据矩阵
           labelMat - 数据标签
       Returns:
           无
       """

    # for i in range (len (dataMat)):
    #     if labelMat[0][i] > 0:
    #         data_plus.append (dataMat[i])
    #     else:
    #         data_minus.append (dataMat[i])



    data_minus_np = np.array (data_p)  # 转换为numpy矩阵
    data_maxus_np = np.array (data_n)

    plt.figure(figsize=(10, 6), dpi=80)

    plt.grid() # 添加网络

    plt.scatter (np.transpose (data_minus_np)[0], np.transpose (data_minus_np)[1],c='y',marker='o')  # 正样本散点图
    plt.scatter (np.transpose (data_maxus_np)[0], np.transpose (data_maxus_np)[1],c='b',marker='x')  # 负样本散点图
    plt.legend(['data_p','data_N'],loc='upper right')

    plt.show ()

def showDataSet1(i,data_p,data_n):
    """
       数据可视化
       Parameters:
           dataMat - 数据矩阵
           labelMat - 数据标签
       Returns:
           无
       """

    # for i in range (len (dataMat)):
    #     if labelMat[0][i] > 0:
    #         data_plus.append (dataMat[i])
    #     else:
    #         data_minus.append (dataMat[i])

    data_minus_np = np.array (data_p)  # 转换为numpy矩阵
    data_maxus_np = np.array (data_n)
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("The " + str(i) + "th" + " SubSpace:")
    plt.grid()  # 添加网络

    if np.shape(data_p)[1] == 0:
        plt.scatter(np.transpose(data_maxus_np)[0], np.transpose(data_maxus_np)[1], c='b', marker='x')  # 负样本散点图

    elif np.shape(data_n)[1] == 0:
        plt.scatter (np.transpose (data_minus_np)[0], np.transpose (data_minus_np)[1],c='y',marker='o')  # 正样本散点图

    else :
        plt.scatter (np.transpose (data_minus_np)[0], np.transpose (data_minus_np)[1],c='y',marker='o')  # 正样本散点图
        plt.scatter (np.transpose (data_maxus_np)[0], np.transpose (data_maxus_np)[1],c='b',marker='x')  # 负样本散点图


    plt.legend(['data_p','data_N'],loc='upper right')
    plt.show ()



def makeNewTrdata2(dataArr_N,dataArr_P):
    lable_0 = []
    for i in range(len(dataArr_N)):
        lb_0 = -1
        lable_0.append(lb_0)
    lable_0 = pd.DataFrame(lable_0)


    lable_P1 = []
    for i in range(len(dataArr_P)):
        lb_p1 = 1
        lable_P1.append(lb_p1)
    lable_P1 = pd.DataFrame(lable_P1)

    # fake_data = np.vstack((fake_data, dataArr_P))
    # lable_1 = np.vstack((lable_1, lable_P1))
    # print(np.shape(fake_data))
    # print(np.shape(dataArr_N))
    train_x = np.vstack((dataArr_P, dataArr_N))
    train_y = np.vstack((lable_P1, lable_0))
    return train_x,train_y

def Datafiltering(dataArr_N,dataArr_P):

    # # 对扩增后样本进行筛选
    # Kmeans = KMeans(n_clusters=1)
    # kmeans_fit = Kmeans.fit(dataArr_N)  # 模型训练
    # centers_n = kmeans_fit.cluster_centers_  # 负样本的中心点
    # # 找出少数正类的质心：
    # Kmeans3 = KMeans(n_clusters=1)
    # kmeans_fit3 = Kmeans3.fit(dataArr_P)  # 模型训练
    # centers_p = kmeans_fit3.cluster_centers_  # 正样本的中心点

    centers_n = get_density_center(dataArr_N)

    centers_p = get_density_center(dataArr_P)



    dis = []

    for i in range(np.shape(dataArr_P)[0]):
        sum = 0
        # print("fake_data[i,:]",np.shape(fake_data[i,:]))
        # print("centers_n", np.shape(centers_n))
        # distance_n = distEclud2(fake_data[i, :], centers_n, np.shape(centers_n)[0])
        distance_n = embedding_distance(dataArr_P[i, :], centers_n)
        distance_p = embedding_distance(dataArr_P[i, :], centers_p)

        # distance_p = distEclud2(fake_data[i, :], centers_p, np.shape(centers_p)[0])
        # distance_f = distEclud2(fake_data[i, :], centers_f, np.shape(centers_f)[0])
        # 计算扩增的正样本到原正样本质心的距离
        # for j in range(np.shape(dataArr_N)[0]):
        #     b = dataArr_N[j, :]
        #     distance_SUM = distEclud2(fake_data[i, :], b, np.shape(dataArr_N)[0])
        #     sum = sum + distance_SUM
        # distance = sum
        # distance = (distance_n *(1/distance_p))
        distance = distance_n
        dis.append(distance)
    dis = np.mat(dis)
    dis = dis.T

    # 让扩增的正样本与其到负类质心的距离相拼接
    fake_data_dis = np.concatenate((dataArr_P, dis), axis=1)
    # 按照正类样本到质心的距离由大到小的排序
    a = np.lexsort(-fake_data_dis.T)
    # print(dis[np.lexsort(-dis.T)])
    fake_data_dis = fake_data_dis[a[0], :]
    # 剔除后一半的正类样本
    fake_data_dis = fake_data_dis[:np.shape(dataArr_N)[0], :]
    # 剔除距离列
    fake_data = np.delete(fake_data_dis, np.shape(fake_data_dis)[1] - 1, 1)
    # print(np.shape(fake_data))
    return fake_data

def Datafiltering2(fake_data,dataArr_N,dataArr_P):
    # 对扩增后样本进行筛选
    # Kmeans = KMeans(n_clusters=1)
    # kmeans_fit = Kmeans.fit(dataArr_N)  # 模型训练
    # centers_n = Kmeans.cluster_centers_  # 负样本的中心点
    # # 找出少数正类的质心：
    # Kmeans3 = KMeans(n_clusters=1)
    # kmeans_fit3 = Kmeans3.fit(dataArr_P)  # 模型训练
    # centers_p = kmeans_fit3.cluster_centers_  # 正样本的中心点
    # Kmeans2 = KMeans(n_clusters=1)
    # kmeans_fit2 = Kmeans2.fit(fake_data)  # 模型训练
    # centers_f = kmeans_fit2.cluster_centers_  # 扩增样本的中心点

    centers_n = get_density_center(dataArr_N)

    centers_p = get_density_center(dataArr_P)

    centers_f = get_density_center(fake_data)

    dis = []
    fake_data = np.mat(fake_data)
    for i in range(np.shape(fake_data)[0]):
        sum = 0
        # print("fake_data[i,:]",np.shape(fake_data[i,:]))
        # print("centers_n", np.shape(centers_n))
        # distance_n = distEclud2(fake_data[i, :], centers_n, np.shape(centers_n)[0])
        distance_n = embedding_distance(fake_data[i, :], centers_n)
        distance_p = embedding_distance(fake_data[i, :], centers_p)
        distance_f = embedding_distance(fake_data[i, :], centers_f)
        # distance_p = distEclud2(fake_data[i, :], centers_p, np.shape(centers_p)[0])
        # distance_f = distEclud2(fake_data[i, :], centers_f, np.shape(centers_f)[0])
        # 计算扩增的正样本到原正样本质心的距离
        # for j in range(np.shape(dataArr_N)[0]):
        #     b = dataArr_N[j, :]
        #     distance_SUM = distEclud2(fake_data[i, :], b, np.shape(dataArr_N)[0])
        #     sum = sum + distance_SUM
        # distance = sum
        # distance = (distance_n +distance_f)*distance_p
        distance = distance_n
        dis.append(distance)
    dis = np.mat(dis)
    dis = dis.T

    # 让扩增的正样本与其到负类质心的距离相拼接
    fake_data_dis = np.concatenate((fake_data, dis), axis=1)
    # 按照正类样本到质心的距离由大到小的排序
    a = np.lexsort(-fake_data_dis.T)
    # print(dis[np.lexsort(-dis.T)])
    fake_data_dis = fake_data_dis[a[0], :]
    # 剔除后一半的正类样本
    fake_data_dis = fake_data_dis[:np.shape(dataArr_N)[0]-np.shape(dataArr_P)[0], :]
    # 剔除距离列
    fake_data = np.delete(fake_data_dis, np.shape(fake_data_dis)[1] - 1, 1)
    # print(np.shape(fake_data))

    return fake_data

def Datafiltering_No_P(fake_data,dataArr_N,dataArr_P):
    # 对扩增后样本进行筛选
    # Kmeans = KMeans(n_clusters=1)
    # kmeans_fit = Kmeans.fit(dataArr_N)  # 模型训练
    # centers_n = Kmeans.cluster_centers_  # 负样本的中心点
    # # 找出少数正类的质心：
    # Kmeans3 = KMeans(n_clusters=1)
    # kmeans_fit3 = Kmeans3.fit(dataArr_P)  # 模型训练
    # centers_p = kmeans_fit3.cluster_centers_  # 正样本的中心点
    # Kmeans2 = KMeans(n_clusters=1)
    # kmeans_fit2 = Kmeans2.fit(fake_data)  # 模型训练
    # centers_f = kmeans_fit2.cluster_centers_  # 扩增样本的中心点

    centers_n = get_density_center(dataArr_N)

    centers_p = get_density_center(dataArr_P)

    centers_f = get_density_center(fake_data)

    dis = []
    fake_data = np.mat(fake_data)
    for i in range(np.shape(fake_data)[0]):
        sum = 0
        # print("fake_data[i,:]",np.shape(fake_data[i,:]))
        # print("centers_n", np.shape(centers_n))
        # distance_n = distEclud2(fake_data[i, :], centers_n, np.shape(centers_n)[0])
        distance_n = embedding_distance(fake_data[i, :], centers_n)
        distance_p = embedding_distance(fake_data[i, :], centers_p)
        distance_f = embedding_distance(fake_data[i, :], centers_f)
        # distance_p = distEclud2(fake_data[i, :], centers_p, np.shape(centers_p)[0])
        # distance_f = distEclud2(fake_data[i, :], centers_f, np.shape(centers_f)[0])
        # 计算扩增的正样本到原正样本质心的距离
        # for j in range(np.shape(dataArr_N)[0]):
        #     b = dataArr_N[j, :]
        #     distance_SUM = distEclud2(fake_data[i, :], b, np.shape(dataArr_N)[0])
        #     sum = sum + distance_SUM
        # distance = sum
        # distance = (distance_n +distance_f)*distance_p
        distance = distance_n
        dis.append(distance)
    dis = np.mat(dis)
    dis = dis.T

    # 让扩增的正样本与其到负类质心的距离相拼接
    fake_data_dis = np.concatenate((fake_data, dis), axis=1)
    # 按照正类样本到质心的距离由大到小的排序
    a = np.lexsort(-fake_data_dis.T)
    # print(dis[np.lexsort(-dis.T)])
    fake_data_dis = fake_data_dis[a[0], :]
    # 剔除后一半的正类样本
    fake_data_dis = fake_data_dis[:np.shape(dataArr_N)[0], :]
    # 剔除距离列
    fake_data = np.delete(fake_data_dis, np.shape(fake_data_dis)[1] - 1, 1)
    # print(np.shape(fake_data))

    return fake_data

def makeNewTrdata(fake_data,dataArr_N,dataArr_P):
    lable_0 = []
    for i in range(len(dataArr_N)):
        lb_0 = -1
        lable_0.append(lb_0)
    lable_0 = pd.DataFrame(lable_0)
    lable_1 = []
    for i in range(len(fake_data)):
        lb_1 = 1
        lable_1.append(lb_1)
    lable_1 = pd.DataFrame(lable_1)
    lable_P1 = []
    for i in range(len(dataArr_P)):
        lb_p1 = 1
        lable_P1.append(lb_p1)
    lable_P1 = pd.DataFrame(lable_P1)

    fake_data = np.vstack((fake_data, dataArr_P))
    lable_1 = np.vstack((lable_1, lable_P1))
    # print(np.shape(fake_data))
    # print(np.shape(dataArr_N))
    train_x = np.vstack((fake_data, dataArr_N))
    train_y = np.vstack((lable_1, lable_0))
    return train_x,train_y

def makeNewTrdata_no_N(fake_data,dataArr_P):

    lable_0 = []
    for i in range(len(fake_data)):
        lb_0 = -1
        lable_0.append(lb_0)
    lable_0 = pd.DataFrame(lable_0)
    lable_P1 = []
    for i in range(len(dataArr_P)):
        lb_p1 = 1
        lable_P1.append(lb_p1)
    lable_P1 = pd.DataFrame(lable_P1)

    train_x = np.vstack((fake_data, dataArr_P))
    train_y = np.vstack((lable_0, lable_P1))
    # print(np.shape(fake_data))
    # print(np.shape(dataArr_N))

    return train_x,train_y


def makeNewTrdata_no_P(fake_data, dataArr_N):
    lable_0 = []
    for i in range(len(dataArr_N)):
        lb_0 = -1
        lable_0.append(lb_0)
    lable_0 = pd.DataFrame(lable_0)

    lable_1 = []
    for i in range(len(fake_data)):
        lb_1 = 1
        lable_1.append(lb_1)
    lable_1 = pd.DataFrame(lable_1)


    train_x = np.vstack((fake_data, dataArr_N))
    train_y = np.vstack((lable_1, lable_0))
    # print(np.shape(fake_data))
    # print(np.shape(dataArr_N))

    return train_x, train_y

def get_density_center(data):

    dis = []

    for i in range(np.shape(data)[0]):
        sum = 0
        for j in range(np.shape(data)[0]):

            distance = sum + embedding_distance(data[i, :], data[j, :])

        dis.append(distance)

    dis = np.mat(dis)

    data_dis = np.concatenate((data, dis.T), axis=1)

    # 按照距离由小到大的排序
    a = np.lexsort(-data_dis.T)
    data_dis = data_dis[a[0], :]

    density_center = data_dis[0,np.shape(data_dis)[1]-1]

    return density_center

def get_density_center_index(data):

    dis = []

    for i in range(np.shape(data)[0]):
        sum = 0
        for j in range(np.shape(data)[0]):

            # distance = sum + embedding_distance(data[i, :], data[j, :])
            distance = sum + np.linalg.norm(data[i, :] - data[j, :])

        dis.append(distance)

    dis = np.mat(dis).T

    index = []
    for i in range(np.shape(data)[0]):
        index.append(i)
    index = np.mat(index).T

    data_index = np.concatenate((data, index), axis=1)

    data_index_dis = np.concatenate((data_index, dis), axis=1)

    # print(data_index_dis)

    # 按照距离由大到小的排序
    a = np.lexsort(-data_index_dis.T)
    data_index_dis = data_index_dis[a[0], :]

    # print(data_index_dis)

    density_center_MaxIndex = int(data_index_dis[0,np.shape(data_index_dis)[1]-2])

    # density_center_MinIndex = int(data_index_dis[1, np.shape(data_index_dis)[1] - 2])
    # print(data_index_dis)
    density_center_MinIndex = int(data_index_dis[np.shape(data_index_dis)[0]-2, np.shape(data_index_dis)[1] - 2])

    return density_center_MaxIndex,density_center_MinIndex


if __name__ == '__main__':

    test_accuracy = []
    test_recall = []
    test_precision = []
    test_f1_score = []
    test_f2_score = []
    test_AUC = []
    test_TC = []
    test_Specificity = []
    test_Gmean = []

    path = "datasets/D001327" \
           "" \
           "/feature_label.xlsx"

    # Step 1. 导入训练数据,并划分测试集和训练集

    # 参数设置

    for i in range(10):
        dim = 20


        dataArr, testArr, LabelArr, testLabelArr= load_data(path,dim,testsize = 0.2)

        dataArr_Y1 = np.concatenate((dataArr, np.mat(LabelArr)), axis=1)
        dataArr_N1, dataArr_P1 = Matrix_division(dataArr_Y1)

        # 原始训练集散点图:
        showDataSet2(dataArr_P1, dataArr_N1)


        if ((np.shape(dataArr_N1)[0])/(np.shape(dataArr_P1)[0]))<= 250:

            # 训练集数据变换
            lmnn = LargeMarginNearestNeighbor(n_neighbors=8, verbose=2, random_state=0)
            lmnn.fit(dataArr, LabelArr)
            dataArr = lmnn.transform(dataArr)
            # 测试集数据变换
            lmnn = LargeMarginNearestNeighbor(n_neighbors=8, verbose=2, random_state=0)
            lmnn.fit(testArr, testLabelArr)
            testArr = lmnn.transform(testArr)
            dataArr_Y2 = np.concatenate((dataArr, np.mat(LabelArr)), axis=1)
            dataArr_N2, dataArr_P2 = Matrix_division(dataArr_Y2)
            showDataSet2(dataArr_P2, dataArr_N2)


        #对原始数据集进行欠采样，剔除一些与正类重合的负样本
        renn = RepeatedEditedNearestNeighbours(sampling_strategy = 'auto',n_neighbors=4,n_jobs =2)
        dataArr, LabelArr = renn.fit_resample(dataArr, LabelArr)

        dataArr_Y = np.concatenate((dataArr, np.mat(LabelArr).T), axis=1)
        dataArr_N, dataArr_P = Matrix_division(dataArr_Y)

        print("dataArr_N",np.shape(dataArr_N))
        print("dataArr_P",np.shape(dataArr_P))

        #训练集散点图:
        showDataSet2(dataArr_P,dataArr_N)

        dataArr_Y=dataArr_Y.astype(float)
        # print(dataArr_Y)
        testLabelArr = np.mat(testLabelArr).T


        # 剔除标签列
        dataArr_P = np.delete(dataArr_P, np.shape(dataArr_P)[1] - 1, 1)
        dataArr_N = np.delete(dataArr_N, np.shape(dataArr_N)[1] - 1, 1)
        test_samples = dataArr_P.shape[0]
        path_pt = 'WGAN_pt\D00' + str(path[13:17]) + 'Big_' + str(dim) + 'dim' + '.pt'
        # fake_data, fake_label = wgan_Generatorfake(path_pt,(np.shape(SubDataArr_n)[0] - test_samples) * 2)

        fake_data, fake_label = wgan_Generatorfake('WGAN_pt\D001289Big_20dim_datap.pt', (abs(np.shape(dataArr_N)[0] - test_samples))*2 )
        # fake_data, fake_label = wgan_Generatorfake('WGAN_pt\D001289Big_20dim_datap.pt', (np.shape(dataArr_N)[0] - test_samples)*2 )
        fake_data = Datafiltering2(fake_data, dataArr_N, dataArr_P)
        train_x, train_y = makeNewTrdata(fake_data, dataArr_N, dataArr_P)
        if ((np.shape(dataArr_N1)[0]) / (np.shape(dataArr_P1)[0])) <= 30:

            GBDT = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10)

            GBDT.fit(train_x, train_y)
        else:
            adboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5), n_estimators=2000, learning_rate=0.8, algorithm='SAMME')
            adboost.fit(train_x, train_y)

        print("****************************************************************************")\

        if ((np.shape(dataArr_N1)[0]) / (np.shape(dataArr_P1)[0])) <= 30:
            predicts = GBDT.predict(np.array(np.mat(testArr)))
        else:
            predicts = adboost.predict(np.array(np.mat(testArr)))
        predicts = np.mat(predicts).T
        # print(predicts)

        errArr = np.mat(np.ones((len(testArr), 1)))
        print(np.shape(testArr))
        print(np.shape(testLabelArr.T))
        print("测试集集个数：", len(testArr))
        print("正例个数：", np.sum(np.mat(testLabelArr).T == 1))

        print("负例个数：", np.sum(np.mat(testLabelArr).T == -1))
        print("TP: ", errArr[(predicts == 1) & (predicts == np.mat(testLabelArr).T)].sum())
        print("FN: ", errArr[(predicts == -1) & (predicts != np.mat(testLabelArr).T)].sum())
        print("TN: ", errArr[(predicts == -1) & (predicts == np.mat(testLabelArr).T)].sum())
        print("FP: ", errArr[(predicts == 1) & (predicts != np.mat(testLabelArr).T)].sum())
        FN = errArr[(predicts == -1) & (predicts != np.mat(testLabelArr).T)].sum()
        FP = errArr[(predicts == 1) & (predicts != np.mat(testLabelArr).T)].sum()
        P = np.sum(np.mat(testLabelArr).T == 1)
        N = np.sum(np.mat(testLabelArr).T == -1)
        print("TC:", FN * (N / P) + FP)
        print('测试集的错误率:%.3f%%' % float(errArr[predicts != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
        print('测试集的正确率:%.3f%%' % float(errArr[predicts == np.mat(testLabelArr).T].sum() / len(testArr) * 100))
        Specificity_Test = errArr[(predicts == -1) & (np.mat(testLabelArr).T == -1)].sum() / (
                errArr[(predicts == -1) & (np.mat(testLabelArr).T == -1)].sum() + FP)
        print("accuracy: ", accuracy_score(np.mat(testLabelArr).T, predicts))
        print("Specificity_Test: ", Specificity_Test)
        print("recall: ", recall_score(np.mat(testLabelArr).T, predicts))
        print("precision: ", precision_score(np.mat(testLabelArr).T, predicts))
        print("f1_score: ", f1_score(np.mat(testLabelArr).T, predicts))
        print("f2_score: ", (5 * precision_score(np.mat(testLabelArr).T, predicts) * recall_score(np.mat(testLabelArr).T,
                                                                                                  predicts)) /
              (4 * precision_score(np.mat(testLabelArr).T, predicts) + recall_score(np.mat(testLabelArr).T,
                                                                                    predicts)))
        print("AUC: ", roc_auc_score(np.mat(testLabelArr).T, predicts))
        print("G-mean", math.sqrt(recall_score(np.mat(testLabelArr).T, predicts) * Specificity_Test))

        test_accuracy.append(accuracy_score(np.mat(testLabelArr).T, predicts))
        test_recall.append(recall_score(np.mat(testLabelArr).T, predicts))
        test_precision.append(precision_score(np.mat(testLabelArr).T, predicts))
        test_f1_score.append(f1_score(np.mat(testLabelArr).T, predicts))
        test_f2_score.append((5 * precision_score(np.mat(testLabelArr).T, predicts) * recall_score(np.mat(testLabelArr).T,
                                                                                                  predicts)) /
              (4 * precision_score(np.mat(testLabelArr).T, predicts) + recall_score(np.mat(testLabelArr).T,
                                                                                    predicts)))
        test_AUC.append(roc_auc_score(np.mat(testLabelArr).T, predicts))
        test_TC.append(FN * (N / P) + FP)
        test_Gmean.append(math.sqrt(recall_score(np.mat(testLabelArr).T, predicts) * Specificity_Test))
        test_Specificity.append(Specificity_Test)
        print("---------------------------------------------------------------------")

    sum_test_accuracy = 0
    sum_test_recall = 0
    sum_test_precision = 0
    sum_test_f1_score = 0
    sum_test_f2_score = 0
    sum_test_AUC = 0
    sum_test_TC = 0
    sum_Geman = 0
    sum_Specificity = 0

    for i in test_accuracy:
        sum_test_accuracy = sum_test_accuracy + i
    for i in test_recall:
        sum_test_recall = sum_test_recall + i
    for i in test_precision:
        sum_test_precision = sum_test_precision + i
    for i in test_f1_score:
        sum_test_f1_score = sum_test_f1_score + i
    for i in test_f2_score:
        sum_test_f2_score = sum_test_f2_score + i
    for i in test_AUC:
        sum_test_AUC = sum_test_AUC + i
    for i in test_TC:
        sum_test_TC = sum_test_TC + i
    for i in test_Gmean:
        sum_Geman = sum_Geman + i
    for i in test_Specificity:
        sum_Specificity = sum_Specificity + i

    print("test_TC:", sum_test_TC / len(test_TC))
    print("test_accuracy:", (sum_test_accuracy / len(test_accuracy))*100)
    print("test_Specificity:", sum_Specificity / len(test_Specificity))
    print("test_reacll:", sum_test_recall / len(test_recall))
    print("test_precision:", sum_test_precision / len(test_precision))
    print("test_f1_score:", sum_test_f1_score / len(test_f1_score))
    print("test_f2_score:", sum_test_f2_score / len(test_f2_score))
    print("test_AUC:", sum_test_AUC / len(test_AUC))
    print("test_Geman:", sum_Geman / len(test_Gmean))


