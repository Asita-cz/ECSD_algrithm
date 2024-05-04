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

    dataset = '10661'
    dim = 20
    annoy_max_sub_num = 200
    n_neighbors_lmnn = 0
    n_neighbors_renn = 3
    sub_imbrate = 0.5
    divided_n = 3

    path_train = 'Ablation experiment/MFCS_datasets/'+'D0'+str(dataset)+'/kuoz.xls'

    path_test = 'Ablation experiment/MFCS_datasets/' +'D0'+str(dataset)+'/D0'+str(dataset)+'test.xls'

    # Step 1. 导入训练数据,并划分测试集和训练集

    # 参数设置

    # 加载训练集数据
    train_data = pd.read_excel(path_train)


    # 加载测试集数据
    test_data = pd.read_excel(path_test)


    # 分割训练集特征和标签
    y_train = train_data.iloc[:, 0]  # 第一列是标签
    X_train = train_data.iloc[:, 1:]  # 后面的列是特征

    kpca = KernelPCA(n_components=dim, kernel="rbf", gamma=15, fit_inverse_transform=True, remove_zero_eig=True)
    X_train = kpca.fit_transform(X_train)

    # 分割测试集特征和标签
    y_test = test_data.iloc[:, 0]  # 第一列是标签
    X_test = test_data.iloc[:, 1:]  # 后面的列是特征

    kpca = KernelPCA(n_components=dim, kernel="rbf", gamma=15, fit_inverse_transform=True, remove_zero_eig=True)
    X_test = kpca.fit_transform(X_test)

    # 把负类的标签全部用-1表示
    for i in range(np.shape(y_train)[0]):
        if y_train[i] == 0:
            y_train[i] = -1

    for i in range(np.shape(y_test)[0]):
        if y_test[i] == 0:
            y_test[i] = -1

    dataArr = X_train

    LabelArr = np.mat(y_train).T



    testArr = X_test

    testLabelArr = np.mat(y_test).T


    dataArr_Y = np.concatenate((dataArr, np.mat(LabelArr)), axis=1)
    dataArr_N, dataArr_P = Matrix_division(dataArr_Y)

    print(np.shape(dataArr_N)[0])
    print(np.shape(dataArr_P)[0])


    if ((np.shape(dataArr_N)[0])/(np.shape(dataArr_P)[0]))<= 0:

        # 训练集数据变换
        lmnn = LargeMarginNearestNeighbor(n_neighbors=n_neighbors_lmnn, verbose=False, random_state=0)
        lmnn.fit(dataArr, LabelArr)
        dataArr = lmnn.transform(dataArr)
        # 测试集数据变换
        lmnn = LargeMarginNearestNeighbor(n_neighbors=n_neighbors_lmnn, verbose=False, random_state=0)
        lmnn.fit(testArr, testLabelArr)
        testArr = lmnn.transform(testArr)
        dataArr_Y2 = np.concatenate((dataArr, np.mat(LabelArr)), axis=1)
        dataArr_N2, dataArr_P2 = Matrix_division(dataArr_Y2)

    # 对原始数据集进行欠采样，剔除一些与正类重合的负样本
    renn = RepeatedEditedNearestNeighbours(sampling_strategy='auto', n_neighbors=n_neighbors_renn, n_jobs=2)
    dataArr, LabelArr = renn.fit_resample(dataArr, LabelArr)

    dataArr_Y=dataArr_Y.astype(float)
    # print(dataArr_Y)
    testLabelArr = np.mat(testLabelArr).T


    # 剔除标签列
    dataArr_P = np.delete(dataArr_P, np.shape(dataArr_P)[1] - 1, 1)
    dataArr_N = np.delete(dataArr_N, np.shape(dataArr_N)[1] - 1, 1)


    print(type(dataArr))
    at1 = annoytree(dataArr,annoy_max_sub_num)


    # 中序遍历二叉树,并获取叶子结点
    leaf = []
    inorder(at1.root)
    print("子空间的个数为：", get_num_leaf(at1.root))
    print(leaf)
    # 分配子空间样本矩阵
    prepare_list = Partitioned_subspace_matrix(at1, leaf, dataArr_Y)
    print(type(at1))
    print(type(leaf))

    sub_boss = []
    sub_density_center = []
    # 遍历字典中的每个子空间矩阵
    for i in range(len(leaf)):

        print("第" + str(i) + "个子空间的形状:", np.shape(prepare_list['subspace_matrix' + str(i)]))
        print("第" + str(i) + "个子空间的样本个数:", len(prepare_list['subspace_matrix' + str(i)]))

        # 子空间正类样本矩阵
        sub_density_center.append(get_density_center(prepare_list['subspace_matrix' + str(i)]))
        SubDataArr_p = []

        SubDataArr_n = []

        p = n = 0
        # 遍历每个子空间的矩阵的元素
        for j in range(len(prepare_list['subspace_matrix' + str(i)])):
            # 判断是否为正样本：
            if prepare_list['subspace_matrix' + str(i)][
                j, np.shape(prepare_list['subspace_matrix' + str(i)])[1] - 1] == 1:
                SubDataArr_p.append(prepare_list['subspace_matrix' + str(i)][j].tolist())
                p = p + 1
            # 判断是否为负样本：
            if prepare_list['subspace_matrix' + str(i)][
                j, np.shape(prepare_list['subspace_matrix' + str(i)])[1] - 1] == -1:
                SubDataArr_n.append(prepare_list['subspace_matrix' + str(i)][j].tolist())
                n = n + 1
        SubDataArr_p = np.array(SubDataArr_p)
        SubDataArr_p = np.mat(SubDataArr_p)
        SubDataArr_n = np.array(SubDataArr_n)
        SubDataArr_n = np.mat(SubDataArr_n)
        print("扩增前子空间正类样本的个数：", p)
        print("扩增前子空间负类样本的个数：", n)
        print("扩增前子空间正类样本矩阵的形状：", np.shape(SubDataArr_p))
        # print(SubDataArr_p)
        print("扩增前子空间负类样本矩阵的形状：", np.shape(SubDataArr_n))
        # print(SubDataArr_n)
        # 子空间



        if (p == 0) and (n != 0):

            SubDataArr_n = np.delete(SubDataArr_n, np.shape(SubDataArr_n)[1] - 1, 1)
            # test_samples = dataArr_P.shape[0]
            # print("子空间训练集中总的正类样本数量：", test_samples)

            # fake_data, fake_label = wgan_Generatorfake(path_pt,(np.shape(SubDataArr_n)[0] - test_samples) * 2)
            fake_data, fake_label = wgan_Generatorfake('WGAN_pt\D001289Big_20dim_datap.pt',
                                                       (np.shape(SubDataArr_n)[0]) * 2)
            print("筛选前：", np.shape(fake_data))
            # 子空间扩增后

            # fake_data = Datafiltering2(fake_data, SubDataArr_n, SubDataArr_p)
            print("筛选后：", np.shape(fake_data))
            # 子空间扩增筛选后
            train_x, train_y = makeNewTrdata_no_P(fake_data, SubDataArr_n)
            print(np.shape(train_x))

        elif (n == 0) and (p != 0):
            SubDataArr_p = np.delete(SubDataArr_p, np.shape(SubDataArr_p)[1] - 1, 1)
            # test_samples = SubDataArr_p.shape[0]
            # print("子空间训练集中总的正类样本数量：", test_samples)

            # fake_data, fake_label = wgan_Generatorfake(path_pt,(np.shape(SubDataArr_n)[0] - test_samples) * 2)
            fake_data, fake_label = wgan_GeneratorNfake('WGAN_pt\D001289Big_20dim_datap.pt',
                                                        (np.shape(SubDataArr_p)[0]) * 2)
            print("筛选前：", np.shape(fake_data))
            # 子空间扩增后

            # fake_data = Datafiltering_No_P(fake_data, SubDataArr_n, SubDataArr_p)
            print("筛选后：", np.shape(fake_data))
            # 子空间筛选后

            train_x, train_y = makeNewTrdata_no_N(fake_data, SubDataArr_p)
            print(np.shape(train_x))


        elif (p / n) <= sub_imbrate:

            # path_wgan = 'WGAN_pt\D00' + str(path[13:17]) + 'Big_' + str(dim) + 'dim' + '.pt'
            # wgan(SubDataArr_p,3000,path_wgan)
            SubDataArr_p = np.delete(SubDataArr_p, np.shape(SubDataArr_p)[1] - 1, 1)
            SubDataArr_n = np.delete(SubDataArr_n, np.shape(SubDataArr_n)[1] - 1, 1)
            test_samples = SubDataArr_p.shape[0]
            print("子空间训练集中总的正类样本数量：", test_samples)

            # fake_data, fake_label = wgan_Generatorfake(path_pt,(np.shape(SubDataArr_n)[0] - test_samples) * 2)
            fake_data, fake_label = wgan_Generatorfake('WGAN_pt\D001289Big_20dim_datap.pt',
                                                       (np.shape(SubDataArr_n)[0] - test_samples) * 2)
            print("筛选前：", np.shape(fake_data))
            # 子空间扩增

            # fake_data = Datafiltering2(fake_data, SubDataArr_n, SubDataArr_p)
            print("筛选后：", np.shape(fake_data))
            # 子空间扩增筛选后

        # 扩增后的训练集散点图:
        else:
            SubDataArr_p = np.delete(SubDataArr_p, np.shape(SubDataArr_p)[1] - 1, 1)
            SubDataArr_n = np.delete(SubDataArr_n, np.shape(SubDataArr_n)[1] - 1, 1)
            # train_x = np.concatenate((SubDataArr_p, SubDataArr_n), axis=0)
            train_x, train_y = makeNewTrdata2(SubDataArr_n, SubDataArr_p)



            # exec(f"GBDT{i} = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10)")
        exec(f"GBDT{i} = GradientBoostingClassifier(learning_rate=0.8, n_estimators=2000, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=0)")
        # exec(f"MLP{i} = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)")
        # eval(f'MLP{i}').fit(train_x, train_y)
        eval(f'GBDT{i}').fit(train_x, train_y)


    sub_density_center = np.mat(sub_density_center).T


    predicts = []
    for k in range(np.shape(testArr)[0]):
        dis = []
        # 计算到各个空间密度中心的距离
        for j in range(np.shape(sub_density_center)[0]):
            # dis_centers = distEclud2(np.mat(testArr)[k,:],svdd_centers[j,:],np.shape(svdd_centers)[0])
            dis_centers = embedding_distance(np.mat(testArr)[k, :], sub_density_center[j, :])
            # dis_centers = embedding_distance(np.mat(testArr)[k, :], sub_boss[j, :])
            dis.append(dis_centers)
        dis = np.mat(dis).T
        # 子空间序号矩阵
        subspace_center_index = []

        for a in range(np.shape(sub_density_center)[0]):
            subspace_center_index.append(a)
        subspace_center_index = np.mat(subspace_center_index).T
        # print(np.shape(dis))
        # print(np.shape(subspace_center_index))
        dis = np.concatenate((subspace_center_index, dis), axis=1)
        b = np.lexsort(dis.T)
        dis_a = dis[b[0], :]


        all = 0
        divided_num = divided_n
        # for er in range(int(np.shape(sub_density_center)[0]/8)):
        # 设置多少个分类器一起审判
        for er in range(int(np.shape(sub_density_center)[0]/divided_num)):
            all = all +dis_a[er,1]

        predict_all = 0
        # for re in range(int(np.shape(sub_density_center)[0]/2)):
        for re in range(int(np.shape(sub_density_center)[0]/divided_num)):
            # print("re",re)
            # print(dis_a)
            # print(dis_a[re, 0])
            if ((np.shape(dataArr_N)[0]) / (np.shape(dataArr_P)[0])) <= 30:
                predict = (dis_a[re, 1] / all) * eval(f'GBDT{int(dis_a[re, 0])}').predict(np.array(np.mat(testArr[k, :])))
            else:
                predict = (dis_a[re, 1]/all)*eval(f'adboost_{int(dis_a[re, 0])}').predict(np.array(np.mat(testArr[k, :])))

            # print(predict)
            # print(predict[0])
            predict_all = predict_all + predict[0]
            print(predict_all)

        predicts.append(np.sign(predict_all))

    print("****************************************************************************")
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

    print("---------------------------------------------------------------------")




