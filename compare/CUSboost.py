# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:54:20 2017
@author: Farshid Rayhan, United International University
"""
from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import math
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import train_test_split
from compare.cus_sampling import cus_sampler
from sklearn.cluster import KMeans
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


class CUSBoostClassifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth
        self.undersampler = RandomUnderSampler(replacement=False)

        ## Some other samplers to play with ######
        # self.undersampler = EditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = AllKNN(return_indices=True,n_neighbors=neighbours,n_jobs=4)

    def fit(self, X_train, y_train, number_of_clusters, percentage_to_choose_from_each_cluster):
        self.models = []
        self.alphas = []

        N, _ = X_train.shape
        # ,_ 는 특정 값을 무시하기 위해 사용됌ex. x, _ ,y = (1, 2, 3)  ..x=1, y=3
        W = np.ones(N) / N

        # W = weight

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')

            X_undersampled, y_undersampled, chosen_indices = cus_sampler(X_train, y_train, number_of_clusters,
                                                                         percentage_to_choose_from_each_cluster)

            tree.fit(X_undersampled, y_undersampled,
                     sample_weight=W[chosen_indices])  # fitting tree with cluster-sampled instances

            P = tree.predict(
                X_train)  # predicting the trained tree with X_train instances, which is not the undersampled instances
            P_int = P.astype(int)  # for indexing

            Prediction = np.ones(N)  # to index negative values
            negative_index = (Prediction != P_int)  # indexes of prediction 0
            Prediction[negative_index] = -1

            y_train_value = np.ones(N)  # to indexnegative values
            y_int = y_train.astype(int)
            negative_index_y = (y_train_value != y_int)  # index of prediction (label as 0) to -1

            y_train_value[negative_index_y] = -1

            err = np.sum(W[P != y_train])
            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0:
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))  # alpha is assigned vote based on error
                    W = W * np.exp(
                        -alpha * y_train_value * Prediction)  # if y_train_value & Prediction is same, weight become lower
                    W = W / W.sum()  # normalize so it sums to 1         # if y_train_value & Prediction is not same, weight become large.
                except:
                    alpha = 0
                    # W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tree)
                self.alphas.append(alpha)

    def predict(self, X_test):
        N, _ = X_test.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            Prediction = np.ones(N)
            P = tree.predict(X_test)
            P_int = P.astype(int)

            negative_index = (Prediction != P_int)

            Prediction[negative_index] = -1  # instance that have been classified as negative(0) becomes -1

            FX += alpha * Prediction  # alpha1 * 1 + alphas2 *-1 +...
        FX = np.sign(FX)  # threshold = 0

        FX[FX == -1] = 0

        return FX  # 0 for negative, 1 for positive

    def predict_proba(self, X_test):
        # if self.alphas == 'SAMME'
        # 재호 수정 5/20

        N, _ = X_test.shape
        proba = np.zeros([N, 2])
        for tree, alpha in zip(self.models, self.alphas):
            each_prob = tree.predict_proba(X_test) * alpha
            proba += each_prob

        normalizer = sum(self.alphas)
        if normalizer <= 0:
            normalizer = 0.000001

        proba = proba / normalizer

        # proba = sum(tree.predict_proba(X_test) * alpha for tree , alpha in zip(self.models,self.alphas) )

        # proba = np.array(proba)

        # proba = proba / sum(self.alphas)

        # proba = np.exp((1. / (2 - 1)) * proba)
        # normalizer = proba.sum(axis=1)[:, np.newaxis]
        # normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        # proba = proba /  normalizer

        # print(proba)
        return proba




def load_data(path,dim):

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

    dataArr, testArr, LabelArr, testLabelArr = train_test_split(dataset, Y, test_size=0.2, random_state=1, stratify=Y)

    return dataArr,testArr,LabelArr,testLabelArr

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

path = "datasets/D004827" \
       "" \
       "/feature_label.xlsx"

# Step 1. 导入训练数据,并划分测试集和训练集

# 参数设置


dim = 20


dataArr, testArr, LabelArr, testLabelArr = load_data(path, dim)


# predict
# y_pred_test = clf.predict_proba(testArr)
clf2 = CUSBoostClassifier(depth=2,n_estimators=20)

clf2.fit(dataArr, LabelArr,number_of_clusters=2,percentage_to_choose_from_each_cluster=0.5)
predicts = clf2.predict_proba_samme(testArr)
predicts = np.mat(predicts).T
# print(predicts)

errArr = np.mat(np.ones((len(testArr), 1)))
print(np.shape(testArr))
print(np.shape(testLabelArr.T))
print("测试集集个数：", len(testArr))
print("正例个数：", np.sum(np.mat(testLabelArr) == 1))
print("负例个数：", np.sum(np.mat(testLabelArr) == -1))
print("TP: ", errArr[(predicts == 1) & (predicts == np.mat(testLabelArr))].sum())
print("FN: ", errArr[(predicts == -1) & (predicts != np.mat(testLabelArr))].sum())
print("TN: ", errArr[(predicts == -1) & (predicts == np.mat(testLabelArr))].sum())
print("FP: ", errArr[(predicts == 1) & (predicts != np.mat(testLabelArr))].sum())
FN = errArr[(predicts == -1) & (predicts != np.mat(testLabelArr))].sum()
FP = errArr[(predicts == 1) & (predicts != np.mat(testLabelArr))].sum()
P = np.sum(np.mat(testLabelArr).T == 1)
N = np.sum(np.mat(testLabelArr).T == -1)
print("TC:", FN * (N / P) + FP)
print('测试集的错误率:%.3f%%' % float(errArr[predicts != np.mat(testLabelArr)].sum() / len(testArr) * 100))
print('测试集的正确率:%.3f%%' % float(errArr[predicts == np.mat(testLabelArr)].sum() / len(testArr) * 100))
Specificity_Test = errArr[(predicts == -1) & (np.mat(testLabelArr) == -1)].sum() / (
        errArr[(predicts == -1) & (np.mat(testLabelArr) == -1)].sum() + FP)
print("accuracy: ", accuracy_score(np.mat(testLabelArr), predicts))
print("Specificity_Test: ", Specificity_Test)
print("recall: ", recall_score(np.mat(testLabelArr), predicts))
print("precision: ", precision_score(np.mat(testLabelArr), predicts))
print("f1_score: ", f1_score(np.mat(testLabelArr), predicts))
print("f2_score: ", (5 * precision_score(np.mat(testLabelArr), predicts) * recall_score(np.mat(testLabelArr),
                                                                                          predicts)) /
      (4 * precision_score(np.mat(testLabelArr), predicts) + recall_score(np.mat(testLabelArr),
                                                                            predicts)))
print("AUC: ", roc_auc_score(np.mat(testLabelArr), predicts))
print("G-mean", math.sqrt(recall_score(np.mat(testLabelArr), predicts) * Specificity_Test))

print("---------------------------------------------------------------------")