# load dataset & prepare environment
from compare.duplebalance import DupleBalanceClassifier
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.datasets import make_classification


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




# ensemble training
clf = DupleBalanceClassifier(
    n_estimators=10,
    random_state=42,
    k_bins = 3,


    ).fit(dataArr, LabelArr)

# predict
# y_pred_test = clf.predict_proba(testArr)
predicts = clf.predict(testArr)

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