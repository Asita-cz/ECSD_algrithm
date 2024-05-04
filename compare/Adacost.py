# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr
import math
from sklearn.decomposition import KernelPCA
import numpy as np
import pandas as pd
from sklearn import decomposition

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error

import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.ensemble import AdaBoostClassifier

a =0

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
                res.append(-1)   # 正确分类，系数保持不变，按原来的比例减少
            elif i[0] == 1 and i[1] == -1:
                res.append(7.2)  # 在信用卡的情景下，将好人误杀代价应该更大一些，比原来的增加比例要高
            elif i[0] == -1 and i[1] == 1:
                res.append(1)  # 将负例误判为正例，代价不变，按原来的比例增加

        return np.array(res)

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


    kpca = KernelPCA(n_components=dim, kernel="rbf", gamma=10, fit_inverse_transform=True, remove_zero_eig=True)
    dataset = kpca.fit_transform(dataset)

    Y = np.vstack((Y_0, Y_1))

    # 把负类的标签全部用-1表示
    for i in range(np.shape(Y)[0]):
        if Y[i] == 0:
            Y[i] = -1

    dataArr, testArr, LabelArr, testLabelArr = train_test_split(dataset, Y, test_size=0.2, random_state=1, stratify=Y)



    return dataArr,testArr,LabelArr,testLabelArr

if __name__ == '__main__':
    path = "imbalance/D001289_Big" \
           "" \
           "/1289.xlsx"

    # Step 1. 导入训练数据
    dim = 20
    dataArr, testArr, LabelArr, testLabelArr = load_data(path, dim)

    testLabelArr = np.mat(testLabelArr).T

    P = np.sum(np.mat(testLabelArr).T == 1)
    N = np.sum(np.mat(testLabelArr).T == -1)

    clf2 = AdaCostClassifier(n_estimators=10)
    clf2.fit(dataArr, LabelArr)
    predict = clf2.predict(testArr)
    predict = np.mat(predict)
    predict = predict.T
    errArr = np.mat(np.ones((len(testArr), 1)))

    print("训练集个数：", len(dataArr))
    print("正例个数：", np.sum(np.mat(LabelArr).T == 1))
    print("负例个数：", np.sum(np.mat(LabelArr).T == -1))
    print("测试集集个数：", len(testArr))
    print("正例个数：", np.sum(np.mat(testLabelArr).T == 1))
    print("负例个数：", np.sum(np.mat(testLabelArr).T == -1))

    print("TP: ", errArr[(predict == 1) & (predict == np.mat(testLabelArr).T)].sum())
    print("FN: ", errArr[(predict == -1) & (predict != np.mat(testLabelArr).T)].sum())
    print("TN: ", errArr[(predict == -1) & (predict == np.mat(testLabelArr).T)].sum())
    print("FP: ", errArr[(predict == 1) & (predict != np.mat(testLabelArr).T)].sum())
    FN = errArr[(predict == -1) & (np.mat(testLabelArr) == 1).T].sum()
    FP = errArr[(predict == 1) & (np.mat(testLabelArr) == -1).T].sum()
    a = N/P
    print("TC:", FN * (N / P) + FP)
    print('测试集的错误率:%.3f%%' % float(errArr[predict != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
    print('测试集的正确率:%.3f%%' % float(errArr[predict == np.mat(testLabelArr).T].sum() / len(testArr) * 100))
    Specificity_Test = errArr[(predict == -1) & (np.mat(testLabelArr).T == -1)].sum() / (
            errArr[(predict == -1) & (np.mat(testLabelArr).T == -1)].sum() + FP)
    print("accuracy: ", accuracy_score(np.mat(testLabelArr).T, predict))
    print("Specificity_Test: ", Specificity_Test)
    print("recall: ", recall_score(np.mat(testLabelArr).T, predict))
    print("precision: ", precision_score(np.mat(testLabelArr).T, predict))
    print("f1_score: ", f1_score(np.mat(testLabelArr).T, predict))
    print("f2_score: ", (5 * precision_score(np.mat(testLabelArr).T, predict) * recall_score(np.mat(testLabelArr).T,
                                                                                                 predict)) /
          (4 * precision_score(np.mat(testLabelArr).T, predict) + recall_score(np.mat(testLabelArr).T,
                                                                                   predict)))
    print("AUC: ", roc_auc_score(np.mat(testLabelArr).T, predict))
    print("G-mean", math.sqrt(recall_score(np.mat(testLabelArr).T, predict) * Specificity_Test))