import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import math
from sklearn import svm
from sklearn import decomposition
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
import torch.nn as nn
import torch
import warnings
import time
start_time = time.time()
warnings.filterwarnings("ignore")
dataset = '04827'


path_train = 'Ablation experiment/KGA_dataAu/'+str(dataset)+'train.xlsx'

path_test = 'Ablation experiment/KGA_dataAu/' + str(dataset)+'test.xlsx'

# Step 1. 导入训练数据,并划分测试集和训练集

# 参数设置

# 加载训练集数据
train_data = pd.read_excel(path_train)

# 加载测试集数据
test_data = pd.read_excel(path_test)

# 分割训练集特征和标签
X_train = train_data.iloc[:, :-1]  # 前面的列是特征
y_train = train_data.iloc[:, -1]  # 最后一列是标签

# 分割测试集特征和标签
X_test = test_data.iloc[:, :-1]  # 前面的列是特征
y_test = test_data.iloc[:, -1]  # 最后一列是标签


dataArr = X_train

LabelArr = np.mat(y_train).T



testArr = X_test

testLabelArr = np.mat(y_test).T

clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10)
clf.fit(dataArr,LabelArr)

predicts = clf.predict(testArr)
errArr = np.mat(np.ones((len(testArr), 1)))


print("errarr",np.shape(errArr))
print("testArr",np.shape(testArr))

predicts = np.mat(predicts).T
print("predicts",np.shape(predicts))
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
end_time = time.time()
execution_time = end_time - start_time
print("time:\n")
print("execution_time", execution_time)