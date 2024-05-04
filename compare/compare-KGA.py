from sklearn.neural_network import MLPClassifier
import torch.nn.functional as nn_f
import pandas as pd
import numpy as np
import math
from sklearn import svm
from sklearn import decomposition
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)

import warnings

import time
start_time = time.time()
warnings.filterwarnings("ignore")


dim = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CUDA_VISIBLE_DEVICES = 1


# 读取数据--可以按照自己的方式

dataset = '03015'

Batch_size = 2

path = "datasets/D0"+str(dataset)+ \
           "" \
           "/feature_label.xlsx"

data_0 = pd.read_excel(path, sheet_name='major',
                           header=None)  # Read most types of data
data_1 = pd.read_excel(path, sheet_name='minor',
                           header=None)  # Read minority data



# 将新数据保存输出（可以不用，这是做可视化用的）
# Excel_result = pd.ExcelWriter('..\Ablation experiment\KGA_dataAu')

# a = 0
# 此处加循环是为了得到10次结果进而求平均，如果不需要可以直接复制里面的内容到自己的文件即可
for i in range(1):
    # print(i + 1)
    # a = a + 1

    # 接下来是划分数据，将原始数据划分为测试集和训练集，因为想体现随机性和保证测试集的IR（不平衡比），所以自己划分的
    # 可以用自己的原始训练集
    num_test_label1 = int(data_1.shape[0] / 5)  # Select as part of the generated data
    num_test_label0 = int(num_test_label1 * data_0.shape[0] / data_1.shape[0])
    # print(num_test_label1)
    # print(num_test_label0)
    test_label1 = data_1.sample(n=num_test_label1, axis=0, replace=False)
    test_label0 = data_0.sample(n=num_test_label0, axis=0, replace=False)
    # print(num_test_label1)
    # print(test_label1.shape[0])
    # print(test_label0.shape[0])

    index_1 = test_label1.index
    index_0 = test_label0.index
    train_label1 = data_1.drop(index_1)
    train_label0 = data_0.drop(index_0)
    print("num train1: %d" % train_label1.shape[0])
    print("num train0: %d" % train_label0.shape[0])

    test1 = np.vstack([test_label0, test_label1])
    train1 = np.vstack([train_label0, train_label1])
    # print(test1.shape[0])
    # print(train1.shape[0])
    test = pd.DataFrame(test1)
    train = pd.DataFrame(train1)
    # print(train.iloc[:, 0])

    # 得到训练集和测试集---这里只要是原始训练集就行，可以不用我上面的划分方式
    train_f, train_y1 = train.iloc[:, 1:].values, train.iloc[:, 0].values
    print("train num: %d" % train_f.shape[0])
    train_y0 = train.iloc[:train_label0.shape[0], 0].values
    test_f, test_y = test.iloc[:, 1:].values, test.iloc[:, 0].values
    print("test num: %d" % test_f.shape[0])

    # Data normalization processing--归一化（必须要）
    standard_train_x = preprocessing.MinMaxScaler()  # [0,1]
    scaler_standard = standard_train_x.fit(train_f)  # 生成规则
    standard_train_feature = scaler_standard.transform(train_f)
    standard_test_feature = scaler_standard.transform(test_f)

    #  KPCA 特征转换（必须要）--维度可以自己调试
    KPCA_lc_train_x1 = decomposition.KernelPCA(n_components=dim,gamma=8, kernel='rbf')
    kpca_standard = KPCA_lc_train_x1.fit(standard_train_feature)  # 生成规则
    kpca_train_xx = kpca_standard.transform(standard_train_feature)
    kpca_test_xx = kpca_standard.transform(standard_test_feature)

    kpca_train_feature = pd.DataFrame(kpca_train_xx)
    kpca_test_feature = pd.DataFrame(kpca_test_xx)

    test_x = kpca_test_feature

    # 获取用来生成新数据的正样本及个数
    train_label_00 = pd.DataFrame(train_label0)
    num_train0 = train_label_00.shape[0]

    generate1 = kpca_train_feature.iloc[num_train0:, :]
    # standard_train_feature1 = pd.DataFrame(standard_train_feature)

    print(generate1.shape[0])
    # print(train_0)

    # GAN generates data--GAN生成样本的个数，
    test_samples = int(train_label0.shape[0] * 1)  # IR
    print("generate :%d" % test_samples)
    # Cost_multiple = int(test_samples / short_samples)

    X_cols = generate1.shape[1]
    # print(X_cols)

    z_dim = 32
    hidden_dim = 32

    X2 = generate1.values

    # Definition generator
    net_G = nn.Sequential(
        nn.Linear(z_dim, hidden_dim),  # 32*128
        nn.ReLU(),
        nn.Linear(hidden_dim, X_cols))  # 128*n(PCA降成n维数据)

    # Define the discriminator
    net_D = nn.Sequential(
        nn.Linear(X_cols, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid())

    # Put the network on the GPU
    net_G = net_G.to(device)
    net_D = net_D.to(device)

    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=0.000085)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=0.000085)

    # batch_size可以改，应该是要整除输入样本的个数才可以，不然报错
    batch_size = Batch_size
    nb_epochs = 500   # 迭代次数可以改，维度数据集是500效果比较好

    loss_D_epoch = []
    loss_G_epoch = []

    for e in range(nb_epochs):
        np.random.shuffle(X2)
        real_samples = torch.from_numpy(X2).type(torch.FloatTensor)
        loss_G = 0
        loss_D = 0
        for t, real_batch in enumerate(real_samples.split(batch_size)):
            z = torch.empty(batch_size, z_dim).normal_().to(device)
            fake_batch = net_G(z)

            # Input the true and false samples into the discriminator, and get the result
            D_scores_on_real = net_D(real_batch.to(device))
            D_scores_on_fake = net_D(fake_batch)

            loss = -torch.mean(torch.log(1 - D_scores_on_fake) + torch.log(D_scores_on_real))

            optimizer_D.zero_grad()

            loss.backward()

            optimizer_D.step()
            loss_D += loss

            # Fixed discriminator, improved generator
            # Generate a set of random noise, input the generator to get a set of fake samples
            z = torch.empty(batch_size, z_dim).normal_().to(device)
            fake_batch = net_G(z)
            # Fake sample input into the discriminator to get score
            D_scores_on_fake = net_D(fake_batch)

            loss = -torch.mean(torch.log(D_scores_on_fake))
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            loss_G += loss

    # if e % 50 == 0:
    # print(f'\n Epoch {e} , D loss: {loss_D}, G loss: {loss_G}')

    loss_D_epoch.append(loss_D)
    loss_G_epoch.append(loss_G)

    z = torch.empty(test_samples, z_dim).normal_().to(device)
    # print(z)
    fake_samples = net_G(z)

    # 生成数据fake_data1
    fake_data1 = fake_samples.cpu().data.numpy()

    # 如果需要看生成的数据，可以打印出来，不需要就算了
    # pd.DataFrame(fake_data1).to_excel(Excel_result, sheet_name='gan_train', index=None, startrow=0, header=None)

    # 接下来为生成的数据加标签--正类为1
    # fake_data_1 = pd.DataFrame(fake_data1)
    lable_0 = []
    for j in range(len(fake_data1)):
        lb_1 = 1
        lable_0.append(lb_1)
    lable_0p = pd.DataFrame(lable_0)
    # print(fake_data1.shape[1])

    # Divide the data set

    # Training set
    # train_y0 = pd.DataFrame(train_y0)
    # train_x = np.vstack((train_0, fake_data1))  # Generated 1 does not participate in training
    # train_y = np.vstack((train_y0, lable_0p))
    train_y11 = pd.DataFrame(train_y1)
    # train_x = np.vstack((standard_train_feature, fake_data1))  #

    # 将原始正类训练样本和新生成的样本组合在一起，得到新的训练集train_x（特征），train_y（标签）
    train_x = np.vstack((kpca_train_feature, fake_data1))
    train_y = np.vstack((train_y11, lable_0p))

    train_y = np.mat(train_y)
    test_y = np.mat(test_y)
    test_y = test_y.T
    # # 把负类的标签全部用-1表示
    # for i in range(np.shape(train_y)[0]):
    #     if train_y[i] == 0:
    #         train_y[i] = -1
    #
    # for i in range(np.shape(test_y)[0]):
    #     if test_y[i] == 0:
    #         test_y[i] = -1

    dataArr_Y = np.concatenate((train_x, np.mat(train_y)), axis=1)
    test_Y = np.concatenate((test_x, np.mat(test_y)), axis=1)
    # 将训练数据输出，不需要就可以不用
    df = pd.DataFrame(dataArr_Y)
    df2 = pd.DataFrame(test_Y)
    # 将DataFrame保存为.xlsx文件
    # df.to_excel('Ablation experiment/KGA_dataAu/'+str(dataset)+'train.xlsx', index=False)
    # df2.to_excel('Ablation experiment/KGA_dataAu/'+str(dataset)+'test.xlsx', index=False)


train_y = np.mat(train_y)
test_y = np.mat(test_y)
test_y = test_y[0]
test_y = test_y.T

# 把负类的标签全部用-1表示
for i in range(np.shape(train_y)[0]):
    if train_y[i] == 0:
        train_y[i] = -1

for i in range(np.shape(test_y)[0]):
    if test_y[i] == 0:
        test_y[i] = -1

# 接下来将数据train_x（特征），train_y（标签）输入分类器或其他操作
# print(np.shape(train_x))
# print(np.shape(train_y))
# print(np.shape(test_x))
# print(np.shape(test_y))

testArr = test_x
testLabelArr = test_y

clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10)
clf.fit(train_x,train_y)
predicts = clf.predict(testArr)


# # 创建MLP分类器
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
# # 训练分类器
# mlp_classifier.fit(train_x, train_y)
# 使用训练好的分类器进行预测
# predicts = mlp_classifier.predict(test_x)





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