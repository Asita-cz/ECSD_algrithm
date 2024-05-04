# -*- coding: utf-8 -*-
"""
The implementation of SMOTEBoost.
"""

# Authors: Simona Nitti, Gabriel Rozzonelli
# Based on the work of the following paper:
# [1] N. Chawla, A. Lazarevic, L. Hall, et K. Bowyer, « SMOTEBoost: 
#     Improving Prediction  of the Minority Class in Boosting ».
import math

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import is_classifier, clone
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_random_state, check_array
from sklearn.utils import check_X_y
#from sklearn.utils import shuffle


class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).

    SMOTE performs oversampling of the minority class by picking target
    minority class samples and their nearest minority class neighbors and
    generating new samples that linearly combine features of each target
    sample with features of its selected minority class neighbors [1].

    Parameters
    ----------
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE:
           Synthetic Minority Over-Sampling Technique." Journal of Artificial
           Intelligence Research (JAIR), 2002.
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.

        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(
                self.X[j].reshape(1, -1), return_distance=False
            )[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self


class SMOTEBoost(AdaBoostClassifier):
    """Implementation of SMOTEBoost.

    SMOTEBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class using SMOTE on each boosting iteration [1].

    This implementation inherits methods from the scikit-learn
    AdaBoostClassifier class, only modifying the `fit` method.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.

    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.

    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in
           Boosting." European Conference on Principles of Data Mining and
           Knowledge Discovery (PKDD), 2003.
    """

    def __init__(
        self,
        n_samples=100,
        k_neighbors=5,
        base_estimator=None,
        n_estimators=20,
        learning_rate=1.,
        algorithm="SAMME.R",
        random_state=None,
    ):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        minority_target : int
            Minority class label.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or isinstance(
            self.base_estimator, (BaseDecisionTree, BaseForest)
        )):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = "csc"
        else:
            dtype = None
            accept_sparse = ["csr", "csc"]

        X, y = check_X_y(
            X,
            y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            y_numeric=is_regressor(self),
        )

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples."
                )

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            X_min = X[np.where(y == self.minority_target)]

            # SMOTE step.
            if len(X_min) >= self.smote.k:
                self.smote.fit(X_min)
                X_syn = self.smote.sample(self.n_samples)
                y_syn = np.full(
                    X_syn.shape[0],
                    fill_value=self.minority_target,
                    dtype=np.int64,
                )

                # Normalize synthetic sample weights based on current training set.
                sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
                sample_weight_syn[:] = 1. / X.shape[0]

                # Combine the original and synthetic samples.
                X = np.vstack((X, X_syn))
                y = np.append(y, y_syn)

                # Combine the weights.
                sample_weight = np.append(
                    sample_weight, sample_weight_syn
                ).reshape(-1, 1)
                sample_weight = np.squeeze(
                    normalize(sample_weight, axis=0, norm="l1")
                )

                #X, y, sample_weight = shuffle(
                #    X, y, sample_weight, random_state=random_state
                #)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X,
                y,
                sample_weight,
                random_state,
            )

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self


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
if __name__ == '__main__':

    # Step 1. 导入训练数据
    # 划分少数类和多数类
    path = "imbalance/D0067877" \
           "" \
           "/feature_label.xlsx"
    # Step 1. 导入训练数据
    # 划分少数类和多数类
    data_0 = pd.read_excel(path, sheet_name='major',
                           header=None)  # Read most types of data
    data_1 = pd.read_excel(path, sheet_name='minor',
                           header=None)  # Read minority data

    # X_0为多数类，X_1为少数类
    X_0, Y_0 = data_0.iloc[:, 1:].values, data_0.iloc[:, 0].values
    X_1, Y_1 = data_1.iloc[:, 1:].values, data_1.iloc[:, 0].values
    X_0 = pd.DataFrame(X_0)
    X_1 = pd.DataFrame(X_1)
    Y_0 = pd.DataFrame(Y_0)
    Y_1 = pd.DataFrame(Y_1)

    dataset = np.vstack((X_0, X_1))

    Y = np.vstack((Y_0, Y_1))

    # 把负类的标签全部用-1表示
    for i in range(np.shape(Y)[0]):
        if Y[i] == 0:
            Y[i] = -1

    dataArr, testArr, LabelArr, testLabelArr = train_test_split(dataset, Y, test_size=0.2, random_state=1, stratify=Y)

    std_scaler_1 = StandardScaler()
    std_scaler_2 = StandardScaler()
    std_scaler_1.fit(dataArr)
    dataArr = std_scaler_1.transform(dataArr)
    std_scaler_2.fit(testArr)
    testArr = std_scaler_1.transform(testArr)

    dim = 10

    dataArr_Y = np.concatenate((dataArr, np.mat(LabelArr)), axis=1)
    dataArr_N, dataArr_P = Matrix_division(dataArr_Y)

    # KPCA dimensionality reduction
    KPCA_lc_1 = decomposition.KernelPCA(n_components=dim, kernel='rbf')
    KPCA_lc_1.fit(dataArr)
    dataArr = KPCA_lc_1.transform(dataArr)
    KPCA_lc_2 = decomposition.KernelPCA(n_components=dim, kernel='rbf')
    KPCA_lc_2.fit(testArr)
    testArr = KPCA_lc_2.transform(testArr)
    # test_samples = test_samples - np.shape(train_data_1)[0]
    # LabelArr = np.mat(LabelArr).T
    testLabelArr = np.mat(testLabelArr).T
    P = np.sum(np.mat(testLabelArr).T == 1)
    N = np.sum(np.mat(testLabelArr).T == -1)
    base_clf_sb = LogisticRegression()

    clf2 = SMOTEBoost()
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