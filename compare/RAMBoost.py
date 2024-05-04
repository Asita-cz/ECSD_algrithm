#作   者：Asita
#开发时间：2022/4/28 16:07
import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.decomposition import KernelPCA
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_random_state, check_array
from sklearn.utils import check_X_y
#from sklearn.utils import shuffle
import warnings
import time
start_time = time.time()
warnings.filterwarnings("ignore")

class RankedMinorityOversampler(object):
    """Implementation of Ranked Minority Oversampling (RAMO).

    Oversample the minority class by picking samples according to a specified
    sampling distribution.

    Parameters
    ----------
    k_neighbors_1 : int, optional (default=5)
        Number of nearest neighbors used to adjust the sampling probability of
        the minority examples.

    k_neighbors_2 : int, optional (default=5)
        Number of nearest neighbors used to generate the synthetic data
        instances.

    alpha : float, optional (default=0.3)
        Scaling coefficient.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(
        self,
        k_neighbors_1=5,
        k_neighbors_2=5,
        alpha=0.3,
        random_state=None,
    ):
        self.k_neighbors_1 = k_neighbors_1
        self.k_neighbors_2 = k_neighbors_2
        self.alpha = alpha
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
            # Choose a sample according to the sampling distribution, r.
            j = np.random.choice(self.n_minority_samples, p=self.r)

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh_2.kneighbors(
                self.X_min[j].reshape(1, -1), return_distance=False
            )[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X_min[nn_index] - self.X_min[j]
            gap = np.random.random()

            S[i, :] = self.X_min[j, :] + gap * dif[:]

        return S

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_total_samples, n_features]
            Holds the majority and minority samples.

        y : array-like, shape = [n_total_samples]
            Holds the class targets for samples.

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights multiplier. If None, the multiplier is 1.

        minority_target : int, optional (default=None)
            Minority class label.
        """
        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        self.X_min = X[y == self.minority_target]
        self.n_minority_samples, self.n_features = self.X_min.shape

        neigh_1 = NearestNeighbors(n_neighbors=self.k_neighbors_1 + 1)
        neigh_1.fit(X)
        nn = neigh_1.kneighbors(self.X_min, return_distance=False)[:, 1:]

        if sample_weight is None:
            sample_weight_min = np.ones(shape=(len(self.minority_target)))
        else:
            assert(len(y) == len(sample_weight))
            sample_weight_min = sample_weight[y == self.minority_target]

        self.r = np.zeros(shape=(self.n_minority_samples))
        for i in range(self.n_minority_samples):
            majority_neighbors = 0
            for n in nn[i]:
                if y[n] != self.minority_target:
                    majority_neighbors += 1

            self.r[i] = 1. / (1 + np.exp(-self.alpha * majority_neighbors))

        self.r = (self.r * sample_weight_min).reshape(1, -1)
        self.r = np.squeeze(normalize(self.r, axis=1, norm="l1"))

        # Learn nearest neighbors.
        self.neigh_2 = NearestNeighbors(n_neighbors=self.k_neighbors_2 + 1)
        self.neigh_2.fit(self.X_min)

        return self


class RAMOBoost(AdaBoostClassifier):
    """Implementation of RAMOBoost.

    RAMOBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class according to a specified sampling
    distribution on each boosting iteration [1].

    This implementation inherits methods from the scikit-learn
    AdaBoostClassifier class, only modifying the `fit` method.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.

    k_neighbors_1 : int, optional (default=5)
        Number of nearest neighbors used to adjust the sampling probability of
        the minority examples.

    k_neighbors_2 : int, optional (default=5)
        Number of nearest neighbors used to generate the synthetic data
        instances.

    alpha : float, optional (default=0.3)
        Scaling coefficient.

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
    .. [1] S. Chen, H. He, and E. A. Garcia. "RAMOBoost: Ranked Minority
           Oversampling in Boosting". IEEE Transactions on Neural Networks,
           2010.
    """

    def __init__(
        self,
        n_samples=100,
        k_neighbors_1=5,
        k_neighbors_2=5,
        alpha=0.3,
        base_estimator=None,
        n_estimators=50,
        learning_rate=1.,
        algorithm="SAMME.R",
        random_state=None,
    ):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.ramo = RankedMinorityOversampler(
            k_neighbors_1, k_neighbors_2, alpha, random_state=random_state
        )

        super(RAMOBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.

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
            # RAMO step.
            self.ramo.fit(X, y, sample_weight=sample_weight)
            X_syn = self.ramo.sample(self.n_samples)
            y_syn = np.full(
                X_syn.shape[0], fill_value=self.minority_target, dtype=np.int64
            )

            # Combine the minority and majority class samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the weights.
            sample_weight = np.append(
                sample_weight, sample_weight_syn
            ).reshape(-1, 1)
            sample_weight = np.squeeze(
                normalize(sample_weight, axis=0, norm="l1")
            )

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

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

dim = 20
path = "datasets/D003015" \
       "" \
       "/feature_label.xlsx"
dataArr, testArr, LabelArr, testLabelArr = load_data(path, dim)

# LabelArr = np.mat(LabelArr).T
testLabelArr = np.mat(testLabelArr).T
P = np.sum(np.mat(testLabelArr).T == 1)
N = np.sum(np.mat(testLabelArr).T == -1)

clf2 = RAMOBoost(n_estimators=10,n_samples=10,k_neighbors_1=2)
clf2.fit(dataArr, LabelArr)

predicts = clf2.predict(testArr)

predicts = np.mat(predicts).T
# print(predicts)
testLabelArr = testLabelArr.T
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
end_time = time.time()
execution_time = end_time - start_time
print("time:\n")
print("execution_time", execution_time)