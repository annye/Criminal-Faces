

import sys
import visualise as vis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def get_covariance_matrix(x_std):
    mean_vec = np.mean(x_std, axis=0)
    cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0] - 1)
    # print('Covariance matrix \n%s' % cov_mat)
    print('NumPy covariance matrix: \n%s' % np.cov(x_std.T))


def perform_pca(x, y, st):
    """
    df = pd.read_csv('C:\dev\Unison\data\iris.csv')
    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    # drops the empty line at file-end
    df.dropna(how="all", inplace=True)
    df.tail()
    x = df.ix[:, 0:4].values
    y = df.ix[:, 4].values
    # vis.visualise_feature_classes(x, y)
    """
    components = 75
    x_std = StandardScaler().fit_transform(x)
    model = PCA(n_components=components)
    x_reduced = model.fit_transform(x_std)
    
    var_explained = model.explained_variance_ratio_.cumsum()
    print components, 'components explains ', var_explained[-1] * 100, '% variance'
    # get_covariance_matrix(x_std)
    return x_reduced
