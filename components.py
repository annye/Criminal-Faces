import sys
import visualise as vis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import scale 




def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)




def get_covariance_matrix(x_std):
    mean_vec = np.mean(x_std, axis=0)
    cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0] - 1)
    print('Covariance matrix \n%s' % cov_mat)
    print('NumPy covariance matrix: \n%s' % np.cov(x_std.T))



# Choosing number of Principal Components ,
def number_pcs(x):
    max_pc = 75 
    pcs = [] 
    totexp_var = [] \


    for i in range(max_pcs): 
        x_std = StandardScaler().fit_transform(x)
        model = PCA(n_components=i+1)
        x_reduced = model.fit_transform(x_std)
        var_explained = model.explained_variance_ratio_.cumsum()
        pcs.append(i+1) 
        var_explained.append(tot_var) 
        plt.plot(pcs,totexp_var,'r') 
        plt.plot(pcs,totexp_var,'bs') 
        plt.xlabel('No. of Principal Components',fontsize = 13) 
        plt.ylabel('Total variance explained',fontsize = 13) 
 
        plt.xticks(pcs,fontsize=13) 
        plt.yticks(fontsize=13) 
        plt.show() 





 
 




