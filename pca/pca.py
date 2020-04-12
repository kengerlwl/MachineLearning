'''
总结一下PCA的算法步骤：
  设有m条n维数据。
  1）将原始数据按列组成n行m列矩阵X
  2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值
  3）求出协方差矩阵C=1/m*(XX^T)
  4）求出协方差矩阵的特征值及对应的特征向量
  5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P
  6）Y=PX即为降维到k维后的数据
  # http://blog.codinglabs.org/articles/pca-tutorial.html
  # https://zhuanlan.zhihu.com/p/37777074
'''
import numpy as np
# 1. 如果原始数据是按照行排列的：
'''
def PCA(original_X,componens_k):
  # 1. 首先获得原始数据X的均值，如果数据按照行排列，特征按照列排列，则axis=0; 否则axis=1
  norm_X = X - np.mean(original_X,axis=0) # 去均值之后的数据X
  # 2. 计算协方差矩阵，由于散列矩阵和协方差矩阵仅相差一个系数，对特征向量的求解不影响，因此可以不加系数
  scatter_matrix = np.dot(np.transpose(norm_X),norm_X) # 由于这里数据是按照列排布的，所以C = X^T·X
  # 3. 计算协方差矩阵(散列矩阵)的特征值和特征向量
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  # 4. 将各自的各自的特征值和特征向量绑定在一起按照从大到小的顺序排列
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(X.shape[1])]
  eig_pairs.sort(reverse=True)
  # 5. 按照特征值从大到小的排列顺序得到的特征向量，取前K行组合成降维矩阵P
  dim_re_matrix = np.array([ele[1] for ele in eig_pairs[:componens_k]])
  dim_re_data = np.dot(norm_X,np.transpose(dim_re_matrix))
  return dim_re_data
'''

# 2. 如果原始数据是按照列排列的：
def PCA(original_X,componens_k):
  original_X = np.transpose(original_X) # 原始数据是行排列的，这里使用转置将其转化为列排列进行试验
  norm_X = original_X - np.mean(original_X,axis=1,keepdims=True)
  covariance_matrix = (1 / norm_X.shape[1]) * np.dot(norm_X,np.transpose(norm_X))
  eig_val, eig_vec = np.linalg.eig(covariance_matrix)
  eig_pairs = [(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(norm_X.shape[0])]
  eig_pairs.sort(reverse=True)
  dim_re_matrix = np.array([ele[1] for ele in eig_pairs[:componens_k]])
  dim_re_data = np.dot(dim_re_matrix,norm_X)
  return dim_re_data # [[-2.12132034 -0.70710678  0.          2.12132034  0.70710678]]


# 3. 使用sklearn的PCA
'''
from sklearn.decomposition import PCA
import numpy as np
def PCA_(original_X,components_k):
  pca = PCA(n_components=1)
  pca.fit(original_X)
  return pca.transform(original_X)
'''


if __name__ == '__main__':
  X = np.array([[-1, -2], [-1, 0], [0, 0], [2, 1], [0, 1]])
  # X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
  print(PCA(X, 1))

