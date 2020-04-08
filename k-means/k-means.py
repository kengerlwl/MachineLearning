from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


#新建数据集
def createData():
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[1,1],[2,2]], cluster_std=[0.2,0.3,0.4])
    return X, y


#计算两个向量的距离
def calculateDistance(vecA, vecB):
    return np.sqrt(np.sum(np.square(vecA - vecB)))

#产生随机中心
def randomCenter(samples, K):
    m,n = np.shape(samples)
    centers = np.mat(np.zeros((K, n)))
    for i in range(n):
		    # 通过np.max获取i列最大值
        mxi = np.max(samples[:, i])
		    # 通过np.min获取i列最小值
        mni = np.min(samples[:, i])
        rangeI = mxi - mni
		    # 为簇中心第i列赋值
        centers[:, i] = np.mat(mni + rangeI * np.random.rand(K, 1))
    return centers


#k-means算法
def KMeans(dataset, k):
    m, n = np.shape(dataset)
    # 最后的返回结果，一共两维，第一维是所属类别，第二维是到簇中心的距离
    clusterPos = np.zeros((m, 2))
    
    centers = randomCenter(dataset, k)
    clusterChange = True
    while clusterChange:
        clusterChange = False
        # 遍历所有样本
        for i in range(m):
            minD = 10000000#设置一个最大初始值
            idx = -1
            # 遍历到各个簇中心的距离 找到最近的中心点
            for j in range(k):
                dis = calculateDistance(centers[j,:], dataset[i, :])
                if dis < minD:
                    minD = dis
                    idx = j
            
            # 如果所属类别发生变化
            if clusterPos[i,0] != idx:
                clusterChange = True
            
            # 更新样本聚类结果
            #第二维两个数字就是代表第几类，以及距离
            clusterPos[i,:] = idx, minD
        # 更新簇中心的坐标
        for i in range(k):
            nxtClust = dataset[np.nonzero(clusterPos[:,0] == i)[0]]#找到该类的点
            centers[i,:] = np.mean(nxtClust, axis=0)#求平均值，并且更新中心
    return centers, clusterPos



if __name__=="__main__":

    x, y = createData()

    centers, clusterRet = KMeans(x, 3)
    plt.scatter(x[:,0],x[:,1],c=clusterRet[:,0] ,s=3,marker='o')
    plt.scatter(centers[:, 0].A, centers[:, 1].A, c='red', s=100, marker='x')
    plt.show()

