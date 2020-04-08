#-*-coding:utf-8 -*-
import sys
sys.path.append("...文件路径...")
import knn
from numpy import *
dataSet,labels = knn.createDataSet()
input = array([1.1,0.3])
K = 3
output = knn.classify(input,dataSet,labels,K)
print("测试数据为:",input,"分类结果为：",output)