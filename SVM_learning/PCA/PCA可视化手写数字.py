# PCA 可视化: 手写数字
'''
将二维数据降维一维数据所带来的便利并不明显，
但是如果是高纬数据，这个优势就显露出来了.
看看手写数字图像降维的例子.
'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
import numpy as np 
digits = load_digits() 
print(digits.data.shape) # (179764)

'''
手写数字是8x8像素的图像，将8行的数据串起来就得到了
一个64维的数据点为了得到不同数据点之间的关系，我们
可以对原来的数据降维为2维，然后进行可视化:

'''

pca =PCA(2) # project from 64 to 2 dimensions projected =pca.fit transform(digits.data) print(digits.data.shape) print(projected.shape)
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)
#将数据最重要的2维可视化:
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5, 
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1') 
plt.ylabel('component 2') 
plt.colorbar()
plt.show()


# 

pca=PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


