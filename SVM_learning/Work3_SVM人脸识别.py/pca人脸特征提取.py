'''
展示了如何使用支持向量机实现人脸的分类识别，
对输入的人脸图像，使用PCA(主成分分析)将图像
进行了降维处理，然后将降维后的向量作为支持
向量机的输入。PCA降维的目的可以看作是特征提取，
将图像里面真正对分类有决定性影响的数据提取出来.
####机器学习对数据很敏感，需要高质量数据####
'''

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60) 
print(faces.target_names) 
print(faces.images.shape)

# 我们尝试选择150个主成分：
from sklearn.decomposition import PCA
pca = PCA(150)
pca.fit(faces.data) # 训练

# 可视化比较重要的主成分(也被称为"eigenvectors"，特征脸"eigenfaces"):
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                        subplot_kw={'xticks':[],'yticks':[]}, 
                        gridspec_kw=dict(hspace=0.1,wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47),cmap='bone')
plt.show()
# 前面的特征脸对应了光照后面的特征量对应了某些人脸特征，例如研究，鼻子，嘴唇等等。
# 我们看看前多少个特征对应了多少的重要性:
plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# 前150个主成分包含了90%的方差，我们可以比较使用前150个主成分重构的图像和原来图像的区别:
# Compute the components and projected faces 
pca = PCA(150).fit(faces.data)
components =pca.transform(faces.data)
projected = pca.inverse_transform(components)

# plot the results
fig, ax=plt.subplots(2, 10, figsize=(10, 2.5),
                    subplot_kw={'xticks':[],'yticks':[]}, 
                    gridspec_kw=dict(hspace=0.1,wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62,47),cmap='binary_r') 
    ax[1, i].imshow(projected[i].reshape(62,47),cmap='binary_r')
    
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction')
plt.show()

# 第一行对应了原来的图像，第二行对应了150个主成分构成的图像，尽管150个主成分相当于降维到原来的
# 的二十分之一(原来的维度为3000左右)，图像的信息却比较完整的保留了。











