import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

sns.palplot(sns.color_palette("hls",8))
plt.show()

# 降维演示
'''将二维数据降为一维数据，考虑如下200个数据点：'''
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2,2),rng.randn(2, 200)).T
plt.scatter(X[:,0], X[:,1])
plt.axis('equal')
plt.show()

'''
肉眼可见数据的x与y是线性相关的，PCA试图找到数据的
主坐标轴(principalaxes)，将数据从原来的形式转换为
使用主坐标轴来表达,使用Scikit-Learn's PCA 类，
我们可以如下计算:
'''
from sklearn.decomposition import PCA 
pca=PCA(n_components=2) 
pca.fit(X)

# PCA找到了数据的坐标轴("components")以及可解释的方差( "explained variance"):
print(pca.components_) 
print(pca.explained_variance_) 
'''
我们使用可视化的方式来解释“components”和
“explained variance”的意义“compontents”
为两个向量的方向，“explained variance”为
向量的长度的平方:
'''


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
    linewidth=2,
    shrinkA=0,shrinkB=0, color='green')
    ax.annotate('', v1, v0,arrowprops=arrowprops)

# plot data
plt.scatter(X[:,0],X[:,1],alpha=0.2, color='blue')
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_,pca.mean_ + v) 
plt.axis('equal')
plt.show()

'''
两个向量分别代表了数据的两个主坐标轴(principalaxes)，
向量的长度代表了此坐标轴对于描述数据分布的重要性，
长度带表了数据投影到这个轴上面以后的方差.
将每一个数据点投影到主坐标轴(principalaxes)以后得到
的值为数据的主成分("principalcomponents)如果我们将
以下左图的数据的主成分画出来，为以下右图所示(PCA\新轴投影.jpg):
'''

'''
将数据从原来的轴转换到新的轴的过程是一个仿射变换(affine transformation)，意味着这个转换是由平移，旋转，和均匀缩放完成的.
PCA在机器学习和数据可视化上面的用途非常广泛
'''

# 使用PCA进行降维
'''
使用PCA降维就是将比较不重要的主成分删除的操作，
保留下来的低维度的数据最大程度的保留了原来数据的方差。
例如
'''
pca=PCA(n_components=1) 
pca.fit(X)
X_pca=pca.transform(X)
print("original shape: ", X.shape) 
print("transformed shape:",X_pca.shape)

# 将原来的2维数据降维一维，然后我们可以把一维的数据反转为二维，察看其效果:
X_new=pca.inverse_transform(X_pca)
plt.scatter(X[:,0],X[:,1],alpha=0.2)
plt.scatter(X_new[:,0],X_new[:,1],alpha=0.8) 
plt.axis('equal')
plt.show()

'''
浅蓝色的数据点为原来的数据，橙色的点为降维后又转为
原来2维的数据点，我们看到所谓PCA降维就是将不重要的
坐标轴(principle axis)上面的信息删除，只保留具有
较大的方差的维度上面的信息。
被删除的信息对应的方差表达了降维造成的信息丢失。
被降维的数据在某种程度上而言已经包含了原来数据点
最重要的关系信息:尽管在2维降维1维的情况下50%的信息
被丢弃了，数据点之间的主要关系被保留了.
'''


