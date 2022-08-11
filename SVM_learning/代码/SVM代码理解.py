# SVM支持向量机
'''此教程分为两个部分:
。第一个部分旨在使用可视化的方式让同学们理解SVM的工作原理，SVM分割线，SVM的支持向量并且使用实例证明SVM的分
割线只由支持向量唯一确定，与线性回归/逻辑回归不一样，SVM对异常数据具有较强的鲁棒性
。第二个部分展示了如何使用SVM检测蘑菇是否有毒.对输入的蘑菇的特征，使用PCA(主成分分析)将特征(一维向量)进行了
降维处理，然后将降维后的向量作为支持向量机的输入.PCA降维的目的可以看作是特征提取，将特征里面真正对分类有决定性影响的数据提取出来.
'''
# 一.理解SVM
from pyexpat import model
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # 统计分析的库

# use seaborn plotting defaults
import seaborn as sns;sns.set()

# from sklearn.datasets.samples_generator import make_blobs # ModuleNotFoundError: No module named 'sklearn.datasets.samples_generator'
from sklearn.datasets import make_blobs  #制作产生分类的数据
# 
X,y = make_blobs(n_samples=50,       # 50个数据点(默认2维)
                 centers=2,          # 两个中心点(两类)
                 random_state=0,     # 设一个随机数种子(保证结果一样)
                 cluster_std=0.60)   # 控制类别有多分散
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plt.show()

# 可行的分割线
xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')

for m, b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit,m*xfit+b,'-k')

plt.show()


# SVM:设想每一条分割线是有宽度的
'''在SVM的框架下，认为最宽的线为最优的分割线'''

xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')

for m, b, d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    yfit = m*xfit+b
    plt.plot(xfit,yfit,'-k')
    plt.fill_between(xfit, yfit-d, yfit+d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1,3.5)
plt.show()


# 训练SVM
'''
因为我们知道它是线性可分的所以C我们设一个比较大的数，
防止它穿过数据点，C小时允许一些小错误
'''
from sklearn.svm import SVC # Support vector classifier
# kernel 边界线的种类
model = SVC(kernel='linear',C=1E10) # 因为是线性可分的数据集
model.fit(X,y)


# 创建一个显示SVM分割线的函数
def plot_svc_decision_function(model, ax=None, plot_support=True):
    '''Plot the decision function for 2D SVC'''
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1,0,1],alpha=0.5,
               linestyles=['--','-','--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                   model.support_vectors_[:,1],
                   s=300,linewidth=1,facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
# 显示分割线
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(model)
plt.show()



# 显示支持向量
print(model.support_vectors_)

# 演示非支持向量的数据，对分割没有影响
'''只要支持向量会影响分割线，如果我们添加一些非支持向量的数据，对分割没有影响'''
def plot_svm(N=10,ax=None):
    #random_state=0一样所以之前的点还在只是新增150个点
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0,cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear',C=1E10)
    model.fit(X,y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,6)
    plot_svc_decision_function(model,ax)

fig, ax = plt.subplots(1,2,figsize=(16,6))
fig.subplots_adjust(left=0.0625,right=0.95,wspace=0.1)
for axi, N in zip(ax,[50, 200]):
    plot_svm(N, axi)
    axi.set_title('N={0}'.format(N))
plt.show()







