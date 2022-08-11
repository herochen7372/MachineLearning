# PCA用于去燥
'''如果一个主成分的方差远大于噪声的方差，
就应该不太受到噪声影响.如果仅仅使用比较
重要的主成分来重构原来的数据，相当于保留
了重要的信号,而丢弃了噪声，
我们察看手写数字图像去燥的过程,原始的数字图像如下:
'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
import numpy as np 
digits = load_digits() 

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
    subplot_kw={'xticks' :[],'yticks':[]},
    gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),
        cmap='binary', interpolation='nearest',
        clim=(0,16))

plot_digits(digits.data)
plt.show()


# 添加随机的噪声，图像如下：
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)
plt.show() 

# 使用如上带噪声的数据进行PCA降维和重构，保留50%原来的方差：
pca=PCA(0.50).fit(noisy)
print(pca.n_components_)

# 前12个主成分保留了50%的方差，计算主成分，并且重构图像：
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)
plt.show()

# 这个例子也显示pca可以用于特征提取，在训练机器学习模型之前，对数据进行降维，可以去除不重要的特征。





