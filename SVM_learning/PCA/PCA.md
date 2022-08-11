# PCA
&emsp;&emsp;Principal Component Analysis主成分分析是一种
非监督学习方法(unsupervisedlearning),处理的
数据不包括类别信息(label/target).
主成分分析主要用于降维(dimension reduction)，
也常用于数据可视化(visualization)，或者去噪
(noise filtering),特征提取(feature extraction)等等.




# 主成分的意义?
&emsp;&emsp;可以通过线性代数中的基向量的线性组合来理解例如，训练数据中的每一幅图都会又64个数据点构成的X:
$$x=[x_1,x_2,x_3…X_64]$$
&emsp;&emsp;就是说图像由基像素的线性组合构成，即:
$$image(x)=x_1·(pixel 1)+x_2·(pixel2)+x_3·(pixel 3)…X_64·(pixel_64)$$
一种基像素的方式可以是如下方式:
![图1](image\img1.jpg)

&emsp;&emsp;第一行对应的是像素，第二行对应的是像素的强度，第一行和第二行的对应值相乘，然后再相加就得到了一个图像，我们只使用了8个基像素，所以只重构了图像的一小部分。如果我们继续这个过程，就可以重构原来的图像。


&emsp;&emsp;但是基像素的表达并不是唯一的，我们可以想象其它的表达方式，例如:
$$image(x) = mean+x_1·(basis 1)+x_2·(basis 2)+x_3·(basis 3)…$$
&emsp;&emsp;PCA可以被想象成挑选最佳基向量的过程，使得只需要比较少量的基向量就可以用于重构原来的数据，主成分是原来数据的低维表达，其实只是最佳基向量的对应系数值如下图显示了前8个PCA基向量于对应的系数，他们和平均图一起可以大致还原原来的图像:
 
![图2](image\img2.jpg)
 
&emsp;&emsp;与基像素不同，PCA基向量只需少量的8个主成分就可以重构原来的主要特征.

# 如何选择最佳的主成分数量
&emsp;&emsp;PCA重要的一个部分是如何确定最佳的主成分的数量，用于重构原来的数据
&emsp;&emsp;一个方法是查看累积(explained variance ratio)和主成分数量的关系曲线:
```
pca=PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_)) plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
```
![图3](image\img3.jpg)

&emsp;&emsp;曲线表达了前N个维度描述了64个维度的方差的百分比，例如 前10个主成分表达了75%的方差，前50个主成分描述了几乎100%的方差们前20个主成分保留了90%的方差




