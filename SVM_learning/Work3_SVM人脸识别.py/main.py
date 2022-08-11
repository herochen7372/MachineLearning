#基于SVM支持向量机的人脸识别

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60) 
print(faces.target_names) 
print(faces.images.shape)


fig, ax=plt.subplots(3,5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i],cmap='bone') 
    axi.set(xticks=[],yticks=[],
            xlabel=faces.target_names[faces.target[i]])
plt.show()

'''我们可以将整个图像展平为一个长度为3000左右的一维向量，
然后使用这个向量做为特征.通常更有效的方法是通过预处理提取
图像最重要的特征.一个重要的特征提取方法是PCA(主成分分析)，
可以将一副图像转换为一个长度为更短的(150)向量
'''
from sklearn.svm import SVC 
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca =PCA(n_components=150,whiten=True,random_state=42) 
svc=SVC(kernel='linear',class_weight='balanced') 
model=make_pipeline(pca,svc)

# 将数据分为训练和测试数据集
from sklearn.model_selection import train_test_split 
Xtrain, Xtest, ytrain, ytest=train_test_split(faces.data,faces.target,random_state=42)

# 调参:通过交叉验证寻找最佳的C(控制间隔的大小)

from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [1, 5, 10, 50]}
grid =GridSearchCV(model, param_grid)

grid.fit(Xtrain, ytrain)
print(grid.best_params_)


model = grid.best_estimator_
yfit = model.predict(Xtest)

# 使用训练好的SVM做预测

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47),cmap='bone') 
    axi.set(xticks=[1],yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
    color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.show()


# 生成性报告 
from sklearn.metrics import classification_report
print(classification_report(ytest,yfit,
                            target_names=faces.target_names))

# 混淆矩阵
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest,yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt= 'd', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


