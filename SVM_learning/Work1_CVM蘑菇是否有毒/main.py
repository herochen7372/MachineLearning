import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pandas as pd
import numpy as np

# 导入数据
# mush_df = pd.read_csv(os.path.join(cwd, 'Work1_CVM蘑菇是否有毒\data\image\mushrooms.csv'))
mush_df = pd.read_csv(os.path.join(cwd, 'Work1_CVM蘑菇是否有毒\data\mushrooms.csv'))
# print(mush_df.head()) # 打印前五行 [5 rows x 23 columns]

# get_dummies将值从字母转换为热编码
mush_df_encoded = pd.get_dummies(mush_df)
# print(mush_df_encoded.head()) # 打印前五行 [5 rows x 119 columns] #增加的列是每一类的某种可能，eg.class ---> class_e  class_p

# 将特征和类别标签分别赋值给x，y
X_mush = mush_df_encoded.iloc[:,2:]
y_mush = mush_df_encoded.iloc[:,1]  # 两列有一列就可

# 训练SVM
## 建立pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA  # 降维
from sklearn.pipeline import make_pipeline
# n_components降维到的维数，whiten：使得每个特征具有相同的方差。
pca = PCA(n_components=15,whiten=True,random_state=42)
# kernel先尝试使用线性分类器，
svc=SVC(kernel='linear', class_weight='balanced')
# 用make_pipeline创建管道
model = make_pipeline(pca, svc)   #先降维，在送入svc


## 将数据分为训练数据和测试数据
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_mush,y_mush,random_state=41)


## 调参： 通过交叉验证寻找最佳的C(控制松弛度，C越大，越不容易松弛)
'''因为我们选择的是线性分类器，所以只有一个svc_C参数要调
C：控制我们允许不允许我们使用松弛向量(防止线性不可分，允许其犯错)
由于C未知，所以我们可以使用GridSearchCV网格搜索的方式，从预定值依依去试
'''
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [1, 5, 10, 50]}  # 如果不使用pipeline参数key不要scv__
grid= GridSearchCV(model, param_grid)  # CV:指的是交叉验证找最优参数

grid.fit(Xtrain, ytrain)
print(grid.best_params_)


## 使用训练好的SVM做预测
model=grid.best_estimator_
yfit = model.predict(Xtest)

## 生成性能报告
from sklearn.metrics import classification_report
# target_names:对不同类别起名字
'''precision(准确率):正确匹配/总的数量，'''
print(classification_report(ytest,yfit,target_names=['e','p']))



# 使用Kernel SVM
'''
注：这里model没有用pipeline拼接pca与svc，
看更强大的kernel函数，不使用pca降维，直接用SVC，
看能达到多好的效果。
需要调的参数就是kernel的类型
'''
tuned_parameters = [{'kernel':['rbf'],'gamma':[1e-3,1e-4], # 高斯核 有自己的超参数gamma：控制正态分布形状
                    'C':[1,10,100,1000]},
                   {'kernel':['linear'],'C':[1,10,100,1000]}]
scores=['precision', 'recall']
for score in scores:
    print('# Tuning hyper-parameters for %s' % score)
    print()
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' %score)
    clf.fit(Xtrain,ytrain)
    
    print('Best parameters set found on development set:')
    print()
    print(clf.best_params_)
    print('Grid sores on development set:')
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r'
              % (mean, std*2, params))
    print()
    
    print('Detailed classification report:')
    print()
    print('The model is trained on the full development set.')
    print('The scores are computed on the full evalution set.')
    print()
    y_true, y_pred = ytest, clf.predict(Xtest)
    print(classification_report(y_true,y_pred))
    print()
    












