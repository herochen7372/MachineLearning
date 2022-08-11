from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import train_test_split #将数据集分开成训练集和测试集
from sklearn import svm # Support vector classifier                     #xgboost

parameter_candidates=[
    {'C':[1,10,100,1000],'kernel':['linear']},   # 线性核
    {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}  # 高斯核
    # C，gamma是超惨，见公式
]

 
# svm.SVC：用于分类的svm，cv：交叉验证调参次数，n_jobs=-1：计算机有多少gpu来跑
clf = GridSearchCV(estimator=svm.SVC(),param_grid=parameter_candidates,cv=5,n_jobs=-1)

# clf.fit(X_train,y_train)


print('最佳模型的分类分数：',clf.best_score_)
print('最佳C：',clf.best_estimator_.C)
print('最佳Kernel：',clf.best_estimator_.kernel)
print('最佳Gamma：',clf.best_estimator_.gamma)

best_model = clf.best_estimator_
# best_model.score(X_test,y_test)
# best_model.predict(X_test)
