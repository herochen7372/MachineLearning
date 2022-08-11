# 基于支持向量回归(Support Vector Regression)的股票价格预测
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt 
import pandas as pd


# 导入数据
'''本实例使用的股票来自美国股市,使用stockai获取股票交易历史
如果希望下载其它股票价格，
需安装 stockai(https://pypi.org/project/stockai/)包，
安装方法: pip install stockai
可以使用如下代码下载TD.To的股票信息 
from stockai import Stock'''
'''
# 用于下载读取数据
from stockai import Stock
td=Stock('TD.TO')
prices_list =td.get_historical_prices('2019-01-01','2019-01-30')
df =pd.DataFrame.from_dict(prices_list)
df.to_csv(path_or_buf='TD.TO.csv',index=False)
'''

df = pd.read_csv(os.path.join(cwd, 'Work2_SVR股票价格预测\data\TD.TO.csv'))
print(df.head())

# 帮助函数，返回两个值，第一个为日期(只包含日，不包含年和月)，第二个为当日收盘价
def get_data(df):
    data = df.copy()
    data['date'] = data['date'].str.split('-').str[2]
    data['date'] = pd.to_numeric(data['date'])
    return [ data['date'].tolist(), data['close'].tolist() ] # Convert Series to list

dates, prices = get_data(df)
print(dates, prices)

# 使用三种不同的核函数来训练SVR，比较他们的性能
def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates),1)) # 转换为 n x 1 维
    x = np.reshape(x,(len(x),1))
    
    svr_lin = SVR(kernel='linear',C=1e3)   # 线性核
    svr_poly = SVR(kernel='poly',C=1e3, degree=2)  # 多项式核
    svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.1)  # 高斯核
    
    # 训练
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    # 画图
    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates,svr_lin.predict(dates),c='g',label='Linear model')
    plt.plot(dates,svr_poly.predict(dates),c='b',label='Polynomial model')
    plt.plot(dates,svr_rbf.predict(dates),c='r',label='RBF model')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0], 
    

predict_prices = predict_prices(dates, prices, [31])

print(predict_prices)



    
    
    
    







