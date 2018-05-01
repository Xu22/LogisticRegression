# -*- coding: utf-8 -*-  
# 对率回归
import numpy as np
from numpy import linalg
import pandas as pd
# 读取数据集
inputfile =open( 'sample_submission.csv')
data_original = pd.read_csv(inputfile)
# 数据的初步转化与操作--属性x变量2行17列数组，并添加一组1作为吸入的偏置x^=（x;1）
x = np.array(
    [list(data_original[u'密度']), list(data_original[u'含糖率']), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# 定义初始参数
beta = np.array([[0], [0], [1]])
old_l = 0
n = 0
while 1:
    beta_T_x = np.dot(beta.T[0], x)
    cur_l = 0
    for i in range(17):
        cur_l = cur_l + (-y[i] * beta_T_x[i] + np.log(1 + np.exp(beta_T_x[i])))
    if np.abs(cur_l - old_l) <= 0.000001:
        break
    # 牛顿迭代法更新β
    n = n + 1
    old_l = cur_l
    dbeta = 0
    d2beta = 0
    for i in range(17):
        dbeta = dbeta - np.dot(np.array([x[:, i]]).T,(y[i] - (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i])))))  # 一阶导数
        d2beta = d2beta + np.dot(np.array([x[:, i]]).T, np.array([x[:, i]]).T.T) * (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i]))) * (1 - (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i]))))
    beta = beta - np.dot(linalg.inv(d2beta), dbeta) #（1）np.linalg.inv()：矩阵求逆（2）np.linalg.det()：矩阵求行列式（标量）
print('模型参数是：%s'%beta)
print('迭代次数：%s'%n)
#可视化分析
import matplotlib.pyplot as plt
data=data_original.T #转置
x1=[]
y1=[]
x2=[]
y2=[]
z1=data.values[3]
for i in range(0,len(z1)):
    if(z1[i]=='是'):
        x1.append(data.values[1][i])
        y1.append(data.values[2][i])
    else:
        x2.append(data.values[1][i])
        y2.append(data.values[2][i])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x1,y1,c='green')
ax.scatter(x2,y2,c='red')
plt.xlim(0,1)
plt.ylim(0,1)
newton_left = -( beta[0]*0.1 + beta[2] )/beta[1]
newton_right = -( beta[0]*0.9 + beta[2] )/beta[1]
plt.plot([0.1, 0.9], [newton_left, newton_right], 'g-')
plt.show()
