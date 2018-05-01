#可视化分析
import pandas as pda
import numpy as npy
import matplotlib.pyplot as plt
f=open("sample_submission.csv")
data=pda.read_csv(f)
# data.values #[行][列]
data=data.T #转置
z1=data.values[3]
for i in range(0,len(z1)):
    if(z1[i]=='是'):
        x1 = data.values[1][i]
        y1 = data.values[2][i]
        plt.scatter(x1,y1,color="green")
    else:
        x1 = data.values[1][i]
        y1 = data.values[2][i]
        plt.scatter(x1, y1,color= "red")
plt.show()