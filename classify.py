'''
Machine Learning for Cities // New York University
Instructor: Professor Luis Gustavo Nonato

Written by: Dror Ayalon
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import csv
import pandas as pd

'''------------------------------------
LOADING THE DATASET THAT INCLUDES THE FOLLOWING COLUMNS:
    [0] = Neighborhood
    [1] = BldClassif [classes]
    [2] = YearBuilt
    [3] = GrossSqFt [independent]
    [4] = GrossIncomeSqFt [independent]
    [5] = MarketValueperSqFt [independent]
------------------------------------'''

df = pd.read_csv('manhattan-dof.csv',index_col=False,delimiter=';')

'''--------------------
cleaning the data
--------------------'''
df_s = df
### cleaning [4] = GrossIncomeSqFt: removing all relevant rows for GrossIncomeSqFt bigger than 50, or smaller than 10
print ('df before trimming [4]: ', df.shape)
df_s = df_s[(df.GrossIncomeSqFt > 10) & (df.GrossIncomeSqFt < 50)]
print ('df_s after trimming [4]: ', df_s.shape)

### cleaning [3] = GrossSqFt (removing all relevant rows for GrossSqFt bigger than 1,500,000)
print ('df_s before trimming [3]: ', df_s.shape)
df_s = df_s[df.GrossSqFt < 1500000]
print ('df_s after trimming [3]: ', df_s.shape)

'''--------------------
splitting the data to x and y
--------------------'''
# indipendent variables
x_all = df_s.ix[:,3:]
print ('indipendent values: ', x_all.shape)

# predicted variable
y_all = df_s['BldClassif']
print ('dipendent values: ', y_all.shape)

'''--------------------
Visualization: data cleaning
--------------------'''
x_all = x_all.as_matrix()
y_all = y_all.as_matrix()
temp_x = df_s['Neighborhood'].as_matrix()

plt.figure(1)

plt.subplot(1,3,1)
plt.scatter(df['Neighborhood'],df['GrossSqFt'], s=10, alpha=0.2, color='magenta')
plt.scatter(temp_x,x_all[:,0], s=10, color='blue', alpha=0.4)
plt.title('GrossSqFt', fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(1,3,2)
plt.scatter(df['Neighborhood'],df['GrossIncomeSqFt'], s=10, alpha=0.2, color='magenta')
plt.scatter(temp_x,x_all[:,1], s=10, color='blue', alpha=0.4)
plt.title('GrossIncomeSqFt', fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(1,3,3)
plt.scatter(df['Neighborhood'],df['MarketValueperSqFt'], s=10, alpha=0.2, color='magenta')
plt.scatter(temp_x,x_all[:,2], s=10, color='blue', alpha=0.4)
plt.title('MarketValueperSqFt', fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)


'''------------------------------------
Visualization
------------------------------------'''

# SVM Kernels
# plt.subplot(2,2,1)
# plt.scatter(x_test[:,2],x_test[:,0],c=prediction_gnb, s=10)
# plt.title('SVM Kernel:\n%s%% Error' %(int(error_svm_rbf/ts*100)), fontsize= 10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
#
# plt.subplot(2,2,2)
# plt.scatter(x_test[:,2],x_test[:,0],c=prediction_svm_linear, s=10)
# plt.title('SVM Linear:\n%s%% Error' %(int(error_svm_linear/ts*100)), fontsize= 10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
#
# plt.subplot(2,2,3)
# plt.scatter(x_test[:,2],x_test[:,0],c=prediction_svm_rbf, s=10)
# plt.title('SVM Kernel:\n%s%% Error' %(int(error_svm_rbf/ts*100)), fontsize= 10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
#
# plt.subplot(2,2,4)
# plt.scatter(x_test[:,2],x_test[:,0],c=prediction_svm_rbf, s=10)
# plt.title('SVM Kernel:\n%s%% Error' %(int(error_svm_rbf/ts*100)), fontsize= 10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
#
#
plt.show()

print ('\n❤️  ❤️  ❤️\n')
