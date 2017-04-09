'''
Machine Learning for Cities // New York University
Instructor: Professor Luis Gustavo Nonato

Written by: Dror Ayalon
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
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
# print ('df before trimming [4]: ', df.shape)
df_s = df_s[(df.GrossIncomeSqFt > 10) & (df.GrossIncomeSqFt < 50)]
# print ('df_s after trimming [4]: ', df_s.shape)

### cleaning [3] = GrossSqFt (removing all relevant rows for GrossSqFt bigger than 1,500,000)
# print ('df_s before trimming [3]: ', df_s.shape)
df_s = df_s[df.GrossSqFt < 1500000]
# print ('df_s after trimming [3]: ', df_s.shape)

# cleaning [5] = MarketValueperSqFt
df_s = df_s[df.MarketValueperSqFt < 200]


# cleaning [1] = BldClassif [classes]
# print ('df_s before trimming [1]: ', df_s.shape)
# df_s = df_s[(df.BldClassif > 0) & (df.BldClassif < 3)]
df_s = df_s[df.BldClassif > 0]
# df_s = df_s[df.BldClassif < 3]
# print ('df_s after trimming [1]: ', df_s.shape)



'''--------------------
splitting the data to x and y
--------------------'''
# indipendent variables
x_all = df_s.ix[:,3:]
# print ('indipendent values: ', x_all.shape)

# predicted variable
y_all = df_s['BldClassif']
# print ('dipendent values: ', y_all.shape)



'''--------------------
Visualization: data cleaning
--------------------'''
x_all_mat = x_all.as_matrix()
y_all_mat = y_all.as_matrix()
temp_x = df_s['Neighborhood'].as_matrix()

plt.figure(1).set_size_inches(12,8)

plt.subplot(1,3,1)
plt.scatter(df['Neighborhood'],df['GrossSqFt'], s=10, alpha=0.2, color='magenta')
plt.scatter(temp_x,x_all_mat[:,0], s=10, color='blue', alpha=0.4)
plt.title('GrossSqFt', fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(1,3,2)
plt.scatter(df['Neighborhood'],df['GrossIncomeSqFt'], s=10, alpha=0.2, color='magenta')
plt.scatter(temp_x,x_all_mat[:,1], s=10, color='blue', alpha=0.4)
plt.title('GrossIncomeSqFt', fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(1,3,3)
plt.scatter(df['Neighborhood'],df['MarketValueperSqFt'], s=10, alpha=0.2, color='magenta')
plt.scatter(temp_x,x_all_mat[:,2], s=10, color='blue', alpha=0.4)
plt.title('MarketValueperSqFt', fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.savefig('plots/01.png', dpi=300)


'''------------------------------------
Normalizing the data
------------------------------------'''
rows, columns = x_all.shape
for i in range(0,columns):
    # print (i)
    # print (x_all[[i]].max()[0])
    x_max = x_all[[i]].max()[0]
    x_all.iloc[:,i] = x_all.iloc[:,i]/x_max


'''------------------------------------
Splitting the data into training set and validation set
------------------------------------'''
x_all = x_all.as_matrix()
y_all = y_all.as_matrix()
x_training, x_test, y_training, y_test = train_test_split(x_all, y_all, train_size=0.8, random_state=3)
ts, = y_test.shape

# print ('x_training:')
# print (x_training.shape)
# print ('y_training:')
# print (y_training.shape)
#
# print ('x_test:')
# print (x_test.shape)
# print ('y_test:')
# print (y_test.shape)


'''------------------------------------
SVM Kernel
------------------------------------'''
svm_rbf = svm.SVC(kernel='rbf', gamma=500)
# svm_rbf = svm.SVC(kernel='rbf', gamma=100)
# svm_rbf = svm.SVC(kernel='rbf', gamma=500, max_iter=-1, probability=False, random_state=None, shrinking=True,)
svm_rbf.fit(x_training,y_training)
prediction_svm_rbf = svm_rbf.predict(x_test)
error_svm_rbf = np.sum((prediction_svm_rbf[i] != y_test[i]) for i in range(0,ts))

print("\n👉  ----------SVM Kernels----------")
print("number of support vectors",len(svm_rbf.support_))
print(error_svm_rbf, "misclassified data out of", ts, "(",error_svm_rbf/ts,"%)\n")


'''--------------------
CART (Decision Tree)
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
--------------------'''
# Train
dtc = DecisionTreeClassifier(max_depth=8)
dtc.fit(x_training, y_training)

# Predicting
ypred = dtc.predict(x_test)

# Computint prediction error
cart_error = np.mean((ypred - y_test) ** 2)

cart_verror = np.asarray([int(ypred[i] != y_test[i]) for i in range(0,ts)])
cart_error = np.sum(cart_verror)

print("🌲  ----------Decision Tree Classfication----------")
print(cart_error, "misclassified data out of", ts, "(",cart_error/ts,"%)\n")
# print ("")


'''--------------------
CART (Decision Tree) + Bagging
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
--------------------'''
bagb = BaggingClassifier(dtc, n_estimators=30, bootstrap_features=True, bootstrap=True)
# adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=20,learning_rate=1.5,algorithm="SAMME")
bagb.fit(x_training,y_training)

# Predicting
bagb_pred = bagb.predict(x_test)

# Finding mispredicted samples
bagb_verror = np.asarray([int(bagb_pred[i] != y_test[i]) for i in range(0,ts)])
bagb_error = np.sum(bagb_verror)
bagb_ccidx = np.where(bagb_verror == 0)
bagb_mcidx = np.where(bagb_verror == 1)

print("🌲  ----------Decision Tree Classfication + Bagging----------")
print(bagb_error, "misclassified data out of", ts, "(",bagb_error/ts,"%)\n")


'''--------------------
CART (Decision Tree) + Boosting
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
--------------------'''
# adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=20)
# adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=20,learning_rate=1.5,algorithm="SAMME") #[dodiku] there are many more parameters we can play with
adab = AdaBoostClassifier(dtc, n_estimators=20,learning_rate=1.5,algorithm="SAMME.R")
# adab = GradientBoostingClassifier(max_depth=5, n_estimators=30)

adab.fit(x_training,y_training)

# Predicting
adab_pred = adab.predict(x_test)

# Finding mispredicted samples
adab_verror = np.asarray([int(adab_pred[i] != y_test[i]) for i in range(0,ts)])
adab_error = np.sum(adab_verror)
adab_ccidx = np.where(adab_verror == 0)
adab_mcidx = np.where(adab_verror == 1)

print("🌲  ----------Decision Tree Classfication + Boosting----------")
print(adab_error, "misclassified data out of", ts, "(",adab_error/ts,"%)\n")


'''--------------------
Random Trees
--------------------'''
rdf = RandomForestClassifier(max_depth=6, n_estimators=35, bootstrap=True)
rdf.fit(x_all,y_all)

# Predicting
rdf_pred = rdf.predict(x_test)

# Finding mispredicted samples
rdf_verror = np.asarray([int(rdf_pred[i] != y_test[i]) for i in range(0,ts)])
rdf_error = np.sum(rdf_verror)
rdf_ccidx = np.where(rdf_verror == 0)
rdf_mcidx = np.where(rdf_verror == 1)

print("🎲  ----------Random Forest Classfication----------")
print(rdf_error, "misclassified data out of", ts, "(",rdf_error/ts,"%)\n")

'''--------------------
Random Trees + Bagging
--------------------'''
rf_bagb = BaggingClassifier(rdf, n_estimators=45, bootstrap_features=False, bootstrap=True)
rf_bagb.fit(x_training,y_training)

# Predicting
rf_bagb_pred = rf_bagb.predict(x_test)

# Finding mispredicted samples
rf_bagb_verror = np.asarray([int(rf_bagb_pred[i] != y_test[i]) for i in range(0,ts)])
rf_bagb_error = np.sum(rf_bagb_verror)
rf_bagb_ccidx = np.where(rf_bagb_verror == 0)
rf_bagb_mcidx = np.where(rf_bagb_verror == 1)

print("🎲  ----------Random Trees Classfication + Bagging----------")
print(rf_bagb_error, "misclassified data out of", ts, "(",rf_bagb_error/ts,"%)\n")

'''--------------------
Random Trees + Boosting
--------------------'''
# rf_adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=20)
# rf_adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=20,learning_rate=1.5,algorithm="SAMME") #[dodiku] there are many more parameters we can play with
rf_adab = AdaBoostClassifier(rdf, n_estimators=8,learning_rate=1.5,algorithm="SAMME.R")
# rf_adab = GradientBoostingClassifier(max_depth=5, n_estimators=30)

rf_adab.fit(x_training,y_training)

# Predicting
rf_adab_pred = rf_adab.predict(x_test)

# Finding mispredicted samples
rf_adab_verror = np.asarray([int(rf_adab_pred[i] != y_test[i]) for i in range(0,ts)])
rf_adab_error = np.sum(rf_adab_verror)
rf_adab_ccidx = np.where(rf_adab_verror == 0)
rf_adab_mcidx = np.where(rf_adab_verror == 1)

print("🎲  ----------Random Trees Classfication + Boosting----------")
print(rf_adab_error, "misclassified data out of", ts, "(",rf_adab_error/ts,"%)\n")

'''------------------------------------
Visualization
------------------------------------'''

plt.figure(2).set_size_inches(12,8)
plt.subplots_adjust(hspace = 0.6)
ts = y_test.shape[0]

# SVM Kernels
plt.subplot(3,3,1)
plt.scatter(x_test[:,2],x_test[:,0],c=prediction_svm_rbf, s=10, alpha=0.6)
plt.title('SVM Kernel:\n%s%% Error' %(int(error_svm_rbf/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# CART
plt.subplot(3,3,4)
plt.scatter(x_test[:,2],x_test[:,0],c=ypred, s=10, alpha=0.6)
plt.title('CART:\n%s%% Error' %(int(cart_error/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# CART + Bagging
plt.subplot(3,3,5)
plt.scatter(x_test[:,2],x_test[:,0],c=bagb_pred, s=10, alpha=0.6)
plt.title('CART + Bagging:\n%s%% Error' %(int(bagb_error/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# CART + Boosting
plt.subplot(3,3,6)
plt.scatter(x_test[:,2],x_test[:,0],c=adab_pred, s=10, alpha=0.6)
plt.title('CART + Boosting:\n%s%% Error' %(int(adab_error/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)


# Random Forest
plt.subplot(3,3,7)
plt.scatter(x_test[:,2],x_test[:,0],c=rdf_pred, s=10, alpha=0.6)
plt.title('Random Forest:\n%s%% Error' %(int(rdf_error/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Random Forest + Bagging
plt.subplot(3,3,8)
plt.scatter(x_test[:,2],x_test[:,0],c=rf_bagb_pred, s=10, alpha=0.6)
plt.title('Random Forest + Bagging:\n%s%% Error' %(int(rf_bagb_error/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Random Forest + Boosting
plt.subplot(3,3,9)
plt.scatter(x_test[:,2],x_test[:,0],c=rf_adab_pred, s=10, alpha=0.6)
plt.title('Random Forest + Boosting:\n%s%% Error' %(int(rf_adab_error/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)


plt.savefig('plots/02.png', dpi=300)
plt.show()

print ('\n\n~~~', '\nall good 🥑\n')
