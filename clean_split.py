'''------------------------------------
Machine Learning for Cities // New York University
Instructor: Professor Luis Gustavo Nonato

Written by: Dror Ayalon
------------------------------------'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
import csv


'''------------------------------------
LOADING THE DATASET THAT INCLUDES THE FOLLOWING COLUMNS:
    [0] = Neighborhood [classes]
    [1] = BldClassif
    [2] = YearBuilt
    [3] = GrossSqFt [independent]
    [4] = GrossIncomeSqFt [independent]
    [5] = MarketValueperSqFt [independent]
------------------------------------'''

dataset = open('manhattan-dof.csv', "r")
dataset = csv.reader(dataset,delimiter=';')
next(dataset) # moving passed the title row

df = pd.read_csv('manhattan-dof.csv', delimiter=';', index_col=False)

x_all = df.ix[:,3:]
y_all = df.ix[:,0]


'''------------------------------------
CLEANING AND FILTERING THE DATA
(to get rid of outliers and unneeded data)
------------------------------------'''
removed_rows = 0

# removing the number of neighborhoods
outlier = np.where(y_all[0:] > 30)
print ("[0] = Neighborhood outlier:")
print (outlier)

if (len(outlier) > 0):
    y_all = y_all.drop(y_all.index[outlier])
    x_all = x_all.drop(x_all.index[outlier])
    removed_rows = removed_rows + len(outlier[0])

# -----------------------------
# cleaning [3] = GrossSqFt
# -----------------------------

# removing all relevant rows for GrossSqFt bigger than 1,500,000
outlier = np.where(x_all['GrossSqFt'] > 1500000)
print ("[3] = GrossSqFt:")
print (outlier)
if (len(outlier) > 0):
    y_all = y_all.drop(y_all.index[outlier])
    x_all = x_all.drop(x_all.index[outlier])
    removed_rows = removed_rows + len(outlier[0])

# -----------------------------
# cleaning [4] = GrossIncomeSqFt
# -----------------------------

# removing all relevant rows for GrossIncomeSqFt bigger than 50, or smaller than 10
outlier = np.where(((x_all['GrossIncomeSqFt'] < 10) | (x_all['GrossIncomeSqFt'] > 50)))
# outlier = np.where(filtered_data[:,4] < 10)
print ("[4] = GrossIncomeSqFt:")
print (outlier)
if (len(outlier) > 0):
    y_all = y_all.drop(y_all.index[outlier])
    x_all = x_all.drop(x_all.index[outlier])
    removed_rows = removed_rows + len(outlier[0])


# -----------------------------
# cleaning [5] = MarketValueperSqFt
# -----------------------------

# removing all relevant rows for MarketValueperSqFt bigger than 50, or smaller than 10

# outlier = np.where((x_all['MarketValueperSqFt' < 40))
# print ("[5] = MarketValueperSqFt:")
# print (outlier)
# if (len(outlier) > 0):
    # y_all = y_all.drop(y_all.index[outlier])
    # x_all = x_all.drop(x_all.index[outlier])
    # removed_rows = removed_rows + len(outlier[0])


'''------------------------------------
PLOTTING THE DATA: BEFORE AND AFTER DATA CLEANING
------------------------------------'''

titles = ['Neighborhood', 'BldClassif', 'YearBuilt', 'GrossSqFt', 'GrossIncomeSqFt', 'MarketValueperSqFt']
df_y = df['Neighborhood'].value_counts()
df_x = df['Neighborhood'].value_counts().index.tolist()
n_y = y_all.value_counts()
n_x = y_all.value_counts().index.tolist()

df_rows, df_columns = df.shape
rows, columns = x_all.shape
df = df.as_matrix()
x_all = x_all.as_matrix()
y_all = y_all.as_matrix()

plt.figure(1)

plt.subplot(2,2,1)
plt.bar(df_x, df_y, color='orange')
plt.bar(n_x, n_y, color='palegreen')
plt.ylabel('Number of Samples', fontsize=8)
plt.title(titles[0], fontsize= 10)

plt.subplot(2,2,2)
plt.plot(df[:,0],df[:,3], 'o', c="orange", markersize=3)
plt.plot(y_all,x_all[:,0], 'o', c="palegreen", markersize=3)
plt.title(titles[3], fontsize= 10)
plt.xlabel('Neighborhood', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(2,2,3)
plt.plot(df[:,0],df[:,4], 'o', c="orange", markersize=3)
plt.plot(y_all,x_all[:,1], 'o', c="palegreen", markersize=3)
plt.title(titles[4], fontsize= 10)
plt.xlabel('Neighborhood', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(2,2,4)
plt.plot(df[:,0],df[:,5], 'o', c="orange", markersize=3)
plt.plot(y_all,x_all[:,2], 'o', c="palegreen", markersize=3)
plt.title(titles[5], fontsize= 10)
plt.xlabel('Neighborhood', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)


'''------------------------------------
NORMALIZTING THE DATA
------------------------------------'''

###############################
# Normalization method A: using 'whiten'
###############################
# print ('before', x_all)
# x_all = whiten(x_all)
# print ('after', x_all)

###############################
# Normalization method B: using max values (0-1)
###############################
for i in range(0,columns):
    x_max = np.amax(x_all[:,i])
    x_all[:,i] = x_all[:,i]/x_max

# y_max = np.amax(y_all)
# y_all = y_all/y_max


'''------------------------------------
GENERATING TRAINING SET AND TEST SET
------------------------------------'''
x_training, x_test, y_training, y_test = train_test_split(x_all, y_all, train_size=0.8, random_state=3)

print ('x_training:')
print (x_training.shape)
print ('y_training:')
print (y_training.shape)

print ('x_test:')
print (x_test.shape)
print ('y_test:')
print (y_test.shape)


'''------------------------------------
SAVING TRAINING AND TEST CSV FILES
------------------------------------'''
np.savetxt("clean_datasets/x_training.csv", x_training, delimiter=';') # saving x_training as csv
np.savetxt("clean_datasets/x_test.csv", x_test, delimiter=';') # saving x_test as csv
np.savetxt("clean_datasets/y_training.csv", y_training, delimiter=';') # saving x_training as csv
np.savetxt("clean_datasets/y_test.csv", y_test, delimiter=';') # saving x_test as csv
print ('#####', removed_rows, 'were removed from the data set')
plt.show()
