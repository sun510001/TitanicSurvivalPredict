# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

train = pd.read_csv('../../data/titanic/train_proc.csv')
# tit1 = train.select_dtypes(include=['float64', 'int64', 'object'])
# train.info()

test = pd.read_csv('../../data/titanic/test_proc.csv')
# tit2 = test.select_dtypes(include=['float64', 'int64', 'object'])
# test.info()
#
# print("train shape:", train.shape)
# print("test shape :", test.shape)

print(train.isnull().sum())

sns.set(rc={'figure.figsize': (20, 20)})
s = sns.heatmap(train.corr(), annot=True)
s.get_figure().savefig('test_2.png', bbox_inches='tight')

# s = train[train['Pclass'] == 1]['Embarked'].value_counts()
# c = train[train['Pclass'] == 2]['Embarked'].value_counts()
# q = train[train['Pclass'] == 3]['Embarked'].value_counts()
# df = pd.DataFrame([s, c, q])
# df.index = ['S', 'C', 'Q']
# df.plot(kind='bar', stacked=True, figsize=(10, 5))
# plt.savefig('pclass_embarked_ral.png')
# print("Pclass1:\n", s)
# print("Pclass2:\n", c)
# print("Pclass3:\n", q)