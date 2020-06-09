# -*- coding:utf-8 -*-

import pandas as pd
import re
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


def preprocess(df):
    # df['Last_Name'] = df['Name'].apply(lambda df: str.split(df, ",")[0])
    df['Cabin'] = df['Cabin'].fillna('nodata')
    df['cabin_p1'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    category = {'nodata': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'X': 8, 'T': 9}
    df['cabin_p1'] = df['cabin_p1'].map(category)

    # has cabin
    def has_cabin(input):
        if input == 0:
            return 0
        else:
            return 1

    df['has_cabin'] = df['cabin_p1'].map(has_cabin)

    # process family survival data
    df = carbin_pred(df)

    df['Sex'] = df['Sex'].fillna('nodata')
    category = {'male': 0, 'female': 1}
    df['Sex'] = df['Sex'].map(category)

    df['Title'] = df['Name'].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())
    category = {'Mr.': 1, 'Mrs.': 2, 'Miss.': 3, 'Master.': 4, 'Don.': 5, 'Rev.': 6, 'Dr.': 7, 'Mme.': 8, 'Ms.': 9,
                'Major.': 10, 'Lady.': 11, 'Sir.': 12, 'Mlle.': 13, 'Col.': 14, 'Capt.': 15, 'Countess.': 16,
                'Jonkheer.': 17}
    df['Title'] = df['Title'].map(category)
    df['Title'] = df['Title'].fillna(0)

    df['Embarked'] = df['Embarked'].fillna('nodata')
    category = {'nodata': 1, 'S': 1, 'C': 2, 'Q': 3}
    df['Embarked'] = df['Embarked'].map(category)

    # count family size (sns: 0.017->survived)
    def count_fs(input):
        sum_out = int(input[0]) + int(input[1]) + 1
        # print(input)
        return sum_out

    df['fs'] = df[['SibSp', 'Parch']].apply(count_fs, axis=1)

    # is_alone (sns: 0.2->survived)
    def check_it(input):
        if input > 1:
            return 0
        else:
            return 1

    df['is_alone'] = df['fs'].map(check_it)

    # # feature: Fare
    # df['FareBand'] = pd.qcut(df['Fare'], 4)
    # df2 = df[['FareBand', 'Survived']].groupby(['FareBand'],\
    # as_index=False).mean().sort_values(by='FareBand', ascending=True)

    df['Fare'] = df['Fare'].fillna(0)
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    # is the 3rd class
    def check_pclass(input):
        if input == 3:
            return 1
        else:
            return 0

    df['is_3class'] = df['Pclass'].map(check_pclass)

    # process age to age_proc
    df['Age'] = df['Age'].fillna(0)
    df['age_proc'] = 0
    df_age = df[['Age', 'SibSp', 'Parch', 'Pclass', 'Title', 'is_alone']]
    series_age = age_proc(df_age)
    list_0_age = series_age.index.values.tolist()
    df.loc[list_0_age, 'age_proc'] = series_age.values

    # age_g = df.groupby('Title')['Age'].mean().values

    def age_p(input):
        if input[1] != 0:
            return input[1]
        else:
            return input[2]

    df['age_proc'] = df[['Title', 'Age', 'age_proc']].apply(age_p, axis=1)

    # check the relationship of age and others
    # (cabin:0.21;fare:0.18;parch:-0.15;sibsp=fs:-0.24;pclass:-0.41;is_alone:-0.13)
    # df.dropna(subset=['Age'], inplace=True)
    # relation_check(df)

    # # preview age
    # df = df.dropna(subset=['age_proc'])
    # df['AgeBand'] = pd.qcut(df['age_proc'], 10)
    # df2 = df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False). \
    #     mean().sort_values(by='AgeBand', ascending=True)
    # a = df['age_proc'].mean()
    # b = df['age_proc'].std

    df.loc[(df['age_proc'] > 0) & (df['age_proc'] <= 16), 'age_proc'] = 1
    df.loc[(df['age_proc'] > 16) & (df['age_proc'] <= 20), 'age_proc'] = 2
    df.loc[(df['age_proc'] > 20) & (df['age_proc'] <= 23), 'age_proc'] = 3
    df.loc[(df['age_proc'] > 23) & (df['age_proc'] <= 26), 'age_proc'] = 4
    df.loc[(df['age_proc'] > 26) & (df['age_proc'] <= 29.165), 'age_proc'] = 5
    df.loc[(df['age_proc'] > 29.265) & (df['age_proc'] <= 30.141), 'age_proc'] = 6
    df.loc[(df['age_proc'] > 30.141) & (df['age_proc'] <= 34), 'age_proc'] = 7
    df.loc[(df['age_proc'] > 34) & (df['age_proc'] <= 40.14), 'age_proc'] = 8
    df.loc[(df['age_proc'] > 40.14) & (df['age_proc'] <= 48), 'age_proc'] = 9
    df.loc[df['age_proc'] > 48, 'age_proc'] = 10

    # df['Age_Class'] = df['age_proc'] * df['Pclass']

    # length of name
    df['name_len'] = df['Name'].apply(lambda df: len(re.findall(r'[a-zA-Z]+', df)))
    # lastname_list = df['Last_Name'].values.tolist()
    # simple_features_code(lastname_list)
    # df = simple_features_encode(df)

    # df2 = df.drop(['Age'], axis=1)
    # df2 = df2[df2.isnull().any(axis=1)]
    # print(df2)
    return df


def age_proc(df_age):
    df_pred = df_age[df_age['Age'] == 0]
    x_pred = df_pred.loc[:, df_pred.columns != 'Age'].values
    df_train = df_age[df_age['Age'] != 0]

    def split_data(df):
        data_x = df.loc[:, df.columns != 'Age'].values
        data_y = df['Age'].values
        return data_x, data_y

    x_train, y_train = split_data(df_train)
    # y_train = np.array(y_train)
    # x_trainage, x_valage, y_trainage, y_valage = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
    x_trainage = x_train[:790]
    y_trainage = y_train[:790]
    x_valage = x_train[790:]
    y_valage = y_train[790:]

    # ss_X = StandardScaler()
    # ss_y = StandardScaler()

    # X_train = ss_X.fit_transform(x_trainage)
    # X_test = ss_X.transform(x_valage)
    X_train = x_trainage
    X_test = x_valage

    # y_train = ss_y.fit_transform(y_trainage.reshape(-1, 1))
    # y_test = ss_y.transform(y_valage.reshape(-1, 1))
    y_train = y_trainage
    y_test = y_valage

    # uni_knr = KNeighborsRegressor(weights='uniform')  # 初始化平均回归的KNN回归器
    # uni_knr.fit(X_train, y_train)
    # uni_knr_y_predict = uni_knr.predict(X_test)
    # uni_k = uni_knr.score(X_test, y_test)
    # dis_knr = KNeighborsRegressor(weights='distance')  # 初始化距离加权回归的KNN回归器
    # dis_knr.fit(X_train, y_train)
    # dis_knr_y_predict = dis_knr.predict(X_test)
    # dis_k = dis_knr.score(X_test, y_test)

    gbm0 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(
        X_train, y_train)
    mea = mean_squared_error(y_test, gbm0.predict(X_test))

    # df_new = pd.DataFrame(data={'pred': dis_knr_y_predict.tolist(), 'ori': y_test.tolist()})
    # print('R-squared value of uniform-weighted KNR:', uni_k)
    # print('R-squared value of distance-weighted KNR:', dis_k)
    # if uni_k > dis_k:
    #     pred_list = uni_knr.predict(x_pred)
    # else:
    #     pred_list = dis_knr.predict(x_pred)
    pred_list = gbm0.predict(x_pred)
    df_pred['age_proc'] = pred_list
    return df_pred['age_proc']


def carbin_pred(df):
    """Creating 'Family_Survived' feature"""
    # A function working on family survival rate using last names and ticket features
    df['Last_Name'] = df['Name'].apply(lambda df: str.split(df, ",")[0])

    # Adding new feature: 'Survived'
    default_survival_rate = 0.5
    df['Family_Survival'] = default_survival_rate

    for grp, grp_df in df[
        ['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Age',
         'Cabin']].groupby(['Last_Name', 'Fare']):

        if (len(grp_df) != 1):
            # A Family group is found.
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0
                # if passID > 1000:
                #     df_show_fs = df.loc[df['PassengerId'] == passID]
                #     print(df_show_fs)

    for _, grp_df in df.groupby('Ticket'):
        if (len(grp_df) != 1):
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) | (
                        row['Family_Survival'] == 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if (smax == 1.0):
                        df.loc[df['PassengerId'] ==
                               passID, 'Family_Survival'] = 1
                    elif (smin == 0.0):
                        df.loc[df['PassengerId'] ==
                               passID, 'Family_Survival'] = 0

    return df


def relation_check(df):
    sns.set(rc={'figure.figsize': (18, 18)})
    s = sns.heatmap(df.corr(), annot=True)
    s.get_figure().savefig('proc_sns.png', bbox_inches='tight')
    print('proc_sns.png is generated!')


# def simple_features_code(list_in):
#     print("simple_features_code start.")
#     path = '../data/code_feature.txt'
#     with open(path, 'w', encoding='utf-8') as f_w:
#         list_unq = list(set(list_in))
#         string_unq = ",".join(list_unq)
#         f_w.write(string_unq)
#     print("simple_features_code done.")
#
#
# def simple_features_encode(df):
#     print("simple_features_encode start.")
#     path = '../data/code_feature.txt'
#     with open(path, 'r', encoding='utf-8') as f_r:
#         read_list = f_r.readline()
#     list_unq = read_list.split(",")
#
#     df['ln_encode'] = 0
#     for i in range(len(list_unq)):
#         df.loc[df['Last_Name'] == list_unq[i], 'ln_encode'] = i
#     print("simple_features_encode done.")
#     return df


if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    test.insert(loc=1, column='Survived', value=0)
    # train data:[:891]; test data:[891:]
    df_merge = pd.concat([train, test], sort=False, ignore_index=True)
    df_merge = preprocess(df_merge)

    features_list = ['Pclass', 'Sex', 'Fare', 'cabin_p1', 'Embarked', 'age_proc', 'fs',
                     'is_alone', 'Title', 'is_3class', 'has_cabin', 'SibSp', 'Parch', 'name_len', 'Family_Survival']
    # features_list = ['Pclass', 'Sex', 'Fare', 'Cabin', 'is_alone', 'Title']
    df_merge = df_merge[features_list + ['Survived']]
    relation_check(df_merge)
    df_merge.to_csv('../data/df_merge.csv', index=False)

    scaler = StandardScaler()
    df_merge_std = pd.DataFrame(scaler.fit_transform(df_merge[features_list].values), columns=features_list)
    df_merge_std['Survived'] = df_merge['Survived']
    df_train = df_merge_std[:891]
    df_test = df_merge_std[891:]
    df_train.to_csv('../data/train_proc.csv', index=False)
    df_test.to_csv('../data/test_proc.csv', index=False)
