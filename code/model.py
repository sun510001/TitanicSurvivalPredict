# -*- coding:utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
from keras import models, layers
from keras.layers import Embedding, Dense, Flatten, Dropout, Input, LSTM, Bidirectional, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)


def get_data(df, list_features):
    data_x = df[list_features].values
    data_y = df['Survived'].values
    return data_x, data_y


def build_model_1(len_features):
    inputs = Input(shape=(len_features,))
    # x = Flatten()(inputs)
    x = layers.Dense(24, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = layers.Dense(24, activation='relu')(x)
    # x = layers.Dense(24, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # model.summary()
    return model


def train_model(model, train_x, train_y):
    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)

    for fold, (ids_train, ids_test) in enumerate(skf.split(train_x, train_y)):
        sv = ModelCheckpoint(weight_fn.format(VER, fold), monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='max')
        history = model.fit(train_x[ids_train], train_y[ids_train],
                            validation_data=[train_x[ids_test], train_y[ids_test]], epochs=15, batch_size=1,
                            callbacks=[sv], verbose=1, shuffle=False)

        # hist = history.history
        # loss_values = hist['loss']
        # val_loss_values = hist['val_loss']
        # acc_values = hist['acc']
        # val_acc_values = hist['val_acc']
        # # draw image
        # fig = plt.figure()
        # epochs = range(1, len(loss_values) + 1)
        # plt.plot(epochs, loss_values, 'bo', label='Training loss', color='b')
        # plt.plot(epochs, val_loss_values, 'b', label='Validation loss', color='b')
        # plt.plot(epochs, acc_values, 'bo', label='Training acc', color='r')
        # plt.plot(epochs, val_acc_values, 'b', label='Validation acc', color='r')
        # plt.title('Training and val loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # fig.savefig(fig_name, dpi=fig.dpi)


def train_it(list_features):
    df_train = pd.read_csv('../data/train_proc.csv')
    fig_name = 'loss.v1.png'
    # print(train.shape, test.shape)

    len_features = len(list_features)
    train_x, train_y = get_data(df_train, list_features)
    model = build_model_1(len_features)
    train_model(model, train_x, train_y)


def test_it(list_features, num, df_gen):
    # df_gen = pd.read_csv('../data/gender_submission.csv')
    df_test = pd.read_csv('../data/test_proc.csv')

    model = models.load_model(weight_fn.format(VER, num))
    test_x = df_test[list_features].values
    test_y = model.predict(test_x)
    col_n = 'Survived-{0}'.format(num)
    df_gen[col_n] = test_y

    # df_gen.loc[df_gen[col_n] <= 0.5, col_n] = 0
    # df_gen.loc[df_gen[col_n] > 0.5, col_n] = 1
    # df_gen[col_n] = df_gen[col_n].astype(int)
    return df_gen
    # df_gen.to_csv('../data/gender_submission_out_{0}.csv'.format(num), index=False)


def com_answer(df):
    # df = pd.read_csv('../data/gender_submission_out.csv')
    df_ans = pd.read_csv('../data/100answer.csv')
    df_2 = df.drop('PassengerId', axis=1)
    # df['Survived'] = df['Survived-{0}'.format(1)]
    df['Survived'] = df_2.mean(axis=1)
    df.loc[df['Survived'] <= 0.5, 'Survived'] = 0
    df.loc[df['Survived'] > 0.5, 'Survived'] = 1
    df['Survived'] = df['Survived'].astype(int)
    df['ans'] = df_ans['Survived']

    count = [0]

    def com(input):
        if input[0] != input[1]:
            count[0] += 1

    df[['Survived', 'ans']].apply(com, axis=1)

    miss = count[0]
    sum = df['ans'].count()
    rate = (sum - miss) / sum
    print("Miss count:", miss, "\nSum count:", sum, "\nAcc rate:", rate)
    print("Acc rate:", rate)

    df = df[['PassengerId', 'Survived']]
    df.to_csv('../data/gender_submission_out_6_11_2.csv', index=False)


if __name__ == '__main__':
    list_features = ['Pclass', 'Sex', 'Fare', 'cabin_p1', 'Embarked', 'Parch', 'age_proc', 'fs',
                     'is_alone', 'Title', 'is_3class', 'has_cabin', 'name_len', 'Family_Survival']

    weight_fn = '../data/{0}-titanic-{1}.h5'
    rate = 0
    FOLD = 5
    SEED = 161
    df_gen = pd.read_csv('../data/gender_submission.csv')
    df_gen = df_gen.drop('Survived', axis=1)
    VER = 2

    # train_it(list_features)
    for i in range(FOLD):
        df_gen = test_it(list_features, i, df_gen)
    com_answer(df_gen)
