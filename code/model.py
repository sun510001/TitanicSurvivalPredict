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


def train_model(file_path, fig_name, model, train_x, train_y):
    # checkpoint
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # fit
    history = model.fit(train_x, train_y, validation_split=0.17, epochs=15, batch_size=1,
                        callbacks=callbacks_list, verbose=0)
    hist = history.history
    loss_values = hist['loss']
    val_loss_values = hist['val_loss']
    acc_values = hist['acc']
    val_acc_values = hist['val_acc']
    # draw image
    fig = plt.figure()
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss', color='b')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss', color='b')
    plt.plot(epochs, acc_values, 'bo', label='Training acc', color='r')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc', color='r')
    plt.title('Training and val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(fig_name, dpi=fig.dpi)


def Random_Forest(X_train, Y_train, X_test):
    random_forest = RandomForestClassifier(n_estimators=100)

    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print('Random forest acc:', acc_random_forest)
    return Y_pred


def train_it(list_features, filepath, is_forest):
    df_train = pd.read_csv('../data/train_proc.csv')
    fig_name = 'loss.v1.png'
    # print(train.shape, test.shape)

    len_features = len(list_features)
    train_x, train_y = get_data(df_train, list_features)

    if is_forest == 0:
        model = build_model_1(len_features)
        train_model(filepath, fig_name, model, train_x, train_y)
        return []
    else:
        # Random Forest
        df_test = pd.read_csv('../data/test_proc.csv')
        X_test = df_test[list_features].values
        # Y_test = Random_Forest(train_x, train_y, X_test)
        Y_test = Random_Forest(train_x, train_y, X_test)
        return Y_test


def test_it(list_features, filepath, Y_test):
    df_gen = pd.read_csv('../data/gender_submission.csv')
    df_test = pd.read_csv('../data/test_proc.csv')

    if Y_test == []:
        model = models.load_model(filepath)
        test_x = df_test[list_features].values
        test_y = model.predict(test_x)
        df_gen['Survived'] = test_y
    else:
        df_gen['Survived'] = Y_test

    df_gen.loc[df_gen['Survived'] <= 0.5, 'Survived'] = 0
    df_gen.loc[df_gen['Survived'] > 0.5, 'Survived'] = 1
    df_gen['Survived'] = df_gen['Survived'].astype(int)
    df_gen.to_csv('../data/gender_submission_out.csv', index=False)



if __name__ == '__main__':
    list_features = ['Pclass', 'Sex', 'Fare', 'cabin_p1', 'Embarked', 'Parch', 'age_proc', 'fs',
                     'is_alone', 'Title', 'is_3class', 'has_cabin', 'name_len', 'Family_Survival']
    file_path = "../data/best_model.hdf5"
    is_forest = 0
    rate = 0
    Y_test = train_it(list_features, file_path, is_forest)
    test_it(list_features, file_path, Y_test)
