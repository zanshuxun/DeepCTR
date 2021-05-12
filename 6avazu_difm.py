# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if __name__ == "__main__":
    # training parameters
    epochs = 1
    batch_size = 512

    # all the features in avazu dataset are sparse features
    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'device_type', 'device_conn_type',  # 'device_ip', 
                       'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]
    print('len(sparse_features)', len(sparse_features))  # id, click, device_ip, and day are not used. 25-4=21

    target = ['click']

    try:
        # read data from pkl directly
        data=pd.read_pickle('data_avazu_first_3d.pkl')
        print('read_pickle ok')
    except:    
        print('preprocess data and save it by pickle')
        data = pd.read_csv('avazu_first_3d.csv')
        # data = pd.read_csv('avazu_first_3d.csv',nrows=50)  # for test
        data[sparse_features] = data[sparse_features].fillna('-1', )
        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])

        data.to_pickle('data_avazu_first_3d.pkl')
        print('to_pickle ok')

    print(data[:5])
    print(data['day'].unique())

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=8)
                              for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train = data[data['day'] < 23]  # first 3 days: 21 22 23
    test = data[data['day'] == 23]
    print('train.shape',train.shape)
    print('test.shape',test.shape)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    model = DIFM(linear_feature_columns, dnn_feature_columns, task='binary', att_head_num=8)
    print('model', model)
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    model.fit(train_model_input, train[target].values,
              batch_size=batch_size, epochs=epochs, verbose=1)

    pred_ans = model.predict(test_model_input, batch_size*20)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
