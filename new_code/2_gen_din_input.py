# coding: utf-8

import os

import numpy as np
import pandas as pd
# from deepctr.utils import SingleFeat
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from config import DIN_SESS_MAX_LEN, FRAC


def gen_sess_feature_din(row):
    sess_max_len = DIN_SESS_MAX_LEN
    sess_input_dict = {'cate_id': [0], 'brand': [0]}
    sess_input_length = 0
    user, time_stamp = row[1]['user'], row[1]['time_stamp']
    if user not in user_hist_session or len(user_hist_session[user]) == 0:

        sess_input_dict['cate_id'] = [0]
        sess_input_dict['brand'] = [0]
        sess_input_length = 0
    else:
        cur_sess = user_hist_session[user][0]
        for i in reversed(range(len(cur_sess))):
            if cur_sess[i][2] < time_stamp:
                sess_input_dict['cate_id'] = [e[0]
                                              for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_dict['brand'] = [e[1]
                                            for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_length = len(sess_input_dict['brand'])
                break
    return sess_input_dict['cate_id'], sess_input_dict['brand'], sess_input_length


if __name__ == "__main__":

    user_hist_session = {}
    FILE_NUM = len(
        list(
            filter(lambda x: x.startswith('user_hist_session_' + str(FRAC) + '_din_'), os.listdir('../sampled_data/'))))

    print('total', FILE_NUM, 'files')
    for i in range(FILE_NUM):
        user_hist_session_ = pd.read_pickle(
            '../sampled_data/user_hist_session_' + str(FRAC) + '_din_' + str(i) + '.pkl')
        user_hist_session.update(user_hist_session_)
        del user_hist_session_

    sample_sub = pd.read_pickle(
        '../sampled_data/raw_sample_' + str(FRAC) + '.pkl')

    sess_input_dict = {'cate_id': [], 'brand': []}
    sess_input_length = []
    for row in tqdm(sample_sub[['user', 'time_stamp']].iterrows()):
        a, b, c = gen_sess_feature_din(row)
        sess_input_dict['cate_id'].append(a)
        sess_input_dict['brand'].append(b)
        sess_input_length.append(c)

    print('done')

    user = pd.read_pickle('../sampled_data/user_profile_' + str(FRAC) + '.pkl')
    ad = pd.read_pickle('../sampled_data/ad_feature_enc_' + str(FRAC) + '.pkl')
    user = user.fillna(-1)
    user.rename(
        columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)

    sample_sub = pd.read_pickle(
        '../sampled_data/raw_sample_' + str(FRAC) + '.pkl')
    sample_sub.rename(columns={'user': 'userid'}, inplace=True)

    data = pd.merge(sample_sub, user, how='left', on='userid', )
    data = pd.merge(data, ad, how='left', on='adgroup_id')

    sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                       'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
                       'customer'] #+ ['cate_id', 'brand']
    dense_features = ['price']

    # print(data[['cate_id', 'brand']])
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()  # or Hash
        data[feat] = lbe.fit_transform(data[feat])
    # print(data[['cate_id', 'brand']])
    mms = StandardScaler()
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SparseFeat(feat, data[feat].max(
    ) + 1) for feat in sparse_features + ['cate_id', 'brand']]
    # sparse_feature_list = [SparseFeat(feat, data[feat].nunique(
    # ) + 1) for feat in sparse_features + ['cate_id', 'brand']]

    dense_feature_list = [DenseFeat(feat, 1) for feat in dense_features]
    sess_feature = ['cate_id', 'brand']

    feature_columns=sparse_feature_list+dense_feature_list

    # sess_input = [pad_sequences(
    #     sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post') for feat in sess_feature]

    model_input={}
    for feat in sparse_feature_list:
        model_input[feat.name]=data[feat.name].values
    print('model_input',type(model_input),model_input.keys())
    for feat in dense_feature_list:
        model_input[feat.name]=data[feat.name].values
    print('model_input',type(model_input),model_input.keys())

    for feat in sess_feature:
        feature_columns.append(VarLenSparseFeat(SparseFeat('hist_'+feat, data[feat].max(
    ) + 1), maxlen=DIN_SESS_MAX_LEN, length_name="seq_length"))
        model_input['hist_'+feat]=pad_sequences(sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post')
    

    # sess_lists = sess_input  # + [np.array(sess_input_length)]
    # model_input += sess_lists

    if not os.path.exists('../model_input/'):
        os.mkdir('../model_input/')

    pd.to_pickle(model_input, '../model_input/din_input_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    pd.to_pickle([np.array(sess_input_length)], '../model_input/din_input_len_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')

    pd.to_pickle(data['clk'].values, '../model_input/din_label_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    # pd.to_pickle({'sparse': sparse_feature_list, 'dense': dense_feature_list},
    #              '../model_input/din_fd_' + str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl', )


    pd.to_pickle(feature_columns,
                 '../model_input/din_fd_' + str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl', )

    print("gen din input done")