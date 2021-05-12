import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from deepctr.models import DMR

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 去掉按需增长就没报错了？tensorflow.python.framework.errors_impl.InternalError: GPU sync failed
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from config import FRAC
SESS_MAX_LEN = 50
TEST_BATCH_SIZE = 2 ** 14
BATCH_SIZE = 1024

input_file = '../model_input/'
samlpe_file = '../sampled_data/'
feature_columns = pd.read_pickle(input_file + 'din_fd_' + str(FRAC) + '_' + str(SESS_MAX_LEN) +'.pkl')
# feature_columns=feature_columns[:-2] # 多var的话 deepfm如何处理  elif isinstance(fc, VarLenSparseFeat):
# feature_columns=feature_columns[:2] # 0.61
# feature_columns=feature_columns[:-5] 
# feature_columns=feature_columns[:13] 

# feature_columns=feature_columns[:2]+feature_columns[-5:-4] # 0.5
# feature_columns=feature_columns[:2]+feature_columns[-4:-3]
# feature_columns=feature_columns[:2]+feature_columns[-5:-3]
# feature_columns=feature_columns[:2]+feature_columns[-5:-3]+feature_columns[-2:]
# print('feature_columns',feature_columns)
for feat in feature_columns:
    print(feat)
model_input = pd.read_pickle(input_file + 'din_input_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
print('model_input',model_input)
behavior_length = np.count_nonzero(model_input['hist_brand'], axis=1)
model_input['seq_length'] = behavior_length
# input('c')
label = pd.read_pickle(input_file + 'din_label_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
# print('label',label.shape,sum(label))
# input('c')
sample_sub = pd.read_pickle(samlpe_file + 'raw_sample_' + str(FRAC) + '.pkl')
sample_sub['idx'] = list(range(sample_sub.shape[0]))

train_idx = sample_sub.loc[sample_sub.time_stamp < 1494633600, 'idx'].values
test_idx = sample_sub.loc[sample_sub.time_stamp >= 1494633600, 'idx'].values

print('model_input',type(model_input),model_input.keys())
# print('train_idx',train_idx)
# for feat in model_input:
#     print('feat',feat)
#     print('model_input[feat][train_idx]',model_input[feat][train_idx])

train_input = {feat : model_input[feat][train_idx] for feat in model_input}

test_input = {feat : model_input[feat][test_idx] for feat in model_input}

train_label = label[train_idx]
test_label = label[test_idx]

sess_len_max = SESS_MAX_LEN

behavior_feature_list = ['cate_id', 'brand']


# In[6]:


# train_input['seq_length'].shape


# In[4]:


train_input.keys()


# In[15]:


# model = DeepFM(feature_columns, feature_columns, task='binary')
# print('din')
# model = BST(feature_columns, behavior_feature_list)
deep_match_id = "brand"
model = DMR(feature_columns, behavior_feature_list, deep_match_id)
# model = DIN(feature_columns, behavior_feature_list, gru_type="AGRU",use_negsampling=True,
#              dnn_hidden_units=(200, 80),use_bn=False, dnn_activation='relu',
#              att_hidden_units=(64, 16), att_activation="relu", att_weight_normalization=True,
#              l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary',
#              device='cpu')


# In[16]:


model.compile('adagrad', 'binary_crossentropy',
              metrics=['binary_crossentropy'])


# In[ ]:

print('train_label',len(train_label),sum(train_label))
hist_ = model.fit(train_input, train_label, batch_size=BATCH_SIZE,
                      epochs=1, verbose=1)

# epochs=3 :tensorflow.python.framework.errors_impl.InternalError: GPU sync failed   
# ?


pred_ans = model.predict(test_input, batch_size=TEST_BATCH_SIZE)
print('pred_ans',len(pred_ans),sum(pred_ans))
print("test LogLoss", round(log_loss(test_label, pred_ans), 4), 
      "test AUC", round(roc_auc_score(test_label, pred_ans), 4)) 

