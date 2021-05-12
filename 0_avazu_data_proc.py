import pandas as pd
data = pd.read_csv('train.csv',nrows=16000000)  # there are 13330873 samples in the first 3d
print(data.shape)
data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
data['hour'] = data['hour'].apply(lambda x: str(x)[6:])
print(data.shape)
data_first_3d=data[data['day']<'24']  # first 3 days: 21 22 23
print(data_first_3d.shape)
print(data_first_3d['day'].unique())
data_first_3d.to_csv('avazu_first_3d.csv',index=False)
print('ok')

