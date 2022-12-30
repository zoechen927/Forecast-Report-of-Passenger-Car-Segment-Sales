import pandas as pd
import numpy as np
# from mlxtend.regressor import StackingRegressor
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

data_path = './data/'

train_sales_data = pd.read_csv(data_path + 'train_sales_data.csv', encoding='utf-8')
train_search_data = pd.read_csv(data_path + 'train_search_data.csv', encoding='utf-8')
test_data = pd.read_csv(data_path + 'evaluation_public.csv', encoding='utf-8')

data = pd.concat([train_sales_data, test_data], ignore_index=True)
data = data.merge(train_search_data, on=['province', 'adcode', 'model', 'regYear', 'regMonth'],how='left')
print(data)

data['label'] = data['salesVolume']

del data['salesVolume'], data['forecastVolum']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales_data.drop_duplicates('model').set_index('model')['bodyType'])
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))

data['seq'] = (data['regYear']-2016) * 12 + data['regMonth']#获取时间序列标记

data['model_adcode'] = data['adcode'] + data['model']
data['model_adcode_seq'] = data['model_adcode'] * 100 + data['seq']

data['adcode_seq'] = data['adcode']*100+data['seq']
data['model_seq'] = (data['model'])*10000+data['seq']

data['label'] = np.log1p(data['label'])

from sklearn.metrics import mean_squared_error

def metrics(y_true, y_pred, model):
    data = pd.DataFrame({'model': model, 'salesVolume': y_true, 'label': y_pred})
    data['label'] = data['label'].map(lambda index: -index if index < 0 else index)
    res, count = 0, 0
    for index, cars in data.groupby('model'):
        a = np.array(cars['salesVolume'])
        b = np.array(cars['label'])
        temp = np.sqrt(np.sum((a - b) ** 2) / len(a)) / np.mean(a)
        res += temp
        count += 1
        print(temp)
    return 1 - (res / count)

def calculate_sum_mean(feature, month):#计算过去几月的特征和
#     data[feature.format('_diff_1')] = data[feature.format(2)]-data[feature.format(1)]
    data[feature.format('sum_{0}'.format(month))] = 0
    for i in range(1, month+1):
        data[feature.format('sum_{0}'.format(month))] += data[feature.format(i)]
    data[feature.format('mean')] = data[feature.format('sum_{0}'.format(month))]/month

def get_time_shift_feature(Data, month):
    data = Data[['adcode','bodyType','id', 'model', 'regMonth', 'regYear', 'label', 'seq', 'model_adcode', 
                   'model_adcode_seq','adcode_seq', 'model_seq', 'popularity']]
    for j in range(1,13):
        data['model_adcode_seq_{0}'.format(j)] = data['model_adcode_seq'] + j
        data_index = data[~data.label.isnull()].set_index('model_adcode_seq_{0}'.format(j))
        data['shift_label_{0}'.format(j)] = data['model_adcode_seq'].map(data_index['label'])
        if month==1:
            data['shift_popularity_{0}'.format(j)] = data['model_adcode_seq'].map(data_index['popularity'])
        data = data.drop(['model_adcode_seq_{0}'.format(j)], axis=1)
    return data

def get_group_shift_feature(data,group_feature):
    Data = data
    g_data = Data.groupby(by=[group_feature])['label'].sum()
    g_data = g_data.fillna(np.nan).reset_index()
    for j in range(1,13):
        g_data[group_feature+'_{0}'.format(j)] = g_data[group_feature] + j
        g_data_index = g_data[~g_data.label.isnull()].set_index(group_feature+'_{0}'.format(j))
        g_data[group_feature+'_shift_{0}'.format(j)] = g_data[group_feature].map(g_data_index['label'])
        del g_data[group_feature+'_{0}'.format(j)]
    del g_data['label']
    data = pd.merge(data, g_data, on=[group_feature], how='left')
    return data

def get_history_label_feature(month):
    for i in [2,3,4,6,12]:
        calculate_sum_mean('shift_label_{0}', i)
        if month==1:
            calculate_sum_mean('shift_popularity_{0}', i)
        calculate_sum_mean('adcode_seq_shift_{0}', i)
        calculate_sum_mean('model_seq_shift_{0}', i)

lgb_model = lgb.LGBMRegressor(
        num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
        max_depth=-1, learning_rate=0.05, min_child_samples=5, seed=2020,
        n_estimators=2000, subsample=0.9, colsample_bytree=0.7
)

for i in range(1, 5):
    print('=================predict month {0}=================='.format(i))

    data = get_time_shift_feature(data, i)
    data = get_group_shift_feature(data, 'adcode_seq')
    data = get_group_shift_feature(data, 'model_seq')
    get_history_label_feature(i)

    data_columns = list(data.columns)
    dels = ['regMonth', 'regYear', 'adcode', 'bodyType', 'id', 'model', 'province', 'label', 'seq', 'model_adcode',
                'model_adcode_seq', 'adcode_seq', 'model_seq', 'popularity']
    number_feature = []
    for index in data_columns:
        if index in dels:
            continue
        else:
            number_feature.append(index)

    category_feature = ['regMonth', 'regYear', 'adcode', 'bodyType', 'model', 'model_adcode_seq', 'model_adcode']
    features = list(number_feature) + category_feature

    predict_data = data[data['seq'] == 24 + i]
    train_idx = (data['seq'].between(13, 23 + i))

    train_y = data[train_idx]['label']
    train_x = data[train_idx][features]

    print("train LGB model\n")
    lgb_model.fit(train_x, train_y, categorical_feature=category_feature)
    predict_data['lgb_pred_label'] = lgb_model.predict(predict_data[features])
    print('month {} train ending\n'.format(i))

    predict_data = predict_data.sort_values(by=['id'])
    data['transform_label'] = data['id'].map(predict_data.set_index('id')['lgb_pred_label'])
    data['label'] = data['label'].fillna(data['transform_label'])
    del data['transform_label']

data['label'] = np.expm1(data['label'])
predict_data_idx = (data['seq'] > 24)

data['forecastVolum'] = data['label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
data[predict_data_idx][['id', 'forecastVolum']].to_csv('./submit/lgb_two.csv', index=False)

data[predict_data_idx]['forecastVolum'].mean()
data[predict_data_idx].groupby(['regMonth'])['forecastVolum'].mean()
