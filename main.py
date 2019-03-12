# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn import preprocessing

df = pd.read_csv('data/bmw_pricing_challenge.csv')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.sold_at = pd.to_datetime(df.sold_at)
df.registration_date = pd.to_datetime(df.registration_date)
df = df.query('price < 60000')


car_types = list(df.car_type.unique())
for ct in car_types:
    dff = df.query('car_type == "{}"'.format(ct))
    ar_c_m = []
    for i in range(1, 13):
        print("Mes {}:".format(i))
        df_by_month = dff[dff.sold_at.dt.month == i]
        qtd_cars = len(df_by_month)
        cars_mean_price = df_by_month.price.mean()
        if qtd_cars != 0:
            ar_c_m.append(cars_mean_price/qtd_cars)
        else:
            ar_c_m.append(0)
        if qtd_cars != 0:
            print("Foram vendidos {} carros numa média de {} - Razao: {}".format(qtd_cars, cars_mean_price, cars_mean_price/qtd_cars))
        else:
            print("Não foram vendidos carros neste mês")
    plt.plot(ar_c_m, label=ct)
    plt.legend()

plt.show()


def normalize_features(features:list):
    normalized_features = []
    for item in features:
        p = (item['mean'] - features[0]['mean']) / (
                    features[-1]['mean'] - features[0]['mean'])
        normalized_features.append(p)
    return normalized_features


def cat_mean(feature:str, df:pd.DataFrame):
    features_list = list(df[feature].unique())
    features_values_list = []
    for f in features_list:
        m = df[df[feature] == f].price.mean()
        features_values_list.append({feature: f, 'mean': m})
    features_values_list = sorted(features_values_list, key=lambda k: k['mean'])
    normalized_features = normalize_features(features_values_list)
    for f in features_values_list:
        df.loc[df[feature] == f[feature], feature] = normalized_features[features_values_list.index(f)]
    df[feature] = pd.to_numeric(df[feature])
    return df

df = cat_mean('car_type', df)
df = cat_mean('paint_color', df)
df = cat_mean('fuel', df)
df = cat_mean('model_key', df)

feature_w = []



for i in range(1, 9):
    filtered = df[df['feature_{}'.format(i)]]
    feature_w.append({'mean':float(filtered.price.mean()), 'name': 'feature_{}'.format(i)})


df['feature_1'] = df['feature_1'].astype(float)
df['feature_2'] = df['feature_2'].astype(float)
df['feature_3'] = df['feature_3'].astype(float)
df['feature_4'] = df['feature_4'].astype(float)
df['feature_5'] = df['feature_5'].astype(float)
df['feature_6'] = df['feature_6'].astype(float)
df['feature_7'] = df['feature_7'].astype(float)
df['feature_8'] = df['feature_8'].astype(float)

df['sum_features'] = np.zeros(len(df))




feature_w = sorted(feature_w, key=lambda k: k['mean'])
import datetime as dt

df['registration_date'] = df['registration_date'].map(dt.datetime.toordinal)
df['sold_at'] = df['sold_at'].map(dt.datetime.toordinal)
normalized_features = []

for feature in feature_w:
    p = ((feature['mean'] - feature_w[0]['mean']) / (feature_w[-1]['mean'] - feature_w[0]['mean']))
    normalized_features.append(p)

for f in feature_w:
    df.loc[df[f['name']] == True, f['name']] = normalized_features[feature_w.index(f)]

df['sum_features'] = (df['feature_1'] + df['feature_2']+ df['feature_3']+ df['feature_4']+ df['feature_5']+ df['feature_6']+ df['feature_7']+ df['feature_8'])*10
y = df['price']
X = df[[
    'mileage',
    'engine_power',
    'registration_date',
    'paint_color',
    'sum_features',
    'fuel',
    #'car_type',
    'model_key'
]
]

X['mileage'] = (X['mileage']-X['mileage'].min())/(X['mileage'].max()-X['mileage'].min())
X['engine_power'] = (X['engine_power']-X['engine_power'].min())/(X['engine_power'].max()-X['engine_power'].min())
X['registration_date'] = (X['registration_date']-X['registration_date'].min())/(X['registration_date'].max()-X['registration_date'].min())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.scatter(y_test, y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
#sns.distplot(y_test-y_pred)
plt.show()