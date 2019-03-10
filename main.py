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
print(df.head())
"""
for i in range(1, 12):
    print("Mes {}:".format(i))
    df_by_month = df[df.sold_at.dt.month == i]
    qtd_cars = len(df_by_month)
    cars_mean_price = df_by_month.price.mean()
    if qtd_cars != 0:
        print("Foram vendidos {} carros numa média de {} - Razao: {}".format(qtd_cars, cars_mean_price, cars_mean_price/qtd_cars))
    else:
        print("Não foram vendidos carros neste mês")
"""
paint_colors = list(df.paint_color.unique())
price_paint_color = []
for paint_color in paint_colors:
    df[df.paint_color == paint_color].mean()
    price_paint_color.append({'cor': paint_color, 'price_mean': df[df.paint_color == paint_color].price.mean()})
price_paint_color = sorted(price_paint_color, key=lambda k: k['price_mean'])





normalized_paint_colors = []
for price in price_paint_color:
    p = (price['price_mean'] - price_paint_color[0]['price_mean']) / (price_paint_color[-1]['price_mean'] - price_paint_color[0]['price_mean'])
    normalized_paint_colors.append(p)

for pc in price_paint_color:
    df.loc[df['paint_color'] == pc['cor'], 'paint_color'] = normalized_paint_colors[price_paint_color.index(pc)]

df.paint_color = pd.to_numeric(df.paint_color)

feature_w = []



for i in range(1, 9):
    filtered = df[df['feature_{}'.format(i)]]
    plt.xlabel('Price')
    plt.ylabel('Feature {}'.format(i))
    #sns.distplot(filtered.price)
    feature_w.append({'mean':float(filtered.price.mean()), 'name': 'feature_{}'.format(i)})
    #plt.show()

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

df['registration_date']=df['registration_date'].map(dt.datetime.toordinal)
df['sold_at']=df['sold_at'].map(dt.datetime.toordinal)
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
    'sum_features'
]
]

X['mileage']=(X['mileage']-X['mileage'].min())/(X['mileage'].max()-X['mileage'].min())
X['engine_power']=(X['engine_power']-X['engine_power'].min())/(X['engine_power'].max()-X['engine_power'].min())
X['registration_date']=(X['registration_date']-X['registration_date'].min())/(X['registration_date'].max()-X['registration_date'].min())
print(X.head())
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
#plt.scatter(y_test, y_pred)
#plt.xlabel('Y Test')
#plt.ylabel('Predicted Y')
sns.distplot(y_test-y_pred)
plt.show()