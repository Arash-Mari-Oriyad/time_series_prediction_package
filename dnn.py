import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing

import configurations

EPSILON = 10 ** (-10)

data = pd.read_csv('usa_data/historical_data h=3.csv')

data.drop(configurations.NON_FEATURE_COLUMNS_NAMES, axis=1, inplace=True)
data.drop(configurations.NORMAL_TARGET_COLUMN_NAME, axis=1, inplace=True)
data.rename(columns={'Target (normal)': 'target'}, inplace=True)
data = data[data['target'].notna()]
data = data[data['target'] > 0]

data = data[['target', 'virus-pressure t+2', 'population_density', 'Retail t', 'confirmed t',
             'virus-pressure t+3', 'female-percent',
             'total_population']]

X = data.drop('target', axis=1).values
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
y = data['target'].to_numpy()
y = y.reshape(y.shape[0], 1)

model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mape', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, batch_size=8)

y_hat = model.predict(X)

mape = np.mean(np.abs(y - y_hat) / y)

print(y.shape)
print(y_hat.shape)

print('mape =', mape)

print(y[:20])
print(y_hat[:20])
