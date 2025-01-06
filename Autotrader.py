import numpy as np
import pandas as pd
import coinbase as cb
import json
import yfinance as yf
from datetime import datetime, timedelta, timezone
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from sklearn.preprocessing import StandardScaler
from keras import models, layers, regularizers
import xgboost as xgb
from xgboost import XGBRegressor
import cvxpy as cp
import uuid
from sklearn.covariance import LedoitWolf
from keras.layers import BatchNormalization, Dropout
import warnings
warnings.filterwarnings('ignore')

now = datetime.now()
Y = now.year
m = now.month
d = now.day
now = f"{Y}-{m}-{d} 00:00:00+00:00"

coinbase_key_path = "" # Insert Coinbase API Key Here

with open(coinbase_key_path, 'r') as config_file:
    config = json.load(config_file)
api_key_name = config['name']
api_key_secret = config['privateKey']

from coinbase.rest import RESTClient
client = RESTClient(api_key_name, api_key_secret)

pairs = client.get_products()
new_df = {'product_ids':[]}
for dictionary in pairs['products']:
  new_df['product_ids'].append(dictionary['product_id'])

pairs_df = pd.DataFrame.from_dict(new_df, orient='index')
pairs_df = pairs_df.T

tickers = [product for product in pairs_df['product_ids']]
tickers_to_remove = []
for ticker in tickers:
  if not ticker.endswith('-USD'):
    tickers_to_remove.append(ticker)
tickers = [ticker for ticker in tickers if ticker not in tickers_to_remove]

crypto_data = yf.download(tickers, period='max')
crypto_df = crypto_data.swaplevel(axis=1)

class asset_window:
  def __init__(self, df):
      self.original_df = df.copy()

      if self.original_df.empty:
          raise ValueError("The DataFrame is empty.")
      
      
      first_day_close = self.original_df["Close"].iloc[0]
      self.df = df.copy()

      price_cols = ["Open", "High", "Low", "Close"]

      epsilon = 1e-9

      for col in price_cols:
          self.df.loc[:,col + "_LogReturn"] = np.log((self.df[col] + epsilon) / (self.df[col].shift(1) + epsilon))

      self.df.loc[:,"Volume_LogChange"] = np.log((self.df["Volume"] + epsilon) / (self.df["Volume"].shift(1) + epsilon))


      self.df.loc[:,"Close_Relative"] = self.df["Close"] / first_day_close

      vol_mean = self.original_df["Volume"].mean()
      vol_std = self.original_df["Volume"].std() if self.original_df["Volume"].std() != 0 else 1
      self.df.loc[:,"Volume_Zscore"] = (self.original_df["Volume"] - vol_mean) / vol_std

      self.df.loc[:,"Short_MA"] = self.original_df["Close"].rolling(window=3).mean()
      self.df.loc[:,"Short_MA_Relative"] = self.df["Short_MA"] / first_day_close
      self.df.drop(columns=price_cols, inplace=True)


#####################################################
##############ADJUST HERE############################
window_length = 7
lookback_length = 1001
desired_daily_return = 0.01
weight_limit = 0.14
rebalancing_threshold = 0.0001
#####################################################
#####################################################
from datetime import datetime, timedelta
day = crypto_df.index[-1]
train_set = []
test_set = []
for asset in crypto_df.columns.get_level_values(level=0).unique():
  if not asset.endswith('-USD'):
    continue
  asset_df = crypto_df[asset][:day]
  num_days = len(asset_df)
  end_train = num_days - (window_length + 1)
  start_train = end_train - lookback_length

  for i in range (start_train, end_train, window_length):
    window_slice = asset_df.iloc[i:i+window_length]
    if window_slice.isnull().values.any():
      continue
    a_window = asset_window(window_slice)
    if i > 0.85 * lookback_length + start_train:
      test_set.append(a_window)
    else:
      train_set.append(a_window)

X_train = []
y_train = []
for window in train_set:
  X_train.append(window.df.iloc[:-1].dropna().values.flatten())
  y_train.append(window.df.iloc[-1]["Close_LogReturn"])

X_test = []
y_test = []
for window in test_set:
  X_test.append(window.df.iloc[:-1].dropna().values.flatten())
  y_test.append(window.df.iloc[-1]["Close_LogReturn"])

shapes = [arr.shape[0] for arr in X_train if arr.shape[0] > 0]

if not shapes:
    raise ValueError("All arrays in X_train are empty. Check your data pipeline!")
most_common_shape = Counter(shapes).most_common(1)[0][0]

X_train_cleaned = []
y_train_cleaned = []

for i, window in enumerate(X_train):
    if isinstance(window, asset_window):
        df = window.df
        if df.shape[0] == most_common_shape and not np.isinf(df.values).any():
            X_train_cleaned.append(df.values.flatten())
            y_train_cleaned.append(y_train[i])
    else:
        expected_shape = most_common_shape
        if window.shape[0] == expected_shape and not np.isinf(window).any():
            X_train_cleaned.append(window)
            y_train_cleaned.append(y_train[i])

X_test_cleaned = []
y_test_cleaned = []

for i, window in enumerate(X_test):
    if isinstance(window, asset_window):
        df = window.df
        if df.shape[0] == most_common_shape and not np.isinf(df.values).any():
            X_test_cleaned.append(df.values.flatten())
            y_test_cleaned.append(y_test[i])
    else:
        expected_shape = most_common_shape
        if window.shape[0] == expected_shape and not np.isinf(window).any():
            X_test_cleaned.append(window)
            y_test_cleaned.append(y_test[i])


X_train = np.array(X_train_cleaned)
y_train = np.array(y_train_cleaned)

X_test = np.array(X_test_cleaned)
y_test = np.array(y_test_cleaned)

X_train += np.random.normal(loc=0.0, scale=0.01, size=X_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


shapes = [arr.shape[0] for arr in X_train if arr.shape[0] > 0]

if not shapes:
    raise ValueError("All arrays in X_train are empty. Check your data pipeline!")

shape_counts = Counter(shapes)
input_dim = shape_counts.most_common(1)[0][0]

input_layer = layers.Input((input_dim,))


encoder = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(input_layer)
encoder = BatchNormalization()(encoder)
encoder = Dropout(0.3)(encoder)

encoder = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01))(encoder)
encoder = BatchNormalization()(encoder)
encoder = Dropout(0.3)(encoder)


encoder = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.01))(encoder)
encoder = BatchNormalization()(encoder)
encoder = Dropout(0.3)(encoder)

encoder = layers.Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.01))(encoder)


decoder = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoder)
decoder = BatchNormalization()(decoder)

decoder = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(decoder)
decoder = BatchNormalization()(decoder)

decoder = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(decoder)
decoder = BatchNormalization()(decoder)

decoder = layers.Dense(input_dim, activation='tanh')(decoder)

autoencoder = models.Model(inputs=input_layer, outputs=decoder)
encoder = models.Model(inputs=input_layer, outputs=encoder)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss='mse')


callback = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
]

lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)

autoencoder.fit(
    X_train, X_train,
    validation_data=(X_test, X_test),
    epochs=25,
    batch_size=16,
    callbacks=[callback, lr_schedule]
)


X_train_latent = encoder.predict(X_train)
X_test_latent = encoder.predict(X_test)

xgb_model = XGBRegressor(n_estimators=1000, max_depth=25, objective = 'reg:absoluteerror', learning_rate = 0.01)
xgb_model.fit(X_train_latent, y_train)

dtrain = xgb.DMatrix(X_train, label=y_train)

start_date = crypto_df.index[crypto_df.index.get_loc(day) - (window_length - 1)]

current_week_df = crypto_df.loc[start_date:day]

current_week_data = {}
for asset in current_week_df.columns.get_level_values(level=0).unique():

  asset_df = current_week_df[asset]

  w_window = asset_window(asset_df)

  current_week_data[asset] = w_window.df


X_current = []
for window in current_week_data.values():
    X_current.append(window.iloc[:-1].dropna().values.flatten())

assets = [asset for asset in current_week_df.columns.get_level_values(level=0).unique()]

shapes = [arr.shape[0] for arr in X_current if arr.shape[0] > 0]

expected_shape = input_dim

print(f"Expected shape based on input: {expected_shape}")
while 1:
  j = 0
  for i, arr in enumerate(X_current):
      if arr.shape[0] != expected_shape or np.isinf(arr).any():

        X_current.pop(i)
        assets.pop(i)
        j += 1
  print(f'Removed {j} instances of inconsistencies from current set')
  if j == 0:
    break

X_current = np.array(X_current)

X_current = scaler.transform(X_current)
X_current_latent = encoder.predict(X_current)

predicted_close_price_changes = xgb_model.predict(X_current_latent)

predict_changes_dict = {}
for i, asset in enumerate(assets):
  predict_changes_dict[asset] = predicted_close_price_changes[i]

predictions_df = pd.DataFrame.from_dict(predict_changes_dict, orient='index', columns=['Predicted_Close_Price_Change'])

assets_to_remove = []
for asset in predictions_df.index:
  if not asset.endswith('-USD'):
    assets_to_remove.append(asset)
for asset in assets_to_remove:
  predictions_df
  predictions_df.drop(index=asset, inplace=True)

predictions_df['Volatility'] = 0
for asset in predictions_df.index:
  start_date = crypto_df.index[crypto_df.index.get_loc(day) - 30]
  vol = crypto_df[asset]['Close'].pct_change().loc[start_date:day].rolling(window=90).std().iloc[-1]
  predictions_df.loc[asset, 'Volatility'] = vol



adj_close_df = crypto_df.loc[:, (slice(None), 'Close')].droplevel(1, axis=1)



returns_df = adj_close_df.pct_change()

returns_df = returns_df[-90:]

assets_to_remove = []
for asset in returns_df.columns:
  if not(asset in predictions_df.index):
    assets_to_remove.append(asset)
for asset in assets_to_remove:
  returns_df.drop(columns=asset, inplace=True)



returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
cov_matrix = returns_df.cov()

quad_matrix = cov_matrix.values





R = desired_daily_return
missing_assets = set(predictions_df.index) - set(returns_df.columns)

if missing_assets:
  print(f"Removing missing assets from predictions_df: {missing_assets}")
  predictions_df.drop(index=missing_assets, inplace=True)

r = np.array(predictions_df['Predicted_Close_Price_Change'])


lw = LedoitWolf()
quad_matrix = lw.fit(returns_df).covariance_
quad_matrix_psd = cp.psd_wrap(quad_matrix)

w = cp.Variable(len(r))

objective = cp.Maximize(r.T @ w)
constraints = [
    cp.quad_form(w, quad_matrix_psd) <= 1,
    cp.sum(w) == 1,
    w >= 1e-10,
    r.T @ w >= R,
    w <= weight_limit
    ]
prob = cp.Problem(objective, constraints)
result = prob.solve()

weight_df = pd.DataFrame(w.value, index=predictions_df.index, columns=['Weight'])
total_daily_weight = 0
weight_df.sort_values(by='Weight', ascending=False, inplace=True)

for asset in weight_df.index:
  if weight_df.loc[asset, 'Weight'] < 0.001:
    weight_df.drop(index=asset, inplace=True)
    continue
  total_daily_weight += weight_df.loc[asset, 'Weight']

for asset in weight_df.index:
  weight_df.loc[asset, 'Weight'] /= total_daily_weight



auto_portfolio_uuid = client.get_portfolios()['portfolios']
for portfolio in auto_portfolio_uuid:
  if portfolio['name'] == 'Auto':
    auto_portfolio_uuid = portfolio['uuid']
    break

total_balance = client.get_portfolio_breakdown(auto_portfolio_uuid)['breakdown']['portfolio_balances']['total_balance']['value']
total_balance = round(float(total_balance),2)

allocation_df = {}
positions = client.get_portfolio_breakdown(auto_portfolio_uuid)['breakdown']['spot_positions']
for asset in positions:
  if asset['asset'] == 'USD':
    allocation_df[asset['asset']] = asset['allocation']
    continue
  allocation_df[str(asset['asset'])+'-USD'] = asset['allocation']
allocation_df = pd.DataFrame.from_dict(allocation_df, orient='index', columns=['Allocation'])

combined_allocations_df = pd.concat([weight_df, allocation_df], axis=1)
combined_allocations_df.fillna(0, inplace=True)
combined_allocations_df['Difference'] = combined_allocations_df['Weight'] - combined_allocations_df['Allocation']

combined_allocations_df['Trade'] = combined_allocations_df['Difference'] * total_balance * 0.995
combined_allocations_df.sort_values(by='Trade', ascending=True, inplace=True)


def make_trade(asset, amount):
  order_id = str(uuid.uuid4())
  base_amount = float(client.get_product(asset)['base_increment'])
  if base_amount >= 1:
    round_amount = 0
  else:
    round_amount = int(len(client.get_product(asset)['base_increment']))-2
  cushion_amount = float('0.'+'0'*(round_amount - 1) + '1')
  if amount < 0 - cushion_amount:
    trade_type = 'sell'
    amount = amount + float('0.'+'0'*(round_amount - 1) + '1')
    order_configuration = {
      "market_market_ioc" : {
      "base_size": str(round(abs(amount/float(client.get_product(asset)['price'])), round_amount))}}
  elif amount > 0 + cushion_amount:
    trade_type = 'buy'
    amount = amount - float('0.'+'0'*(round_amount - 1) + '1')
    order_configuration = {
      "market_market_ioc" : {
      "quote_size": str(round(amount,2))}}
  else:
    return
  if abs(amount) < float(client.get_product(asset)['quote_min_size']):
    return
  



  if trade_type == 'buy':
    try:
      return client.create_order(order_id, asset, 'BUY', order_configuration=order_configuration)
    except:
      try:
        order_configuration = {
        "limit_limit_gtd" : {
        "quote_size": str(round(amount, 2)),
        "limit_price" : str(round(float(client.get_product(asset)['price']) * 1.01, 2)),
        "end_time" : (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}}
        return client.create_order(order_id, asset, 'BUY', order_configuration=order_configuration)
      except:
         print('Unknown error, unable to place order')
         return
  elif trade_type == 'sell':
    try:
      return client.create_order(order_id, asset, 'SELL', order_configuration=order_configuration)
    except:
      try:
        order_configuration = {
        "limit_limit_gtd" : {
        "base_size": str(round(abs(amount/float(client.get_product(asset)['price'])),round_amount)),
        "limit_price": str(round(float(client.get_product(asset)['price']) * 0.99, 2)),
        "end_time" : (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}}
        return client.create_order(order_id, asset, 'SELL', order_configuration=order_configuration)
      except:
        print('Unknown error, unable to place order')
        return
  else:
    print(f'Trade type unclear, unable to place order for {asset}')
    return




############################################################
############################################################
#### RUN BLOCK WITH CAUTION --> EXECUTES TRADES#############
############################################################
############################################################

for asset in combined_allocations_df.index:
  if asset == 'USD':
    continue
  difference_val = combined_allocations_df.loc[asset, 'Difference']
  if abs(difference_val) < rebalancing_threshold:
    continue
  make_trade(asset, combined_allocations_df.loc[asset, 'Trade'])

############################################################
iteration_count = 0
while True:
    iteration_count += 1
    if iteration_count > 10:
      break
    portfolio_breakdown = client.get_portfolio_breakdown(auto_portfolio_uuid)['breakdown']
    cash_allocation = float(portfolio_breakdown['portfolio_balances']['total_cash_equivalent_balance']['value'])
    total_balance = float(portfolio_breakdown['portfolio_balances']['total_balance']['value'])
    trade_count = 0

    if cash_allocation <= total_balance * 0.05:
      break

    for asset in combined_allocations_df.index:
        if asset == 'USD':
          continue

        trade_value = combined_allocations_df.loc[asset, 'Trade']
        min_trade_size = float(client.get_product(asset)['quote_min_size'])

        if trade_value < min_trade_size or trade_value < 0:
          continue
        if trade_value > min_trade_size or trade_value > 0:
          trade_count += 1
        if trade_count < 1:
          break
        make_trade(asset, trade_value)

    allocation_df = {}
    positions = portfolio_breakdown['spot_positions']
    for asset in positions:
        if asset['asset'] == 'USD':
            allocation_df[asset['asset']] = asset['allocation']
            continue
        allocation_df[f"{asset['asset']}-USD"] = asset['allocation']
    allocation_df = pd.DataFrame.from_dict(allocation_df, orient='index', columns=['Allocation'])

    combined_allocations_df = pd.concat([weight_df, allocation_df], axis=1)
    combined_allocations_df['Difference'] = combined_allocations_df['Weight'] - combined_allocations_df['Allocation']
    combined_allocations_df['Trade'] = combined_allocations_df['Difference'] * total_balance
    combined_allocations_df.sort_values(by='Trade', ascending=False, inplace=True)

############################################################
############################################################
############################################################
############################################################

print('Code ran successfully, please ensure coinbase has received trades.')
