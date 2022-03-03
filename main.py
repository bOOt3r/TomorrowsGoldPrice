from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

current_date = dt.datetime.now()

df = yf.download("GC=F", "2001-01-01", current_date, auto_adjust=True)

df = df[["Close"]]
df = df.dropna()

df["S_3"] = df["Close"].rolling(window=3).mean()
df["S_9"] = df["Close"].rolling(window=9).mean()
df["next_day_price"] = df["Close"].shift(-1)

df = df.dropna()
X = df[["S_3", "S_9"]]

y = df["next_day_price"]

t = .8
t = int(t*len(df))

X_train = X[:t]
y_train = y[:t]

X_test = X[t:]
y_test = y[t:]

linear = LinearRegression().fit(X_train, y_train)

predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=["price"])

gold = pd.DataFrame()

gold["price"] = df[t:]["Close"]
gold["predicted_price_next_day"] = predicted_price
gold["actual_price_next_day"] = y_test
gold["gold_returns"] = gold["price"].pct_change().shift(-1)

gold["signal"] = np.where(gold.predicted_price_next_day.shift(1) < gold.predicted_price_next_day,1,0)

gold["strategy_returns"] = gold.signal * gold["gold_returns"]
((gold["strategy_returns"]+1).cumprod()).plot(figsize=(10,7),color="g")

data = yf.download("GC=F", "2001-01-01", current_date, auto_adjust=True)
data["S_3"] = data["Close"].rolling(window=3).mean()
data["S_9"] = data["Close"].rolling(window=9).mean()
data = data.dropna()

data["Framtida_guldpris"] = linear.predict(data[["S_3", "S_9"]])
data["signal"] = np.where(data.Framtida_guldpris.shift(1) < data.Framtida_guldpris,"Köp","Håll")

print(data.tail(1)[["signal","Framtida_guldpris"]].T)
