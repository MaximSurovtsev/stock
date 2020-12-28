import math 
import pandas_datareader as web 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
# from alpha_vantage.timeseries import TimeSeries
from yahooquery import Ticker
import os 
from datetime import datetime
import json 
from dateutil import parser
import math
# ALPHA_VANTAGE_API_KEY = 'NF3EOXF0XQTL2VG6Y'

def train(ticket):
	# ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

	# df, data_info = ts.get_intraday(ticket, outputsize='full', interval='5min')

	# df = df[::-1]
	ticket = ticket.lower()
	if not os.path.exists('cache.json'):
		cache = []
	else:
		with open('cache.json') as file:
			cache = json.load(file)

	for item in cache:
		if item['ticket'] == ticket and (datetime.now() - parser.parse(item['time'])).seconds < 3600:
			return

	ticker = Ticker(ticket, asynchronous=True)
	df = ticker.history(period='5d', interval='1m')
	
	data = df.filter(['close'])
	print(data)
	dataset = data.values
	training_data_len = math.ceil(len(dataset) * .8)

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(dataset)

	train_data = scaled_data[0:training_data_len:]
	x_train = []
	y_train = []

	for i in range(60, len(train_data)):
	    x_train.append(train_data[i-60:i, 0])
	    y_train.append(train_data[i, 0])


	x_train, y_train = np.array(x_train), np.array(y_train)
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],  1))

	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
	model.add(LSTM(50, return_sequences=False))
	model.add(Dense(25))
	model.add(Dense(1))


	model.compile(optimizer='adam', loss='mean_squared_error')

	model.fit(x_train, y_train, batch_size=1, epochs=1)
	model.save(f'/Users/maximsurovtsev/Stocks prediction/{ticket}')
	
	with open('cache.json', 'w') as file:
		cache.append({
			'ticket': ticket,
			'time': str(datetime.now())
		})
		json.dump(cache, file, indent=4) 




def magic(ticket):
	ticket = ticket.lower()
	model = load_model(f'/Users/maximsurovtsev/Stocks prediction/{ticket}')
	# ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
	ticker = Ticker(ticket, asynchronous=True)
	new_data = ticker.history(period='5d', interval='1m')

	# new_data, data_info = ts.get_intraday(ticket, outputsize='full', interval='5min')

	# new_data = new_data[::-1].filter(['4. close'])
	new_data = new_data.filter(['close'])
	last_60 = new_data[-60:].values
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit_transform(new_data)
	last_60 = scaler.transform(last_60)

	scaled = np.array([last_60])
	scaled = np.reshape(scaled, (scaled.shape[0], scaled.shape[1], 1))
	pred_price = scaler.inverse_transform(model.predict(scaled))
	return round(float(pred_price[0][0]), 3)

