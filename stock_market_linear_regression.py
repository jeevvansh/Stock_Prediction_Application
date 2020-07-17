#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Jeevvansh Lota
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import date
plt.style.use('fivethirtyeight')

print('Welcome! \nThis program will predict the closing stock prices of a certain stock specified by the user \n')
print('It will retrieve the latest financial data from yahoo.com and use the scikit-learn machine learning library to make the predictions\n')
print('80% of the data will be used for training data and the remaining 20% will be predicted\n')
print("After running, the program will produce a graph comparing its predictions and the actual closing prices of the stock. In addtion, it will even make a prediction for tommorow's closing price!")
print("\nObviously this program cannot be used to actually predict stock prices, the purpose of this project is just to demonstarte usage of the pandas, keras, and scikit-learn libraries ")


stock_name = input ("Please enter the abbreviation of the stock to predict: ")

#%matplotlib inline
today = date.today()
df = web.DataReader(stock_name, data_source='yahoo', start='2014-01-01', end= today)
#print(df.describe)
#print(df.shape)

#Visualize closing price history
plt.figure(figsize=(16,8))
plt.title('CLose Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.show()


#Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
#Convert dataframe to numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)
#print(training_data_len)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#print(scaled_data)

#Create the training dataset
#Create the scaled training data set
train_data = scaled_data[0:training_data_len,:]
#Split data into x_train and y_train datasets
x_train = []
y_train = []#dependent

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    #if i<=61:
       # print(x_train)
        #print(y_train)
        #print()
        
        
#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the Data bc LSTM model expects 3 dimensions
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)


#Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))



#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


#Create the testing data set
#Create new array containing scaled values from index 1640 to 2125
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]#values to predict
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data to #D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#unscaling the values


#Evaluate model by getting root mean squared error (RMSE)
#lower value indicates better fit
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print('ROOT MEAN SQUARED ERROR: ',rmse)

#Plot the Data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title(stock_name + ' Stock Prediction Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()

#Show valid and predicted prices
#print(valid)

#Predict the stock for tmrw
stock_quote = web.DataReader(stock_name, data_source='yahoo', start='2014-01-01', end= today)
#Create a new dataframe
new_df = stock_quote.filter(['Close'])
#get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append past 60 days to x_test list
X_test.append(last_60_days_scaled)
#Convert the X_test data set to numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
#print(pred_price)



print('\nThe root mean squared error is an evaluation metric for the measure of differences between values predicted by the model and the actual values. A lower RSME indicates a better fit. ')
print('\nROOT MEAN SQUARED ERROR: ',rmse)
print('\n\n')
print('Here is a comparision of some of the actual closing and predicted values: \n')
print(valid)
print('\n\n')
print("And tomorrow's closing price prediction is: ", pred_price)



