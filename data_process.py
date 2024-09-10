from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import pandas as pd
import os
import datetime as dt
import pytz
import time as t
from lstm_load_forecasting import data

def data_process_without_norm(feature_data_list=None):
    # Load data and prepare for standardization
    path = os.path.join(os.path.abspath(''), './data/fulldataset.csv')
    features = ['all']

    # Splitdate for train and test data. As the TBATS and ARIMA benchmark needs 2 full cycle of all seasonality, needs to be after jan 01. 
    loc_tz = pytz.timezone('Europe/Zurich')
    split_date = loc_tz.localize(dt.datetime(2017,1,1,0,0,0,0))

    df = data.load_dataset(path=path, modules=features)
    # df = df.drop(['hour_0', 'hour_1', 'hour_2',
    #     'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
    #     'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
    #     'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
    #     'hour_22', 'hour_23'],1)
    df = df.loc['20150101':]

    hour_index = ['hour_0', 'hour_1', 'hour_2','hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9','hour_10', 'hour_11', 'hour_12',
                  'hour_13', 'hour_14', 'hour_15','hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21','hour_22', 'hour_23']
    
    week_index = ['weekday_0', 'weekday_1','weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']
    
                          
    month_index = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6','month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']


    hour_data = np.array(df.loc[:,hour_index])
    hour = []
    for i in hour_data:
        hour.append(np.argmax(i)+1)
    
    week_data = np.array(df.loc[:,week_index])
    week = []
    for i in week_data:
        week.append(np.argmax(i)+1)
    
    month_data = np.array(df.loc[:,month_index])
    month = []
    for i in month_data:
        month.append(np.argmax(i)+1)

    df['hour'] = hour
    df['week'] = week
    df['month'] = month
    df = df.drop(hour_index + week_index + month_index, axis=1)       
    return df
