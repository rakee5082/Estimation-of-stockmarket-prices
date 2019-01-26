#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import describe
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from numpy.random import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from copy import deepcopy
from pandas import DataFrame
from pandas import concat
import pickle

def read_data(obj):
    ## get ticker data
    ticker = pd.read_csv('data/'+obj+'_ticker.csv', index_col=0, header=None)
    ticker.index = pd.to_datetime(ticker.index, unit='s')
    ticker.index.name = None
    ticker.columns = [
        'sell_highest_price', 
        'sell_highest_vol', 
        'buy_highest_price', 
        'buy_highest_vol', 
        'last_trade_price', 
        'daily_trade_vol', 
        'daily_high_price', 
        'daily_low_price'
    ]
    ticker.values[np.where(ticker.values <= 0)] = np.nan
    ticker = ticker.interpolate()
    ## trades data
    trades = pd.read_csv('data/'+obj+'_trades.csv', index_col=0, header=None) 

    ## get book data
    book = pd.read_csv('data/'+obj+'_book.csv', index_col=0, header=None)    
    ticker = ticker.add_suffix(obj)
    return ticker,trades,book

#### read data and set it to the dictionary:
########################################################################3
metals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
ticker = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[]}
trades = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[]}
book = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[]}
for currency in metals:
    ticker[currency],trades[currency],book[currency] = read_data(currency)
    
##### load trained models    
with open('my_model.pkl', 'rb') as handle:
    models = pickle.load(handle)
    
def series_to_supervised(data,columns,index, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [(columns[j]+'(t-%d)' % (i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [(columns[j]+'(t)')for j in range(n_vars)]
		else:
			names += [(columns[j]+'(t+%d)' % (i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	agg.index = index
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def predict(cur_met):
    ########## get statistics of each 5 min in vopr - price and vol trade 
    trades[cur_met].replace(-1,np.NaN)
    temp_price_trade = trades[cur_met].iloc[:,3::4]
    temp_vol_trade = trades[cur_met].iloc[:,2::4]
    temp_vopr_trade = pd.DataFrame(trades[cur_met].iloc[:,3::4].values * trades[cur_met].iloc[:,2::4].values)

    temp = np.sum(np.absolute(temp_vopr_trade.get_values()),axis=1) / np.sum(np.absolute(temp_vol_trade.get_values()),axis=1)
    absvopr = {'absvopr' : temp}
    absvopr = pd.DataFrame.from_dict(absvopr)



    K = 10
    ### set index of dataframe to time and merger data 
    absvopr.index = ticker[cur_met].index
    ### merge step ticker and trade
    imp_ticker = ticker[cur_met].iloc[:,[0,2,4,7]]    
    #stock = pd.merge(ticker,vopr_trade, left_index=True, right_index=True)
    stock = pd.DataFrame(index=ticker[cur_met].index)
    stock = pd.merge(imp_ticker,absvopr, left_index=True, right_index=True) ## y
    #stock = pd.merge(stock,vol_trade, left_index=True, right_index=True) ## x
    other_stock = pd.DataFrame(index=ticker[cur_met].index)
    for currency in metals:
        if(currency!=cur_met):
            ticker[currency].index = ticker[cur_met].index
            temp_tic = ticker[currency].iloc[:,:]
            other_stock = pd.merge(other_stock,temp_tic, left_index=True, right_index=True)

    all_feat = [u'sell_highest_price',
     u'buy_highest_price',
     u'last_trade_price',]
    all_feat = [x+cur_met for x in all_feat]
    all_feat.append(u'absvopr')

    #list(range(stock.shape[1]))#[0,2,4,7] #+ [9,10,14,16,17,21,23,24,28]
    temp_stock = stock.loc[:,all_feat];
    data_stock_raw = series_to_supervised(temp_stock.values,temp_stock.columns.get_values(),temp_stock.index,25, 11)
    #other_data_raw = series_to_supervised(other_stock.values,other_stock.columns.get_values(),other_stock.index,0, 1)
    other_data_raw = other_stock
    #data_diffren = data_diff(temp_stock.values,temp_stock.columns.get_values(),temp_stock.index,20)

    #### set feature name 
    input_feat = temp_stock.columns
    past_feat = [0,1,2,10]
    input_names = []
    for i in past_feat:
        if (i==0):
            input_names += [(input_feat[j]+'(t)')for j in range(len(input_feat))]
        else:
             input_names += [(input_feat[j]+'(t-%d)' % (i))for j in range(len(input_feat))]

    output_feat = temp_stock.columns[2]
    fut_feat = list(range(10))
    fut_feat = []
    for i in range(1,11):
         temp = []
         temp.append(output_feat+'(t+%d)' % (i))
         fut_feat.append(temp)

    data =  pd.concat([other_data_raw, data_stock_raw.loc[:,input_names]], axis=1, join='inner')
    poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    X_t = data.loc[:,input_names].get_values()
    X_t = StandardScaler().fit_transform(X_t)  
    #print(X_t.shape)
    X_t = X_t[-1,:]
    X_t = poly.fit_transform(X_t)
    temp_model = models[cur_met]
    pr = temp_model.predict(X_t.reshape(1,-1))
    return pr[0]

    
#### make output
n = int(input())

for i in range(n):
    for currency in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        a = ticker[currency].columns.tolist()
        b = trades[currency].columns.tolist()
        c = book[currency].columns.tolist()
        new_ticker = input()
        new_trade = input()
        new_book = input()
        temp_tick = new_ticker.split(',')
        temp_trade = new_trade.split(',')
        temp_book = new_book.split(',')
        dftic = pd.DataFrame(np.array(temp_tick[1:],dtype=float).reshape(1,-1),columns=a)
        dftr =  pd.DataFrame(np.array(temp_trade[1:],dtype=float).reshape(1,-1),columns=b)
        dfbook = pd.DataFrame(np.array(temp_book[1:],dtype=float).reshape(1,-1),columns=c)
        ### add to previous dataframe
        ticker[currency] = ticker[currency].append(dftic, ignore_index=True)
        trades[currency] = trades[currency].append(dftr, ignore_index=True)
        book[currency] = book[currency].append(dfbook, ignore_index=True)
    #print (ticker[currency].shape)
        


    for currency in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        temp_pr = np.zeros(10)
        temp_pr = predict(currency)     
        for i in range(10):
            print(temp_pr[i],end=' ')
        print()    
