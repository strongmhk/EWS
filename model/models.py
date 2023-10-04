import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
import math

## pycaret
import pycaret.regression as reg
import pycaret.classification as cls

## LSTM
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import keras_tuner
from keras import layers

import socket
from preprocess import *
from dashboard import *
import matplotlib
matplotlib.use('TkAgg')
import statsmodels.formula.api as smf
# visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go



## Regression
def auto_ml_reg(df, target, bank_name):
    df = df[df['bank.code'] == bank_name]
    df = df.drop(columns=['bank.code'])
    model = reg.setup(df, target = target, session_id= 42)
    best = reg.compare_models()
    
    linear_reg = reg.create_model('lr', cross_validation=True)
    ridge_reg = reg.create_model('ridge', cross_validation=True)
    xgb_reg = reg.create_model('xgboost', cross_validation = True)
    lgb_reg = reg.create_model('lightgbm', cross_validation = True)
    
    linear_port = 9000
    XGBoost_port = 9010
    lightGBM_port =9020
    GBM_port =9030
    ridge_port = 9040
    
    ## dashboard of linear
    reg.dashboard(linear_reg, run_kwargs={'port':linear_port})#, 'host':'0.0.0.0'})
    linear_address = make_localhost(linear_port)
    
    ## dashboard of ridge
    reg.dashboard(ridge_reg, run_kwargs={'port':ridge_port})#, 'host':'0.0.0.0'})
    ridge_address = make_localhost(linear_port)
    
    ## dashboard of xgb
    reg.dashboard(xgb_reg, run_kwargs={'port':XGBoost_port})#, 'host':'0.0.0.0'})
    xgb_address = make_localhost(linear_port)
    
    ## dashboard of lightgbm
    reg.dashboard(lgb_reg, run_kwargs={'port':lightGBM_port})#, 'host':'0.0.0.0'})
    lightgbbm_address = make_localhost(linear_port)
    
    address_dict = {'lr' : linear_address, 'ridge': ridge_address, 'xgboost' : xgb_address, 'lightgbm' : lightgbbm_address}
    return address_dict

def auto_ml_reg_first(df, target):
    
    model = reg.setup(df, target = target, session_id= 42)
    # best = reg.compare_models()
    
    linear_reg = reg.create_model('lr', cross_validation=True)
    reg_dashboard_html(model,linear_reg,"lr")
    ridge_reg = reg.create_model('ridge', cross_validation=True)
    reg_dashboard_html(model, ridge_reg,"ridge")
    xgb_reg = reg.create_model('xgboost', cross_validation = True)
    reg_dashboard_html(model, xgb_reg,'XGB')
    lgb_reg = reg.create_model('lightgbm', cross_validation = True)
    reg_dashboard_html(model, lgb_reg, "Lightgbm")
    
    # linear_port = 9050
    # XGBoost_port = 9060
    # lightGBM_port =9070
    # GBM_port =9080
    # ridge_port = 9090
    
    ## dashboard of linear
#     reg.dashboard(linear_reg, run_kwargs={'port':linear_port})#, 'host':'0.0.0.0'})
#     linear_address = make_localhost(linear_port)
#
#     ## dashboard of ridge
#     reg.dashboard(ridge_reg, run_kwargs={'port':ridge_port})#, 'host':'0.0.0.0'})
#     ridge_address = make_localhost(linear_port)
#
#     ## dashboard of xgb
#     reg.dashboard(xgb_reg, run_kwargs={'port':XGBoost_port})#, 'host':'0.0.0.0'})
#     xgb_address = make_localhost(linear_port)
#
#     ## dashboard of lightgbm
#     reg.dashboard(lgb_reg, run_kwargs={'port':lightGBM_port})#, 'host':'0.0.0.0'})
#     lightgbbm_address = make_localhost(linear_port)
#
#     address_list = [
#      {
#           "name" : "lr",
#           "url" : linear_address
#      },
#      {
#           "name" : "ridge",
#           "url" : ridge_address
#      },
#      {
#           "name" : "xgboost",
#           "url" : xgb_address
#      },
#      {
#           "name" : "lightgbm",
#           "url" : lightgbbm_address
#      }
# ]
#     return address_list
    return None

# def lstm_model(df,bank_lists, target):  # V1
#     '''
#     df : pd.DataFrame (df2)
#     bank_lists : list (e.g. ['BANK #1']
#     target : str "KV001"
#     '''
#     ## data prep
#     x_train, x_val, x_test, y_train, y_val, y_test, scalar_target = lstm_train_test(df,bank_lists,target)
#
#     ## lstm model
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=(1,x_train.shape[2])))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(x_train, y_train,
#               validation_data=(x_val, y_val),
#               epochs=100)
#     pred = model.predict(x_test)
#     print(pred,y_test)
#
#     rescaled_actual = scalar_target.inverse_transform(y_test.reshape(-1,1))
#     rescaled_pred = scalar_target.inverse_transform(pred.reshape(-1,1))
#     print(rescaled_actual, rescaled_pred)
#
#
#     # inverse_transformered_X = scalar.inverse_transform(transformed_X)
#
#     ## hp tuning
#
#     '''trace1 = go.Scatter(
#         x=rescaled_pred.index,
#         y=stock_data['Close'],
#         mode='lines',
#         name='Actual Price'
#     )
#
#     # Create a trace for the predicted data
#     trace2 = go.Scatter(
#         x=valid.index,
#         y=valid['Predictions'],
#         mode='lines',
#         name='Predicted Price'
#     )
#
#     # Define the layout
#     layout = go.Layout(
#         title='Stock Price Prediction using LSTM',
#         xaxis={'title': 'Date'},
#         yaxis={'title': 'Close Price USD'}
#     )
#     # Create a Figure and plot the graph
#     fig = go.Figure(data=[trace1, trace2], layout=layout)
#     fig.show()
#    '''
#     return None


def lstm_model(df1, df2, bank_lists, target, timestep):  # V2
    '''
    df1 : pd.DataFrame (df1)
    df2 : pd.DataFrame (df2)
    bank_lists : list (e.g. ['BANK #1']
    target : str (e.g. "KV001")
    timestep : int (e.g. 1, 2,,,,)
    '''
    ## data prep
    x_train, x_val, x_test, y_train, y_val, y_test, scaler_feature, scaler_target, date = lstm_train_test(df1, df2,
                                                                                                          bank_lists,
                                                                                                          target,
                                                                                                          timestep=timestep)

    ## lstm model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(timestep, x_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              verbose=0,
              epochs=100)
    pred = model.predict(x_test)

    rescaled_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    rescaled_pred = scaler_target.inverse_transform(pred.reshape(-1, 1))
    # predict
    df_pred = pd.DataFrame(rescaled_pred, columns=['pred'])
    pred_date = date[-len(rescaled_pred):]

    ''' hyperparameter tuning'''

    # Visualization

    fig = go.Figure()

    # 첫 번째 데이터셋 추가
    fig.add_trace(
        go.Scatter(x=date, y=select_bank(df2, bank_lists, 'delete')[target], mode='lines+markers', name='Actual'))

    # 두 번째 데이터셋 추가
    fig.add_trace(go.Scatter(x=pred_date, y=df_pred['pred'], mode='lines+markers', name='predict'))

    # 레이아웃 설정
    fig.update_layout(title=f'{target} Time Series Forecasting with LSTM',
                      xaxis_title='Date',
                      yaxis_title='Value')

    # 그래프 보여주기 & 저장
    fig.show()
    fig.write_html('./bank_sol/HTML/analysis/lstm_figure.html')

    return None
## classification 
## preprocess.py의 binning()을 통해 df 전처리 후 밑의 기능 수행 가능.
def auto_ml_cls_first(df, target):
    model = cls.setup(data = df, target = target, train_size = 0.8)
   
    fold_value = math.floor((df[target].value_counts().min())/2)

    # best = cls.compare_models(fold = fold_value)
    
    xgb_cls = cls.create_model('xgboost',cross_validation=True, fold = fold_value)
    cls_dashboard_html(model,xgb_cls,"XGB")
    # lightgbm_cls = cls.create_model('lightgbm',cross_validation=True, fold = fold_value)
    # cls_dashboard_html(model,lightgbm_cls,"Lightgbm")
    lr_cls = cls.create_model('lr',cross_validation=True, fold = fold_value)
    cls_dashboard_html(model,lr_cls,"lr")
    

    xgb_port = 9010
    lightgbm_port =9020
    logistic_port = 9030    
    
    # ## dashboard of xgboost
    # cls.dashboard(xgb, run_kwargs={'port':xgb_port})#, 'host':'0.0.0.0'})
    # xgboost_address = make_localhost(xgb_port)
    
    # ## dashboard of lightgbm
    # cls.dashboard(lightgbm, run_kwargs={'port':lightgbm_port})#, 'host':'0.0.0.0'})
    # lightGBM_address = make_localhost(lightgbm_port)
    
    # ## dashboard of logitic
    # cls.dashboard(lr, run_kwargs={'port':logistic_port})#, 'host':'0.0.0.0'})
    # logistic_address = make_localhost(logistic_port)
    
    # address_classification = [
    #     {
    #         'name' : 'lr',
    #         'url' : logistic_address
    #     },
    #     {
    #         'name' : 'xgb',
    #         'url' : xgboost_address
    #     },
    #     {
    #         'name' : 'lightGBM',
    #         'url' : lightGBM_address
    #     }
    # ]
    # return address_classification
    return None

def quantreg(df):
    '''
    df : DataFrame
    feature dataframe 만 사용(df3)
    첫행이 Target, 나머지행이 feature

    '''
    y_target = df.columns[0]
    x_columns = df.columns[1:]
    formula = y_target + " ~ " + ' + '.join(x_columns)

    quantiles = [0.25, 0.5, 0.75]  # 원하는 분위수 설정

    results = []
    for q in quantiles:
        model = smf.quantreg(formula, df)
        result = model.fit(q=q)
        results.append(result)

# 결과 출력
    for q, result in zip(quantiles, results):
        print(f"Quantile: {q}")
        print(result.summary())
        plt.figure(figsize=(10, 6))
        plt.scatter(df[y_target], result.fittedvalues, color='blue', alpha=0.5)
        plt.xlabel('Actual')
        plt.ylabel('Fitted')
        plt.title(f'Quantile Regression - Quantile: {q}')
        plt.plot([min(df[y_target]), max(df[y_target])], [min(df[y_target]), max(df[y_target])], color='red', linestyle='--')
        plt.grid(True)
        plt.show()
        print("=" * 80)