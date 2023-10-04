import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from IPython.display import display

## pycaret
import pycaret.regression as reg
import pycaret.classification as cls

## FinBERT
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

## SDV
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.sampling import Condition
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import get_column_pair_plot
from sdv.sequential import PARSynthesizer
from sdv.sampling import Condition
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.utils import get_column_plot
from sdmetrics.reports.utils import get_column_plot

## LSTM
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import keras_tuner
from keras import layers

from syndata import *
from preprocess import *
from models import *
from visualize import *

seed_num = 42
np.random.seed(seed_num)
random.seed(seed_num)
tf.random.set_seed(seed_num)
tf.keras.utils.set_random_seed(seed_num)

bank_lists = ["BANK #1"] ## server
target = "KV001"  ## server

''' Metadata '''
# df = read_data('KVALL.WD.csv', 'csv') # load data
# # df = read_data('C:/Users/siyou/OneDrive/바탕 화면/학교/Dragonfly/EWS(23-여름)/data/KVALL.WD.csv', 'csv')
# metadata = make_meta(df)
# print(metadata)

''' prep df generating '''
df = read_data('KVALL.WD.csv', 'csv')
# df = read_data('C:/Users/siyou/OneDrive/바탕 화면/학교/Dragonfly/EWS(23-여름)/data/KVALL.WD.csv', 'csv')
df1, df2, df3 = order_df(df)
print(df1,df2,df3)

'''classification_df'''
## target_class, bank는 클라이언트가 선택하는 값에 따라 적용됨
# df_class = binning(select_bank(df2, bank_lists,"delete"), target = target)
# print(f'df_class: {df_class}')

''' prep LSTM '''
# x_train, x_val, x_test, y_train, y_val, y_test, scalar_target = lstm_train_test(df2, bank_lists, target)
# print(f'x_train shape {x_train.shape[2]}')
# print(f'y_train shape {y_train.shape}')

''' syndata'''
# syn_df1 = sequential_gen(select_bank(df1, bank_lists,'origin'), bank_list=bank_lists, bank_num=1, seq_len=30 )
# syn_df2 = feature_gen(select_bank(df2, bank_lists,'origin'),gen_num= 100,case='gaussian')
# syn_df3 = feature_gen(df3, 100, 'gaussian')
# print(syn_df1, syn_df2, syn_df3)
#
# ''' Prep visualize'''
# prep_visualize(df1, 'origin')
# prep_visualize(syn_df1, 'synthetic')
# prep_visualize(select_bank(df1, ['BANK #1'], 'origin'),'bank_1')

# ''' analysis '''
# first_anal = auto_ml_reg_first(df3,target=target) ## html 생성 or dashboard url
# second_anal = auto_ml_cls_first(df = df_class, target = target)
# quantreg(df3,target) ## quantile reg
lstm_model(df1,df2,['BANK #1'],'KV001',1) # 튜닝X

'''
Task 1. LSTM 튜닝 & 시각화
Task 2. Transformer forecast 
Task 3. NLP(한국은행 뉴스심리지수 벤치마크) <- 시영
Task 4. dashboard layout to make HTML file
Task 5. django server
'''
