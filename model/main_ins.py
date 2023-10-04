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

path = 'final.csv'
'''Initiating process'''
def first_page(path):
    try:
        df = read_data(path, 'csv') # load data
        html_dqreport(df)
        metadata = make_meta(df)
    except :
        df = read_data(path, 'excel')  # load data
        html_dqreport(df)
        metadata = make_meta(df)
    return df, metadata


object_col = '회사별'
object_list = [] # ex) ['메리츠','KB']
date_col = '시점'
target_col = "지급여력비율"
feature_list = [] #ex) ['계약건수[건]','시점']
missing_dic = {}  # ex) {'시점' : 'drop', '지급여력비율' : 'mean'}
using_col = feature_list.copy()
using_col.append(target_col)

''' transform & data vizualization'''
def trans_df(df, date_col, using_col, target_col=None,  missing_dic=None ):
    '''
    df : dataframe,
    date_col : str,
    target_col : str,
    feature_col : list,
    missing_dic : dic
    '''
    df = trans_date(df, date_col)
    df = df[using_col]
    # 현재는 Missing data는 무조건 drop
    df = missing_df(df)
    prep_visualize(df, 'preprocess')  # preprocess_merged.html으로 저장됨

    return df

def prepare_df(df, object_col):
    '''
    df : dataframe
    object_col : str
    output : df_feat, df
    '''
    df_feat = df.drop(columns = object_col)

    return df_feat, df

def analyze(df, target):
    '''
    df : dataframe, prepare_df output df_feat 사용
    target : str
    '''
    auto_ml_reg_first(df, target)
    df_class = binning(df, target = target)
    auto_ml_cls_first(df = df_class, target = target)

    return None


## 초기화면
df, meta = first_page(path)
## 입력받은 정보로 진행
df = trans_df(df, date_col,using_col)
df_feat, df = prepare_df(df, object_col)
analyze(df_feat, target_col)