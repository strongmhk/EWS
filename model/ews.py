import sys
sys.path.append('/Users/hiksang/Documents/GitHub/EWS_b/model')

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

class EWS:
    '''
    Ex)
    ews = EWS('path')
    ews.setup(feature_list, target_col, date_col,  object_col, object_list, missing_dic, using_col)
    ews.run()

    '''
    def __init__(self, path):
        self.trans_df = 'a'
        self.path = path
        self.df, self.meta = 'a', 'b'


    def setup(self, feature_list, target_col, date_col,  object_col, object_list, missing_dic, using_col=None):
        '''
        type: list, feature_list = []  # ex) ['계약건수[건]','시점'] 입력받은 df의 전체 col 리스트
        type: str,  target_col = "지급여력비율" target column
        type: str,  date_col = '시점'  date type을 가지고 있는 column
        type: str,  object_col = '회사별'
        type: list, object_list = []  # ex) ['메리츠','KB']
        type: dic,  missing_dic = {}  # ex) {'시점' : 'drop', '지급여력비율' : 'mean'}
        type: list,using_col = self.feature_list.copy() : feature_list 동일
        type: list,using_col.append(self.target_col)
        '''
        self.feature_list = feature_list
        self.target_col = target_col
        self.date_col = date_col
        self.object_col = object_col
        self.object_list = object_list
        self.using_col = using_col
        self.missing_dic = missing_dic


    def first_page(self, dq_report_path):
        # try:
        self.df = read_data(self.path, 'csv')  # load data
        html_dqreport(self.df, dq_report_path)
        self.metadata = make_meta(self.df)
        # except:
        #     self.df = read_data(self.path, 'excel')  # load data
        #     html_dqreport(self.df, dq_report_path)
        #     self.metadata = make_meta(self.df)
        return self.df, self.metadata

    def trans_df(self):
        '''
        df: dataframe,
        date_col: str,
        target_col: str,
        feature_col: list,
        missing_dic: dic
        '''
        df = trans_date(self.df, self.date_col)
        df = df[self.using_col]
        # 현재는 Missing data는 무조건 drop
        df = missing_df(df)
        prep_visualize(df, 'preprocess')  # preprocess_merged.html으로 저장됨
        self.trans_df = df

        return None

    def prepare_df(self):
        '''
        df : dataframe
        object_col : str
        output : df_feat, df
        '''
        df_feat = self.df.drop(columns=self.object_col)

        return df_feat
    def analyze(self):
        '''
        df : dataframe, prepare_df output df_feat 사용
        target : str
        변경시 경로 수정해야함.
        '''
        df = self.trans_df
        target = self.target_col
        auto_ml_reg_first(df, target)
        df_class = binning(df, target=target)
        auto_ml_cls_first(df=df_class, target=target)

        return None

    def run(self):
        self.trans_df()
        self.analyze()
        return None

