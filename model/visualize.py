from syndata import *
from preprocess import *
from models import *
import autoviz
import os
from autoviz import AutoViz_Class
from pandas_dq import dq_report
from pathlib import Path

def prep_visualize(df, df_type=''):
    '''
    df : df3
    df_type : str ("origin" or "synthetic")
    '''
    AV = AutoViz_Class()

    html_maker = AV.AutoViz(
        filename="",
        sep=",",
        depVar="",
        dfte=df,
        header=0,
        verbose=1,
        lowess=True,
        chart_format="html",
        max_rows_analyzed=150000,
        max_cols_analyzed=30,
        save_plot_dir='./bank_sol/HTML/prep/'
    )
    ## html 병합과정
    folder_path = './bank_sol/HTML/prep/AutoViz/'  # 가져올 폴더의 경로를 지정해주세요
    html_files = [file for file in os.listdir(folder_path) if file.endswith('.html')]
    modified_list = [folder_path + item for item in html_files]

    merged_content = ''
    for file_name in modified_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            merged_content += f.read()

    # 결과를 새로운 파일에 저장
    with open('./bank_sol/HTML/prep/' + f'/{df_type}_merged.html', 'w', encoding='utf-8') as f:
        f.write(merged_content)

    return None

def html_dqreport(data, dq_report_path):
    '''
    data : dataframe

    결과는 dq_report.html로 저장됨
    path : Data/bank_sol/HTML/prep
    추후에 db 설정시 변경해야함.
    '''
    os.chdir(dq_report_path)
    # 경로가 존재하지 않으면 생성
    if not os.path.exists(dq_report_path):
        os.makedirs(dq_report_path)
    report = dq_report(data, html=True, csv_engine='pandas', verbose=1)
    os.chdir('/Users/hiksang/Documents/GitHub/EWS/Data')  # 추후 DB로 변환

    return None



