from explainerdashboard import ExplainerDashboard, RegressionExplainer, ClassifierExplainer
from pycaret.regression import *
from preprocess import *
import panel as pn
import pandas as pd

# df = read_data('KVALL.WD.csv', 'csv')
# # df = read_data('C:/Users/siyou/OneDrive/바탕 화면/학교/Dragonfly/EWS(23-여름)/data/KVALL.WD.csv', 'csv')
# df1, df2, df3 = order_df(df)
# print(df1,df2,df3)

# pn.extension()
# df_pane = pn.pane.DataFrame(df, width=400)
# df_pane

import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np

pn.extension() # for notebook

pn.pane.Markdown('''
# H1
## H2
### H3
''')