from explainerdashboard import ExplainerDashboard, RegressionExplainer, ClassifierExplainer
from pycaret.regression import *
from preprocess import *
import panel as pn
import pandas as pd

def reg_dashboard_html(model,base,name):
    X_test_df = model.X_test_transformed.copy()
    y_test = model.y_test_transformed
    X_test_df.columns = [col.replace(".", "__").replace("{", "__").replace("}", "__") for col in X_test_df.columns]
    explainer = RegressionExplainer(base, X_test_df, y_test)
    linear_dash = ExplainerDashboard(explainer, mode='dash')
    linear_dash.save_html(f'./bank_sol/HTML/analysis/reg_{name}.html')
    # linear_dash.save_html(f'./EWS/Data/bank_sol/HTML/analysis/reg_{name}.html')
    return None

def cls_dashboard_html(model,base,name):
    X_test_df = model.X_test_transformed.copy()
    y_test = model.y_test_transformed
    X_test_df.columns = [col.replace(".", "__").replace("{", "__").replace("}", "__") for col in X_test_df.columns]
    explainer = ClassifierExplainer(base, X_test_df, y_test)
    linear_dash = ExplainerDashboard(explainer, mode='dash')
    linear_dash.save_html(f'./bank_sol/HTML/analysis/cls_{name}.html')
    # linear_dash.save_html(f'./EWS/Data/bank_sol/HTML/analysis/cls_{name}.html')
    return None
