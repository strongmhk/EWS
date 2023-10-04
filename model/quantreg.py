from preprocess import *
import matplotlib
matplotlib.use('TkAgg')
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = read_data('KVALL.WD.csv', 'csv')
df1, df2, df3 = order_df(df)
print(df1,df2,df3)

def quantreg(df,target):
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

quantreg(df3)