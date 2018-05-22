from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from data import DataSet
import math
import matplotlib.pyplot as plt


def fit_value_by_random_forest():
    catalog = ['time', 'DOC', 'feed', 'material']

    data = DataSet()

    # vb_val = data.vb_value
    # vb_pd_val = pd.DataFrame(vb_val)

    df = data.export_as_pd
    df.dropna()
    null_index_list = []
    for index, list in enumerate(df['VB']):
        if math.isnan(list):
            null_index_list.append(index)
        else:
            pass
            # print(list,type(list))
    print(null_index_list)
    df_drop = df.drop(null_index_list)
    # print(df_drop)

    regression = RandomForestRegressor(n_jobs=4, n_estimators=100, oob_score=True)
    regression.fit(df_drop[catalog], df_drop['VB'])

    preds = regression.predict(df[catalog])
    # plt.plot(preds, label='prediction')
    # plt.plot(df['VB'], label='real')

    # fit data
    for index,value in enumerate(df['VB']):
        if math.isnan(value):
            df['VB'][index] = round(preds[index],2)
            null_index_list.append(index)

    # plt.plot(df['VB'],label='fit_data_with_null')
    # plt.legend()
    # plt.show()
    return df['VB']

if __name__ == '__main__':
    fit_value_by_random_forest()