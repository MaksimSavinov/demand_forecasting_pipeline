import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from datetime import datetime
from datetime import timedelta
import sys
from copy import deepcopy
import warnings, pylab
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings('ignore')

# setting of the plotting style, registers pandas date converters for matplotlib and the default figure size
import seaborn as sns
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc("figure", figsize=(25, 15))
sns.mpl.rc("font", size=14)


def forecast_ml(abt : pd.DataFrame) -> (pd.DataFrame):
    """
    ML forecasting.
    
    ML model for forecasting demand. Method: XGBoost. granularity SKU-location-day
    
    Parameters
    ----------
    abt : pd.DataFrame
        Input table containing the features for training forecasting model like cost, lag features and so on.
        
    Returns
    -------
    pd.DataFrame
        ML forecast value
    
    """

    abt['PERIOD_DT'] = pd.to_datetime(abt['PERIOD_DT'])
    abt = abt.sort_values(by=['PERIOD_DT']).reset_index(drop = True)
    # train/test split
    train, test= np.split(abt, [int(.67 *len(abt))])
    X, y = train.reset_index(drop = True), train[['TGT_QTY']].reset_index(drop = True)
    X_test, y_test = test.reset_index(drop = True), test[['TGT_QTY']].reset_index(drop = True)
    # drop keys
    feature_names = X.loc[:, (X.columns != 'PRODUCT_ID') & (X.columns != 'LOCATION_ID') & (X.columns != 'RETURNS_AMOUNT') & (X.columns != 'TGT_QTY') &(X.columns != 'RETURNS_QTY') & (X.columns != 'PERIOD_DT') & (X.columns != 'PROMO_ID') & (X.columns != 'MODIFIED_DTTM')& (X.columns != 'SALES_AMOUNT')].columns
    X_train, X_val, y_train, y_val = train_test_split(X[feature_names], y, \
                                    test_size=0.2, random_state=42)
    # training params
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    dval  = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
    evals = [(dtrain, "train"), (dval, "validation")]
    dtest = xgb.DMatrix(X_test[feature_names])
    dall = xgb.DMatrix(abt[feature_names])
    params = {"objective": "reg:squarederror"}
    # training model with eval stop
    model = xgb.train(params, dtrain, num_boost_round = 100, evals=evals,
       verbose_eval=5,
       early_stopping_rounds=10)
    X_test['y_pred'] = model.predict(dtest)
    X_test['y_pred'] = np.clip(X_test['y_pred'], a_min = 0, a_max = None)
    abt['FORECAST_VALUE'] = model.predict(dall)
    abt['FORECAST_VALUE']  = np.clip(abt['FORECAST_VALUE'] , a_min = 0, a_max = None)
    abt['ASSORTMENT_TYPE'] = np.random.choice(['new', 'old'], abt.shape[0])
    abt['DEMAND_TYPE'] = np.where(abt['PROMO_FLG'] == 1, 'promo', 'regular')
    abt['CUSTOMER_ID'] = abt['LOCATION_ID'].astype('str').str[:5]
    return abt[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'PERIOD_DT', 'ASSORTMENT_TYPE', 'DEMAND_TYPE', 'FORECAST_VALUE']]


def forecast_ts(abt : pd.DataFrame) -> (pd.DataFrame):
    """
    TS forecasting.
    
    TS model for forecasting demand. Method: ARIMA (ARMA), horizont of forecasting 12 weeks, granularity SKU-location town-week.
    
    Parameters
    ----------
    abt : pd.DataFrame
        Input table containing the timeperiods and sales quantity for training model
        
    Returns
    -------
    pd.DataFrame
        TS forecast value
    """    
    
    abt['PERIOD_DT'] = pd.to_datetime(abt['PERIOD_DT'])
    # split data by weeks 
    abt = abt.sort_values(by=['PERIOD_DT']).reset_index(drop = True)
    abt['week'] = pd.to_datetime(abt.PERIOD_DT.values.astype('datetime64[W]'))
    # creating town feature
    abt['town'] = abt['LOCATION_ID'].astype('str').str[:5]
    # aggregate data by weeks with creating mean sales qty feature
    sales = abt.groupby(['PRODUCT_ID', 'town', 'week']).agg(mean_sent=('TGT_QTY', 'mean')).reset_index()
    pairs = abt[['PRODUCT_ID', 'town']].groupby(['PRODUCT_ID', 'town']).apply(list).reset_index()
    df_ans_ts = pd.DataFrame(columns = {'week','PRODUCT_ID', 'town', 'mean_sent','FORECAST_VALUE'})
    # pipeline for training and forecasting model for each town
    for i in range(pairs.shape[0]):
        product = pairs.iloc[[i]].PRODUCT_ID
        town = pairs.iloc[[i]].town
        temp_sales = sales[(sales['PRODUCT_ID']==product.values[0]) & (sales['town']==town.values[0])]
        temp_sales = temp_sales.set_index('week').sort_index()
        temp_sales =  temp_sales.resample("W").last()
        temp_sales = temp_sales.fillna(0)
        src_data_model = temp_sales['mean_sent'][:-12]
        model = sm.tsa.ARMA(src_data_model, order=(1,0), freq='W').fit()
        pred = model.predict(end = pd.Timestamp(temp_sales.tail(1).index.tolist()[0]), typ='levels')
        df = pd.DataFrame(pred, columns = {'FORECAST_VALUE'})
        temp_sales  = temp_sales.merge(df, how = 'inner', left_index = True, right_index = True)
        temp_sales.reset_index(inplace = True)
        temp_sales.rename(columns = {'index':'week'},inplace = True)
        df_ans_ts = pd.concat([df_ans_ts, temp_sales])
    df_ans_ts['SEGMENT_NAME'] = 'Other'
    df_ans_ts = df_ans_ts[df_ans_ts['town']!=0]
    df_ans_ts.drop(columns = {'mean_sent'})
    df_ans_ts.rename(columns = {'town': 'CUSTOMER_ID', 'week': 'PERIOD_DT'}, inplace = True)
    return df_ans_ts[['PRODUCT_ID', 'CUSTOMER_ID', 'PERIOD_DT', 'SEGMENT_NAME', 'FORECAST_VALUE']]




