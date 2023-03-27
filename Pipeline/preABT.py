
import pandas as pd
import numpy as np
import sqlite3
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from ipywidgets import IntProgress
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category



def percentile(n):
    '''Calculate n - percentile of data'''
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'pctl%s' % n
    return percentile_


def fill_missing_dates(x, date_col):

    
    min_date, max_date = x[date_col].min(), x[date_col].max()
    groupby_day = x.groupby(pd.PeriodIndex(x[date_col], freq='D'))
    results = groupby_day.sum(min_count=1).sort_values(by=date_col)
    return results

    idx = pd.period_range(min_date, max_date)
    results = results.reindex(idx, fill_value=np.nan)

    results.index.rename(date_col, inplace=True)

    return results


def calc_preag_fill(data, group_col, date_col, target_cols, preagg_method):
    ## calc preaggregation
    data_preag = data.groupby(group_col).agg(
        preagg_method)[target_cols].reset_index().sort_values(by=date_col)



    ## fill missing dates
    data_preag_filled = data_preag.groupby(group_col[:-1]).apply(
        fill_missing_dates, date_col=date_col).drop(group_col[:-1],
                                                    axis=1).reset_index()

    ## return DataFrame with calculated preaggregation and filled missing dates
    return data_preag_filled




def calc_ewm(data_preag_filled, group_col, date_col, span):
    ## calc ewm stats

    lf_df_filled = data_preag_filled.groupby(group_col[:-1]).        apply(lambda x: x.set_index(date_col).ewm(span=span).mean()).drop(group_col[:-1], axis=1)

    ## return DataFrame with rolled columns from target_vars
    return lf_df_filled

    
def shift(lf_df_filled, group_col, date_col, lag):

    lf_df = lf_df_filled.groupby(
        level=group_col[:-1]).apply(lambda x: x.shift(lag)).reset_index()
    lf_df[date_col] = pd.to_datetime(lf_df[date_col].astype(str))

    ## return DataFrame with following columns: filter_col, id_cols, date_col and shifted stats
    return lf_df


def calc_rolling(data_preag_filled, group_col, date_col, method, w):

    ## calc rolling stats
    lf_df_filled = data_preag_filled.groupby(group_col[:-1]).        apply(lambda x: x.set_index(date_col).rolling(window=w, min_periods=1).agg(method)).drop(group_col[:-1], axis=1)

    ## return DataFrame with rolled columns from target_vars
    return lf_df_filled

def day_features(result2):
    result2["weekday"] = result2.period_dt.dt.weekday
    result2["monthday"] = result2.period_dt.dt.day
    result2['is_weekend'] = result2.weekday.isin([5,6])*1

    return result2


def generate_lagged_features(
        data: pd.DataFrame,
        target_cols: list = ['Demand'],
        id_cols: list = ['SKU_id', 'Store_id'],
        date_col: str = 'Date',
        lags: list = [7, 14, 21, 28],
        windows: list = ['7D', '14D', '28D', '56D'],
        preagg_methods: list = ['mean'],
        agg_methods: list = ['mean', 'median', percentile(10), pd.Series.skew],
        dynamic_filters: list = ['weekday', 'Promo'],
        ewm_params: dict = {'weekday': [14, 28], 'Promo': [14, 42]}) -> pd.DataFrame:
    
    ''' 
    data - dataframe with default index
    target_cols - column names for lags calculation
    id_cols - key columns to identify unique values
    date_col - column with datetime format values
    lags - lag values(days)
    windows - windows(days/weeks/months/etc.),
        calculation is performed within time range length of window
    preagg_methods - applied methods before rolling to make
        every value unique for given id_cols
    agg_methods - method of aggregation('mean', 'median', percentile, etc.)
    dynamic_filters - column names to use as filter
    ewm_params - span values(days) for each dynamic_filter
    '''

    data = data.sort_values(date_col)
    out_df = deepcopy(data)
    dates = [min(data[date_col]), max(data[date_col])]

    total = len(target_cols) * len(lags) * len(windows) * len(preagg_methods) * len(agg_methods) * len(dynamic_filters)
    progress = IntProgress(min=0, max=total)
    display(progress)

    for filter_col in dynamic_filters:
        group_col = [filter_col] + id_cols + [date_col]

        for lag in lags:
            for preagg in preagg_methods:
                data_preag_filled = calc_preag_fill(data, group_col, date_col,
                                                    target_cols, preagg)
                ## add ewm features
                for alpha in ewm_params.get(filter_col, []):

                    #print("%s %s %s %s" % (filter_col, lag, preagg, alpha))
                    ewm_filled = calc_ewm(data_preag_filled, group_col,
                                          date_col, alpha)
                    
                    ewm = shift(ewm_filled, group_col, date_col, lag)
                    new_names = {x: "{0}_lag{1}d_alpha{2}_{3}".                        format(x, lag, alpha, filter_col) for x in target_cols}

                        


                    out_df = pd.merge(out_df,
                                      ewm.rename(columns=new_names),
                                      how='outer',
                                      on=group_col)

                
                ## add rolling features
                for w in windows:
                    for method in agg_methods:
                        rolling_filled = calc_rolling(data_preag_filled,
                                                      group_col, date_col,
                                                      method, w)
                        

                        ## lf_df - DataFrame with following columns: filter_col, id_cols, date_col, shifted rolling stats
                        rolling = shift(rolling_filled, group_col, date_col,
                                        lag)
                        

                        method_name = method.__name__ if type(
                            method) != str else method

                        new_names = {x: "{0}_lag{1}d_w{2}_{3}".                                     format(x, lag, w, filter_col) for x in target_cols}
                        out_df = pd.merge(out_df,
                                          rolling.rename(columns=new_names),
                                          how='outer',
                                          on=group_col)
                        progress.value += 1

    return out_df



def preABT_modification(data : pd.DataFrame) -> (pd.DataFrame):

    target_cols = ['TGT_QTY']
    id_cols = ['PRODUCT_ID', 'LOCATION_ID']
    date_col = 'PERIOD_DT'
    built_in_funcs = [pd.Series.kurtosis, pd.Series.skew]

    # flts = {'Promo': {'oprm':'>0', 'npromo':'==0', 'aprm':'>-1'}, 'weekday' : {'md':'==0', 'tue':'==1', 'wd':'==2', 'th':'==3', 'fr':'==4', 'sa':'==5', 'su':'==6', 'anyday':'>-1'}}


    data['NoFilter'] = 1


    data_lagged_features = generate_lagged_features(data 
                        , target_cols = target_cols
                        , id_cols = id_cols
                        , date_col = date_col
                        , lags = [22, 28, 35]
                        , windows = ['14D', '21D', '28D', '56D']
                        , preagg_methods = ['sum'] # ['mean', 'count']
                        , agg_methods = ['mean'] #, percentile(10),  percentile(90)]
                        , dynamic_filters = ['PROMO_FLG', 'NoFilter']
                        , ewm_params={'NoFilter': [14, 28], 'PROMO_FLG': [14, 28]}
                        )
    
    return data_lagged_features




