import pandas as pd
import numpy as np
import datetime
from calendar import monthrange
import reconcilation
import warnings
warnings.filterwarnings('ignore')


def add_period_ends(VF_FORECAST : pd.DataFrame, ML_FORECAST : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Step 3.2
    Adding ends of forecasting periods
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column.
    ML_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column,
        FORECAST_VALUE column, columns with types of DEMAND and ASSORTMENT.
    config : dict
        Configuration parameters used within the step
        
    
    Returns
    -------
    pd.DataFrame
        VF_FORECAST with added periods ends
    pd.DataFrame
        MF_FORECAST with added periods ends
    
    """
    VF_FORECAST['PERIOD_END_DT'] =  VF_FORECAST['PERIOD_DT'].apply(
        lambda x : pd.date_range(x, periods=2, freq='W')[1]
    )
    VF_FORECAST['PERIOD_END_DT']-= pd.Timedelta('1D')

    ML_FORECAST['PERIOD_END_DT'] =  ML_FORECAST['PERIOD_DT'].apply(
        lambda x : pd.date_range(x, periods=2, freq='D')[1]
    )
    ML_FORECAST['PERIOD_END_DT']-= pd.Timedelta('1D')
    ML_FORECAST = ML_FORECAST.rename(columns={'FORECAST_VALUE' : 'ML_FORECAST_VALUE'})
    VF_FORECAST = VF_FORECAST.rename(columns={'FORECAST_VALUE' : 'VF_FORECAST_VALUE'})
    return VF_FORECAST, ML_FORECAST

def match_forecasts(VF_FORECAST : pd.DataFrame, ML_FORECAST : pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.5
    JOIN VF to ML. Transforming period columns.
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column
    
    Returns
    -------
    pd.DataFrame
        Table containing joint vf and ml forecasts
    
    """
    merge_keys = VF_FORECAST.columns[VF_FORECAST.columns.str.contains('ID')].tolist()
    df = pd.merge(ML_FORECAST, VF_FORECAST, on=merge_keys, how='left', suffixes=['_ML', '_VF'])
    df = df[(df['PERIOD_DT_ML'] <= df['PERIOD_END_DT_VF']) & (df['PERIOD_END_DT_ML'] >= df['PERIOD_DT_VF'])]
    df['PERIOD_DT'] = np.maximum(df['PERIOD_DT_ML'], df['PERIOD_DT_VF'])
    df['PERIOD_END_DT'] = np.maximum(df['PERIOD_END_DT_ML'], df['PERIOD_END_DT_VF'])
    df['VF_FORECAST_VALUE'] = df['VF_FORECAST_VALUE'].fillna(0)
    df = df.drop(['PERIOD_DT_ML', 'PERIOD_DT_VF', 'PERIOD_END_DT_ML', 'PERIOD_END_DT_VF'], axis=1)
    df = df.reset_index(drop=True)
    return df


def number_days(time_lvl : str, period_dt : datetime.datetime) -> int:
    """
    Function that calculate days number in a time_lvl interval that contains PERIOD_DT
    
    Parameters
    ----------
    time_lvl : {'DAY', 'WEEK', 'MONTH'}
        vf_time_lvl or ml_time_lvl from config
    period_dt : datetime.datetime
        The start of forecasting period of one object
        
    Returns
    -------
    int
       Days number in a time_lvl interval that contains PERIOD_DT 
    """
    if time_lvl == 'DAY':
        return 1
    if time_lvl == 'WEEK':
        return 7
    if time_lvl == 'MONTH':
        return monthrange(period_dt.year, period_dt.month)[1]


def interval_forecast_correction(df : pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.6
    Calculate forecast share and volume of VF_FORECAST_VALUE and ML_FORECAST_VALUE
    proportionaly to number of day in interval [PERIOD_DT, PERIOD_END_DT]
    
    Parameters
    ----------
    df : pd.DataFrame
        The forecasts table obtained in the previous steps of the algorithm
    config : dict
        Configuration parameters used within the step
    Returns
    -------
    pd.DataFrame
        The forecasts table with shared and volumed forecasts
    """
    number_days_vf = df['PERIOD_DT'].apply(lambda x : number_days('WEEK', x))
    number_days_ml = df['PERIOD_DT'].apply(lambda x : number_days('DAY', x))
    period_len = (df['PERIOD_END_DT'] - df['PERIOD_DT']).dt.days + 1
    df['ML_FORECAST_VALUE'] *= (period_len / number_days_ml)
    df['VF_FORECAST_VALUE'] *= (period_len / number_days_vf)
    return df


def reconcile(df : pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.2
    Reconcile VF_FORECAST_VALUE to ML_FORECAST_VALUE
    
    Parameters
    ----------
    df : pd.DataFrame
        The forecasts table obtained in the previous steps of the algorithm
    config : dict
        Configuration parameters used within the step
    
    Returns
    -------
    pd.DataFrame
        The forecasts table with reconciled forecasts
    """
    keys = df.columns[df.columns.str.contains('ID')].tolist()
    sums = df.groupby(keys)[['ML_FORECAST_VALUE', 'VF_FORECAST_VALUE']].sum()
    ratio = pd.DataFrame(sums['VF_FORECAST_VALUE'] / sums['ML_FORECAST_VALUE'], columns=['ratio']).reset_index()
    df = pd.merge(df, ratio, on=keys)
    df['VF_FORECAST_VALUE_REC'] = df['ML_FORECAST_VALUE'] * df['ratio']
    df = df.drop(['VF_FORECAST_VALUE', 'ratio'], axis=1)
    return df


def reconcilation_algorithm(VF_FORECAST : pd.DataFrame, ML_FORECAST : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Pipeline for reconcilation algorithm.
    
    Forecast Reconciliation step is aimed at bringing ML and Statistical
        forecasts to the same granularity level (e.g. at product/location/day level).
    The result of this step is used to generate hybrid forecast value. Usually,
        the forecast is split to more granular level at the Reconciliation step.
    This step may include some PL processing logic (e.g. phase-in and phase-out dates) as well as  
    This step is not needed if only one approach (ML or Stat) is used for forecasting.
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column.
    ML_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column,
        FORECAST_VALUE column, columns with types of DEMAND and ASSORTMENT.
    VF_TS_SEGMENTS : pd.DataFrame
        VF_TS_SEGMENTS containg information about segment names
    config : dict
        Configuration parameters used within the step

        
    Returns
    -------
    pd.DataFrame
        Reconciled forecast
    
    """
    VF_FORECAST, ML_FORECAST = add_period_ends(VF_FORECAST, ML_FORECAST)
    df = match_forecasts(VF_FORECAST, ML_FORECAST)
    df = interval_forecast_correction(df)
    df = reconcile(df)
    return df
    
