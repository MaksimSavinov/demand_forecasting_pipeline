import pandas as pd
import numpy as np
import datetime
from calendar import monthrange
import warnings
warnings.filterwarnings('ignore')
import sys  


def hybridization(df : pd.DataFrame) -> (pd.DataFrame):
    """
    Hybridization algorithm.
   
    Forecast Consolidation step is aimed to merge several forecast values for 
    a triple product/location/day and provide one (hybrid forecast) value for it.
    Input forecasts are to be in one granularity level due to Reconciliation step.
   
    Parameters
    ----------
    df : pd.DataFrame
        Input table containing the VF and ML forecasting and name of segments.
       
    Returns
    -------
    pd.DataFrame
        Data with new fields Hybrid forecast value
     """   

    df['IB_ZERO_DEMAND_THRESHOLD'] = 0.1
    df['VF_FORECAST_VALUE_F'] = df.VF_FORECAST_VALUE_REC.combine_first(df.ML_FORECAST_VALUE)
    df['ML_FORECAST_VALUE_F'] = df.ML_FORECAST_VALUE.combine_first(df.VF_FORECAST_VALUE_REC)
    df['HYBRID_FORECAST_VALUE'] = np.where(((df['DEMAND_TYPE']=='promo') & (df['SEGMENT_NAME']!='Retired'))|
                                         (df['SEGMENT_NAME']!='Short')|
                                         (df['ASSORTMENT_TYPE']=='new'), df['ML_FORECAST_VALUE_F'], 
                                         np.where((df['SEGMENT_NAME'] == 'Retired')|
                                                  (df['SEGMENT_NAME'] == 'Low Volume')|
                                                 (df['VF_FORECAST_VALUE_F']<=df['IB_ZERO_DEMAND_THRESHOLD']), df['VF_FORECAST_VALUE_F'],
                                                 (df['VF_FORECAST_VALUE_F'] + df['ML_FORECAST_VALUE_F'])/2))
    df['FORECAST_SOURCE'] = np.where(((df['DEMAND_TYPE']=='promo') & (df['SEGMENT_NAME']!='Retired'))|
                                         (df['SEGMENT_NAME']!='Short')|
                                         (df['ASSORTMENT_TYPE']=='new'), 'ml', 
                                         np.where((df['SEGMENT_NAME'] == 'Retired')|
                                                  (df['SEGMENT_NAME'] == 'Low Volume')|
                                                 (df['VF_FORECAST_VALUE_F']<=df['IB_ZERO_DEMAND_THRESHOLD']), 'vf',
                                                 'ensemble'))
    df.drop(columns = {'IB_ZERO_DEMAND_THRESHOLD', 'VF_FORECAST_VALUE_REC','ML_FORECAST_VALUE'}, inplace = True)
    df.rename(columns = {'VF_FORECAST_VALUE_F':'VF_FORECAST_VALUE','ML_FORECAST_VALUE_F':'ML_FORECAST_VALUE'}, inplace = True)
    
    return df





