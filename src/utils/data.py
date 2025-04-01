import io_manager as iom
import variables as vrb
import pandas as pd
import numpy as np

from sklearn import preprocessing

TARGET = vrb.COLUMN_CLOSE

def get_train_valid_test_sets(dataframe, columns, seq_length=vrb.DEFAULT_SEQUENCE_LENGTH):
    """
    Returns train (60%), valid(20%) and test sets(20%) from the original dataframe

    Args:
    dataframe (dataframe): original dataframe
    columns (str array): array of columns to include in sets
    """
    dataframe_train = dataframe["2017-01":"2022-02"][columns].copy()
    dataframe_valid = dataframe["2022-03":"2023-08"][columns].copy()
    dataframe_test = pd.concat([dataframe_valid[-seq_length:], dataframe["2023-09":][columns]]) 

    return dataframe_train, dataframe_valid, dataframe_test

def get_colums_target_features_1():
    """
    Returns 1 column array (Close)

    """
    columns = [vrb.COLUMN_CLOSE]
    return columns

def get_colums_original_features_2():
    """
    Returns 2 columns array (Close, Volume)

    """
    columns = [vrb.COLUMN_CLOSE, vrb.COLUMN_VOLUME]
    return columns

def get_columns_sizes_features_5():
    """
    Returns 5 columns array (Close, Volume, Body, UpperShadow, LowerSadow)

    """
    columns_sizes = get_colums_original_features_2() + [vrb.COLUMN_BODY, vrb.COLUMN_UPPER_SHADOW, vrb.COLUMN_LOWER_SHADOW]
    return columns_sizes

def get_columns_all_features_11(dataframe):
    """
    Returns 11 columns array (Close, Volume, Body, UpperShadow, LowerSadow, DaysOfWeek_X), including get_dummies for column DayOfWeek

    Args:
    dataframe (dataframe): original dataframe
    """
    columns_all = get_columns_sizes_features_5()
    for col in dataframe.columns:
        if col.startswith(vrb.COLUMN_DAY_OF_WEEK):
            columns_all.append(col)
    return columns_all

def get_features_num(dataframe, remove_target=True):
    """
    Returns 11 columns array (Close, Volume, Body, UpperShadow, LowerSadow, DaysOfWeek_X), including get_dummies for column DayOfWeek

    Args:
    dataframe (dataframe): original dataframe
    """
    features_num = list(dataframe.columns)
    features_num.remove(vrb.COLUMN_DATE)
    features_num.remove(vrb.COLUMN_DAY_OF_WEEK)
    if remove_target:
        features_num.remove(vrb.COLUMN_CLOSE)
    return features_num

def prepare_dataframe(dataframe):
    """
    Create calculated columns based on original column's dataframe

    Args:
    dataframe (dataframe): original dataframe
    """
    dataframe[vrb.COLUMN_DATE] = dataframe.index
    dataframe.sort_index(inplace=True)

    # Set COLUM_DATE first
    order_columns = dataframe.columns.to_list()
    order_columns = order_columns[-1:] + order_columns[:-1]

    dataframe = pd.DataFrame(dataframe, columns=order_columns)

    #Add calculated columns
    dataframe[vrb.COLUMN_BODY] = dataframe[vrb.COLUMN_CLOSE] - dataframe[vrb.COLUMN_OPEN]
    dataframe[vrb.COLUMN_UPPER_SHADOW] = dataframe[vrb.COLUMN_HIGH] - dataframe[[vrb.COLUMN_OPEN, vrb.COLUMN_CLOSE]].max(axis=1)
    dataframe[vrb.COLUMN_LOWER_SHADOW] = dataframe[[vrb.COLUMN_OPEN, vrb.COLUMN_CLOSE]].min(axis=1) - dataframe[vrb.COLUMN_LOW]

    dataframe[vrb.COLUMN_DAY_OF_WEEK] = dataframe[vrb.COLUMN_DATE].dt.day_name()

    dataframe = dataframe.round(vrb.EURUSD_DECIMALS)
    return dataframe

def transform_data(dataframe, set_dummies=True):
    """
    Apply log1p to numerics columns, except target, scale volume column and convert DayOfWeek to dummies 

    Args:
    dataframe (dataframe): original dataframe
    set_dummmies (bool):  Default True. Generate DayOfWeek dummies columns.
    """
    
    for col in get_features_num(dataframe, remove_target=True):
        dataframe[col] = dataframe[col].apply(np.log1p)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    dataframe[vrb.COLUMN_VOLUME] = min_max_scaler.fit_transform(dataframe[[vrb.COLUMN_VOLUME]])

    if set_dummies:
        dataframe = pd.get_dummies(dataframe, dtype= float)

    return dataframe

def get_data_model(path=vrb.DATASOURCE_PATH):
    """
    Returns dataframe from csv, applying transformations ready to use with models

    Args:
    path (string): Data source filename.
    """
    dataframe = iom.read_data(path)
    dataframe = prepare_dataframe(dataframe)
    dataframe = transform_data(dataframe)

    return dataframe

