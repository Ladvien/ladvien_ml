#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:05:13 2019

@author: ladvien
"""

import pandas as pd
from datetime import datetime
import numpy as np


class FeaturePrep:
    
    DATE_FORMAT = '%Y-%m-%d'
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    def __init__(self):
        pass
        
    def remove_unwanted_columns(self, df, cols_to_remove=['Unnamed: 0', 'Unnamed: 0.1', 'looker_error']):
        """
            Drops the following columns: 'Unnamed: 0', 'Unnamed: 0.1', 'looker_error'
        """
        cols_to_remove = [col for col in cols_to_remove if col in df.columns.tolist()]
        df = df.drop(cols_to_remove, axis = 1)
        return df
    
    def clean_date_feature(self, df, feature_name, fill_date = '1800-01-01', look_back = '1800-01-01', mark_weird = False):
        """
            df: DataFrame containing feature to clean.
            feature_name: str, name of the date features to clean.
            mark_weird: boolean, if True a column will be created and each row
            will = 1 where date coercion failed.
            
            Cleans date columns.
                
                1. Detects date_type {'date', 'datetime'} default: 'date'
                2. Removes zero nanoseconds. I.e, '00:00:00.0' becomes  '00:00:00'
                2. Fills NA with 'fill_date'
                3. Creates a 'date_error' column marking dates with issues.
                4. Sets future dates to fill_date.
                5. Sets dates further back than look_back to fill_date.
        """
        # Problem dates to address:
        # "0000-00-00" -- invalid dates
        # "5600-02-03" -- bizare future dates
        # "1000-02-14" -- bizare past dates
        # "NaT"        -- Null, None, NaT
        
        date_length = int(df[feature_name].astype(str).str.len().mean())
        
        df[feature_name] = df[feature_name].astype(str)
        
        # Remove zero nanoseconds.
        df.loc[df[feature_name].str[-2:] == '.0', feature_name] = df[feature_name].str[:-2]
        
        if date_length > 10:
            date_type = 'datetime'
            to_date_format = self.DATETIME_FORMAT
            fill_date += ' 00:00:00'
            look_back += ' 00:00:00'
        else:
            date_type = 'date'
            to_date_format = self.DATE_FORMAT
        
        if mark_weird:
            df = self.mark_weird_dates(df, feature_name, date_type)
        
        # Standardize to YYYY-MM-DD HH-MM-SS
        df[feature_name].fillna(datetime.strptime(f'{fill_date}', to_date_format), inplace = True)
        try:
            # Malformed dates coerced to NaT
            df[feature_name] = pd.to_datetime(df[feature_name], format = to_date_format, errors = 'coerce')
            # NaT replaced with fill-in date.
            df[feature_name].fillna(datetime.strptime(f'{fill_date}', to_date_format), inplace = True)
            # Future dates are set as today.
            df.loc[df[feature_name] > datetime.now(), feature_name] = datetime.today()
            # Weird past dates are set to filla_date.
            df.loc[df[feature_name] <= datetime.strptime(f'{look_back}', to_date_format), feature_name] = datetime.strptime(f'{fill_date}', to_date_format)
        except:
            print(f'Failed to parse {feature_name} as a datetime series.')
    
    
    def clean_dates(self, df, feature_names, fill_date = '1800-01-01', look_back = '1800-01-01', mark_weird = False):
        """
            df: DataFrame containing feature to clean.
            feature_names: array[str] names of the date features to clean.
            mark_weird: boolean, if True a column will be created and each row
            will = 1 where date coercion failed.
            
            Cleans date columns.
                1. Detects date_type {'date', 'datetime'} default: 'date'
                2. Fills NA with 'fill_date'
                3. Creates a 'date_error' column marking dates with issues.
                4. Sets future dates to fill_date.
                5. Sets dates further than look_back to fill_date.
        """
        for feature_name in feature_names:
            print(f'Cleaning: {feature_name}')
            self.clean_date_feature(df, feature_name, fill_date, look_back, mark_weird)

        
    def mark_weird_dates(self, df, feature_name, date_type = 'date', look_back = '1800-01-01'):
        """
        
            df: DataFrame, dataset containing the date feature.
            feature_name: str, name of the date feature.
            date_type, str, {'date', 'datetime'}, default: date 
            look_back: str, any dates before this date will be replaced with the lookback date. default: '1800-01-01'
            
            # 1. Create a new 'feature_name_error' column.
            # 2. Coerce to datetime.
            # 3. Remove '00:00:00' if it is a date.
            # 4. Coerce original and new series to string, marking wherever they 
            #    are not equal, as these items have failed conversion.
            # 5. Mark dates going back too far.
            # 6. Mark future dates.
            
        """
        
        df[feature_name + '_error'] = 0
        
        original_date_series = df[feature_name]
        coerced_data_series = df[feature_name]
        
        # Coercion causes bad dates to become NaT, which will not match the original.
        coerced_data_series = pd.to_datetime(original_date_series, format = self.DATE_FORMAT, errors = 'coerce')
        coerced_data_series[coerced_data_series <= datetime.strptime(f'{look_back}', self.DATE_FORMAT)] = 1
        
        # Need to trim the time compare against date original.
        coerced_data_series = coerced_data_series.astype(str)
        
        if date_type == 'date':
            coerced_data_series = coerced_data_series.str.replace(' 00:00:00.0', '')

        # Create the error column.  Anywhere not matching should be marked
        # as an error.
        df.loc[coerced_data_series.astype(str) != original_date_series.astype(str), feature_name + '_error'] = 1

        return df

    def make_ohe_feature_readable(self, feature_name, seperator = '.', print_output = False, separate_value = False):
        """

                feature_name: str, name of the feature.
                seperator: str, the character or string representing label from value.
                print_output: Boolean, print the processed feature name.
                separate_value: Boolean, if true, returns a tuple containing the feature name and the value.

                When one-hot encoding (OHE) values the column names often get garbled.  This method will take a
                OHE string and return a human readable name.  The string should be of the type

                name_of_feature_value, e.g., 'weight_range.150'

                >>> print(make_ohe_feature_readable(weight_range.150))
                "Weight Range = 150"
        """

        # When avoiding the the dummy variable trap two category variables
        # will not have a seperator or value in the feature name.
        if seperator not in feature_name:
                feature_name += f'{seperator}1'

        # If underscores are used instead of spaces, replace them.
        feature_name = feature_name.replace('_', ' ')

        split_feature_name = feature_name.split(seperator)
        value = split_feature_name[1]

        # Use SQL like nulls.
        if value == 'nan':
                value = 'NULL'
        
        # If there's a third split, we don't want it. E.g., 'neat_feature.0.0'.
        split_feature_name = split_feature_name[0:1]  
        feature_name = ' '.join(split_feature_name).title()
        full_output = f'{feature_name} = {value}'

        if print_output:
            print(full_output)

        if separate_value:
            return (feature_name, value)
        else:
            return full_output

   # This code was borrowed from ArjanGroen
    # https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    
    def reduce_mem_usage(self, df):
        start_mem_usg = df.memory_usage().sum() / 1024**2 
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
        NAlist = [] # Keeps track of columns that have missing values filled in. 

        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        cols = [col for col in df.columns.tolist() if col not in date_cols]

        for col in cols:
            if df[col].dtype != object:  # Exclude strings
                
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",df[col].dtype)
                
                # make variables for Int, max and min
                IsInt = False
                mx = df[col].max()
                mn = df[col].min()
                
                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(df[col]).all(): 
                    NAlist.append(col)
                    df[col].fillna(mn-1,inplace=True)  
                       
                # test if column can be converted to an integer
                asint = df[col].fillna(0).astype(np.int64)
                result = (df[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True
    
                
                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif mx < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif mx < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)    
                
                # Make float datatypes 32 bit
                else:
                    df[col] = df[col].astype(np.float32)
                
                # Print new column type
                print("dtype after: ",df[col].dtype)
                print("******************************")
        
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
        return df, NAlist

