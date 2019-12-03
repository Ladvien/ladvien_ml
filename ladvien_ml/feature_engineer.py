#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 07:12:51 2019

@author: ladvien
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# Statistics packages
from scipy.stats import spearmanr

class FeatureEngineer:
    
    def __init__(self):
        pass
    
    def fragment_date(self, df, feature_name, drop_parent = True):
        """        
        Converts a pandas.DataFrame datetime field into interval counter parts (
        e.g., 2018-10 becomes two fields, 2018 and 10).
        
        Parameters
        ----------
        df : DataFrame
            The dataframe containing the feature to fragment.
        feature_name : str
            The name of the feature column.
        drop_parent : boolean, defualt: True
            If `True` the date_feture will be dropped from the dataframe after 
            adding the fragmented columns.
        
        Procedure
        ---------
            1. Get average length of date as string
            2. Force feature to datime
            3. Add columns 'feature_year,' 'feature_month,' and 'feature_day.'
            4. If datetime, add columns 'feature_hour,' 'feature_minute,' and 'feature_day'
        """
        
        print(f'Fragmenting: {feature_name}')

        feature_name_len_min = df[feature_name].astype(str).str.len().min()
        
        try:
            df[feature_name] = pd.to_datetime(df[feature_name], format='%Y-%m-%d %H:%M:%S')
        except:
            print(f'Failed to cast feature to datetime. Feature is type: {df[feature_name].dtype}')
            return 
        
        df[feature_name + '_year'] = df[feature_name].dt.year
        df[feature_name + '_month'] = df[feature_name].dt.month
        df[feature_name + '_day'] = df[feature_name].dt.day
        
        if feature_name_len_min > 10: # Timestamp length.
            df[feature_name + '_hour'] = df[feature_name].dt.hour
            df[feature_name + '_minute'] = df[feature_name].dt.minute
            df[feature_name + '_second'] = df[feature_name].dt.second

        if drop_parent:
            df.drop(feature_name, axis = 1, inplace = True)

    
    def fragment_dates(self, df, feature_names, drop_parent = True):
        """        
        Converts pandas.DataFrame datetime fields into interval counter parts (
        e.g., 2018-10 becomes two fields, 2018 and 10).
        
        Parameters
        ----------
        df : DataFrame
            The dataframe containing the feature to fragment.
        feature_names : [str]
            A list containing the names of the feature columns.
        drop_parent : boolean, defualt: True
            If True the date_feture will be dropped from the dataframe after 
            adding the fragmented columns.
        
        Procedure
        ---------
        Loops through the feature_names array and does the following
            1. Get average length of date as string
            2. Force feature to datime
            3. Add columns 'feature_year,' 'feature_month,' and 'feature_day.'
            4. If datetime, add columns 'feature_hour,' 'feature_minute,' and 'feature_day'
        """
        
        for feature_name in feature_names:
            self.fragment_date(df, feature_name, drop_parent)

    def create_days_between_feature(self, df, feature_name_one, feature_name_two):
        """
        Takes two pandas.DataFrame datetime columns and calculates the absolute
        days between the dates.
        
        Parameters
        ----------
        df : DataFrame
            The dataframe containing the dates to calculate days between.
        feature_name_one : str
            The name of the primary feature column.
        feature_name_two : str
            The name of the secondary feature column.
        
        Procedure
        ---------
            1. Ensure both features are pd.datetime
            2. Take the absolute values from the difference of the secondary date from primary and cast to datetime.days
            3. Add this as a column to dateframe as, 'days_between_primary_and_secondary.'
        """
        if is_datetime(df[feature_name_one]) and is_datetime(df[feature_name_two]):
            df[f'days_between_{feature_name_one}_and_{feature_name_two}'] = \
                abs((df[feature_name_one] - df[feature_name_two]).dt.days)
        else:
            print(f'Excepted datetime features, received 1: {df[feature_name_two].dtype} 2:{df[feature_name_two].dtype}')
            return

            
    def get_target_correlations(self, df, target_name, nan_policy = 'omit'):
        """
        Calculates Spearman Coefficient across all fields and a target field.
        
        Parameters
        ----------
        df : DataFrame
            pandas dataframe containing independent and dependent variables.
        target_name : str
            The name of the column containing the dependent variable.
        nan_policy : str, default: 'omit'
            Defines how to handle when input contains nan. ‘propagate’ 
            returns nan, ‘raise’ throws an error, ‘omit’ performs the 
            calculations ignoring nan values. Default is ‘propagate’.
        
        Description
        -----------
        A Spearman correlation coefficient is calculated for all columns in
        the dataframe. A dataframe is returned containing:
            1. The coefficient (rho)
            2. P-value
            3. Determination of significance (rho > p)
        """
        
        correlations_df = pd.DataFrame()
        for feature in df.columns.tolist():
            if feature == target_name:
                continue
            
            corr_value, p = spearmanr(df[feature], df[target_name], nan_policy = 'omit')
            significance = 0
            if p < abs(corr_value):
                significance = 1
            if not np.isnan(corr_value):
                correlations_df = correlations_df.append({'feature_name': feature,
                                                          'relation_with_' + target_name: corr_value,
                                                          'p_value': p,
                                                          'significant': significance}, ignore_index = True, sort = False)
        return correlations_df