#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 06:00:17 2019

@author: ladvien
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Flatten, \
                                    BatchNormalization, GaussianDropout, \
                                    AveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta, \
                                        Nadam, SGD
import tensorflow.keras.regularizers
from sklearn.metrics import confusion_matrix as cm, roc_auc_score
from tensorflow.keras import backend as K

class FeatureModel:
    
    def confusion_matrix_printed(self, actual_y, y_hat):
        tn, fn, tp, fp = cm(actual_y, y_hat).ravel()
        error = fn + fp
        correct = tn + tp
        total = error + correct
        print(f'Error: {round(error / total, 4) * 100}%')
        print(f'Accuracy: {round(correct / total, 4) * 100}%')
    
    def test_regression_model(self, model, X_test, y_test, output_path = '', id = ''):
        # ------------------------------------------------------
        # Create a dataframe from prediction and test
        # ------------------------------------------------------
        y_pred = model.predict(X_test)
        y_pred = y_pred[:,0:]
        compare = pd.DataFrame(data = {'y_test': y_test, 'y_pred': y_pred[:,0]})
        compare['abs_dff'] = abs(compare['y_test'] - compare['y_pred'])
    
        
        if output_path != '':
            file_path = output_path + '/compare_' + id + '.csv'
            compare.to_csv(file_path)
            print(f'Saved to {file_path}')
        return compare
    
    
    def test_classification_model(self, model, X_test, y_test, output_path = '', id = ''):
        y_hat = y_hat = model.predict_classes(X_test).ravel()
        y_hat = pd.Series(y_hat)
        
        y_test.reset_index(drop = True, inplace = True)
        
        # Check predictions.
        predictions = pd.concat([y_test, y_hat], axis = 1)
        predictions.columns = ['actual_y', 'y_hat']
        predictions['diff'] = abs(predictions['actual_y'] - predictions['y_hat'])
        predictions['correct'] = (predictions['diff'] < 1)
        self.confusion_matrix_printed(predictions['actual_y'], predictions['y_hat'])
    
    # ------------------------------------------------------
    # Network layout
    # ------------------------------------------------------
    from keras import regularizers
    def pile_layers(self, shape_size, optimizer, loss, layers, last_layer_activator, model_path = '', last_layer_output = 1, embed_model_path = ''):
        
        if model_path != '':
            print(f'Loading model from: {model_path}')
            return load_model(model_path)
        
        model = Sequential()
        
        # First Layer
        first_layer = layers[0]
        
        # First Layer is Dense.
        if first_layer['type'] == 'dense':
            if layers[0]['activation'] == 'lrelu':
                model.add(Dense(int(shape_size*layers[0]['widthModifier']), input_dim=shape_size))
                model.add(LeakyReLU(alpha=0.1))
            else:
                model.add(Dense(int(shape_size*layers[0]['widthModifier']), 
                           input_dim=shape_size, 
                           activation=layers[0]['activation']))
       
        elif first_layer['type'] == 'noise':
            GaussianDropout(layers[0]['widthModifier'])
            
        # First Layers is Embeddings.
        elif first_layer['type'] == 'embedding':
            try:
                embed_model = load_model(embed_model_path)
            except:
                print(f'No model found at {embed_model_path} path.')
            model.add(embed_model.layers[0])
            
            # Make sure the embeddings aren't updated.
            if not first_layer['update']:
                model.layers[0].trainable = False
            
            model.add(Flatten())   
            if layers[0]['dropout'] > 0:
                model.add(Dropout(rate = layers[0]['dropout']))
    
        # Other Layers
        for layer in layers[1:]:
            
            model.add(BatchNormalization())
            
            if layer['type'] == 'lnorm':
                l1 = layer['l1']
                l2 = layer['l2']
                model.add(Dense(int(shape_size*layer['widthModifier']), 
                                input_dim=shape_size, init='normal', 
                                kernel_regularize = regularizers.l1_l2(l1=l1, l2=l2), 
                                activation=layer['activation']))
                
            elif layer['type'] == 'pooling':
                if layer['pool_type'] == 'avg':
                    AveragePooling1D(pool_size=layer['pool_size'], strides=None, padding='valid', data_format='channels_last')
                elif layer['pool_type'] == 'max':
                    MaxPooling1D(pool_size=layer['pool_size'], strides=None, padding='valid', data_format='channels_last')
    
            elif layer['type'] == 'dense' and layer['activation'] == 'lrelu':
                model.add(Dense(int(shape_size*layer['widthModifier']), 
                input_dim=shape_size, init='normal'))  
                model.add(LeakyReLU(alpha=0.1))
                                
            elif layer['type'] == 'dense':
                model.add(Dense(int(shape_size*layer['widthModifier']), 
                input_dim=shape_size, init='normal', 
                activation=layer['activation']))     
    
            
            # Should we add some Dropout normilization?
            if layer['dropout'] > 0:
                model.add(Dropout(rate = layer['dropout']))
        
        
        if last_layer_activator:
            model.add(Dense(last_layer_output, activation=last_layer_activator))
        else:
            model.add(Dense(last_layer_output))
        
    
        # Compile the layer
        model.compile(loss=loss, optimizer = optimizer, metrics=[loss, 'accuracy'])
        
        return model
    
        
    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc
        
    def auroc(y_true, y_pred):
        try:
            return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
        except:
            auc = tf.metrics.auc(y_true, y_pred)[1]
            K.get_session().run(tf.local_variables_initializer())
            return auc
        
    # ------------------------------------------------------
    # Lazy load optimizer
    # ------------------------------------------------------
    def select_optimizer(self, opt_type, learning_rate, clipnorm = 0.5):
        if opt_type == 'adam':
            return Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif opt_type == 'rmsprop':
            return RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
        elif opt_type == 'adagrad':
            return Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
        elif opt_type == 'adadelta':
            return Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)    
        elif opt_type == 'nadam':
            return Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        elif opt_type == 'sgd':
            return SGD(lr = learning_rate, momentum = 0.0, decay = 0.0, nesterov = False, clipnorm = clipnorm)
        else:
            print('No optimizer')
            quit()
    
    # ------------------------------------------------------
    # Setup Data Loading
    # ------------------------------------------------------
    def move_dependent_var_to_end(self, df, dv_col_name):
        cols = list(df.columns.values) #Make a list of all of the columns in the df
        cols.pop(cols.index(dv_col_name)) #Remove b from list
        df = df[cols+[dv_col_name]] #Create new dataframe with columns in the order you want
        return df
    
    def load_train_data(self, path, dep_var, cols_to_drop = [], cols_to_keep = [], 
                        del_columns_containing = [], preserve_columns = [], 
                        samples = -1, split_rate = 0.2, encoding = 'ASCII'):
        
        print(path)
        
        import numpy as np
        from sklearn.model_selection import train_test_split
    
        # Load data using numpy to keep memory low.
        df = np.load(path, allow_pickle = True, encoding = encoding)
        
        # Do we need to sample the data?
        if samples > 0:
            df = df.sample(samples)
            df.reset_index(inplace = True, drop = True)
        
        # Let's move the dep_var to the end to split off.
        df = self.move_dependent_var_to_end(df, dep_var)
        y = df.iloc[:,-1].values
        df.drop(dep_var, axis = 1, inplace = True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = split_rate, random_state = 123) 
        # Clean
        df = None
        
        # Package X_train
        X_train, X_train_preserved_cols = self.preserve_delete_cols(X_train, preserve_columns, cols_to_drop, del_columns_containing, cols_to_keep)
    
        # Package X_test
        X_test, X_test_preserved_cols = self.preserve_delete_cols(X_test, preserve_columns, cols_to_drop, del_columns_containing, cols_to_keep)
        
        # Reset indexes.
        X_train.reset_index(inplace = True, drop = True) 
        X_train_preserved_cols.reset_index(inplace = True, drop = True)
        X_test.reset_index(inplace = True, drop = True)
        X_test_preserved_cols.reset_index(inplace = True, drop = True)
    
        return X_train, X_train_preserved_cols, X_test, X_test_preserved_cols, y_train, y_test
    
    def preserve_delete_cols(self, df, preserve_columns, cols_to_drop, del_columns_containing, cols_to_keep):
        preserved_cols = pd.DataFrame()
        # If there are ids we want for post-portem, preserve them
        if len(preserve_columns) > 0:
            # If a column doesn't exist in the dataframe, don't try to preserve.
            for col in preserve_columns:
                if not col in df:
                    preserve_columns.remove(col)
            preserved_cols = df[preserve_columns]
            
        # Drop unwanted columns.
        if len(cols_to_drop) > 0:
            for col in cols_to_drop:
                if col in list(df.columns.values):
                    df = df[df.columns.drop(col)]
                else:
                    print(f'Could not find {col} to drop.')
            
        if len(del_columns_containing) > 0:
            for col in del_columns_containing:
                df = df[df.columns.drop(list(df.filter(regex=col)))]
                
        # Keep only wanted columns, if defined.
        if len(cols_to_keep) > 0:
            df = df[cols_to_keep]
        
        return df, preserved_cols
        
    
    def feature_selection(self, data_path, cols_to_drop, preserve_cols, dep_var):
        # ------------------------------------------------------
        # Load Data
        # ------------------------------------------------------
        X, y, preserved_cols = self.load_train_data(data_path, dep_var, cols_to_drop, preserve_columns = preserve_cols, samples = 50000 )
        
        # ------------------------------------------------------
        # Feature Selection Method #1 -- Backward Elimmination
        # ------------------------------------------------------
        import statsmodels.api as sm
        # Backward Elimination
        cols = list(X.columns)
        pmax = 1
        while (len(cols) > 0):
        
            p = []
            X_1 = X[cols]
            X_1 = sm.add_constant(X_1)
            model = sm.OLS(y,X_1).fit()
            p = pd.Series(model.pvalues.values[1:], index = cols)      
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if(pmax > 0.05):
                print(f'Removing feature: {feature_with_p_max}')
                cols.remove(feature_with_p_max)
            else:
                break
        
        # Clear out the old data.
        print('Loading only selected features')
        X, y, preserved_cols = (None, None, None)
        return self.load_train_data(data_path, dep_var, cols_to_drop, cols_to_keep = cols, preserve_columns = preserve_cols )
    
    
    # Freedman Diaconis Estimator
    # Bins continuous variables well
    # TODO: Untested.
    # EXAMPLE:
    # for feat_name in cat_vars:
    #     train_df[feat_name] = smart_bin_feature(train_df[feat_name])
    def smart_bin_feature(feature):
        _, bin_edges = np.histogram(feature.fillna(0), bins = 'fd')
        return np.digitize(feature, bin_edges.flatten(), right = True)
    
    
    
    # This code was borrowed from ArjanGroen
    # https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    
    def reduce_mem_usage(self, props):
        start_mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
        NAlist = [] # Keeps track of columns that have missing values filled in. 
        for col in props.columns:
            if props[col].dtype != object:  # Exclude strings
                
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",props[col].dtype)
                
                # make variables for Int, max and min
                IsInt = False
                mx = props[col].max()
                mn = props[col].min()
                
                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(props[col]).all(): 
                    NAlist.append(col)
                    props[col].fillna(mn-1,inplace=True)  
                       
                # test if column can be converted to an integer
                asint = props[col].fillna(0).astype(np.int64)
                result = (props[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True
    
                
                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            props[col] = props[col].astype(np.uint8)
                        elif mx < 65535:
                            props[col] = props[col].astype(np.uint16)
                        elif mx < 4294967295:
                            props[col] = props[col].astype(np.uint32)
                        else:
                            props[col] = props[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props[col] = props[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props[col] = props[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props[col] = props[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props[col] = props[col].astype(np.int64)    
                
                # Make float datatypes 32 bit
                else:
                    props[col] = props[col].astype(np.float32)
                
                # Print new column type
                print("dtype after: ",props[col].dtype)
                print("******************************")
        
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
        return props, NAlist
