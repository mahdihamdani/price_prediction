#!/usr/bin/env python
# # -*- coding: utf-8 -*-

'''
train_price.py: Training script for price prediction using text/meta information

The app predicts the product price given a description and meta information about the product

__author__      = "Mahdi Hamdani"
__email__ = "mahdi.hamdani@gmail.com"
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from scipy.sparse import hstack
from category_encoders import *

from feature_transformation import *
from pickle_io import *
import argparse

parser = argparse.ArgumentParser("Training script for price prediction")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--models_dir', type=str, default='models')
parser.add_argument('--config_pkl', type=str, default='best_config')
parser.add_argument('--models_pkl', type=str, default='best_models')
args = parser.parse_args()

def conv_list(x):
    '''
        Helper to convert string list fields to actual lists
        Args:
            x: A string representing the list
        Returns: list(str): A sorted list of strings converted to lower case
    '''
    try:
        #transforming string to string list 
        return eval(x)
    except SyntaxError as e:
        return np.nan

str_fields = ['product_name', 'menu_category', 'product_description']
list_fields = ['cuisine_characteristics', 'taste_characteristics', \
               'preparation_style_characteristics', 'dish_type_characteristics']
cat_fields = ['postcode', 'city_id']
num_fields = ['latitude', 'longitude']
pred_fields = ['price']

feature_fields = str_fields+list_fields+cat_fields+num_fields

converters = dict(zip(list_fields, [conv_list] * len(list_fields)))

data = pd.read_csv(args.dataset, sep=';', usecols=feature_fields + pred_fields, \
                  converters=converters)

#removing all rows with nan in log or lattitude
data.dropna(subset=num_fields, inplace=True)

'''
Splitting the dataset to train, dev and test sets
We will try many data transformations --> better to keep test data for final tets to avoid "overfitting" on dev data
'''

x_train, x_dev, y_train, y_dev = train_test_split(data[feature_fields], data['price'], \
                                                                    test_size=0.2, random_state=1234)

print 'Summary of used data:'
print 'Train data size', x_train.shape[0], 'rows'
print 'Dev data size', x_dev.shape[0], 'rows'
print

print 'Loading config'
try:
    best_solutions = load_obj(args.models_dir + '/' + args.config_pkl)
except:
    best_solutions = {'lat_long': HashingEncoder(), 'city_id': BinaryEncoder(), 'taste_characteristics': BackwardDifferenceEncoder(), \
                      'regressor': RandomForestRegressor(n_estimators=10, n_jobs=-1), 'product_description': 2, \
                      'cuisine_characteristics': OrdinalEncoder(), 'postcode': BinaryEncoder(), 'dish_type_characteristics': PolynomialEncoder(), \
                      'preparation_style_characteristics': LeaveOneOutEncoder(), 'product_name': 1, 'menu_category': 1}

print 'Processing features...'

x_train_feats = None
x_dev_feats = None

'''
transforming the string fields, these are free text fields so we will use a text analyzer
Bag of n-grams can be used in this case: 
The context is important in this case, e.g. Pizza Salami vs Pizza al Tonno
'''

x_train_feats, vectorizers = transform_strings(x_train, str_fields, best_solutions, x_train_feats)
x_dev_feats, _ = transform_strings(x_dev, str_fields, best_solutions, x_dev_feats, vectorizers)

'''
1) Transforming lists to indicator features
2) Transforming indicator features to more sophisticated categorical features
'''

x_train_feats, transformers_lists, unique_cols = transform_lists(x_train, list_fields, best_solutions, x_train_feats, labels=y_train)
x_dev_feats, _, _ = transform_lists(x_dev, list_fields, best_solutions, x_dev_feats, transformers=transformers_lists, unique_cols=unique_cols)

'''
Using sophisticated categorical features
'''

x_train_feats, transformers_cat = transform_categorical(x_train, cat_fields, best_solutions, x_train_feats, labels=y_train)
x_dev_feats, _ = transform_categorical(x_dev, cat_fields, best_solutions, x_dev_feats, transformers=transformers_cat)

'''
1) clustering the lattitude/longitude
2) constructing a feature with the cluster index
3) transforming to more sophisticated categorical features
'''

x_train_feats, transformer_geo, clusterer = transform_geographic(x_train, best_solutions, x_train_feats, labels=y_train)
x_dev_feats,_,_ = transform_geographic(x_dev, best_solutions, x_dev_feats, transformer=transformer_geo, clusterer=clusterer)

print 'final number of featues', x_train_feats.shape[1]
assert x_train_feats.shape[1] == x_dev_feats.shape[1]

print 'Starting training...'

regressor = best_solutions['regressor']
regressor.fit(x_train_feats , y_train)
train_hyp = regressor.predict(x_train_feats)
dev_hyp = regressor.predict(x_dev_feats)

print 'Results using regressor', type(regressor).__name__
print 'Train MSE=', mean_squared_error(y_train, train_hyp)
print 'Dev MSE=', mean_squared_error(y_dev, dev_hyp)
print 'Dev R2 score=',r2_score(y_dev, dev_hyp)
print

print ('Saving models to file ' + args.models_dir + '/best_models')
models = {'vectorizers': vectorizers, 'transformers_lists': transformers_lists, 'unique_cols': unique_cols, \
        'transformers_cat': transformers_cat, 'transformer_geo': transformer_geo, 'clusterer': clusterer, \
        'regressor': regressor}

save_obj(models, args.models_dir + '/' + args.models_pkl)

