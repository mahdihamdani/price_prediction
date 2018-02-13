#!/usr/bin/env python
# # -*- coding: utf-8 -*-

'''
predict_price.py: Test script for price prediction using text/meta information

The app predicts the product price given a description and meta information about the product

__author__      = "Mahdi Hamdani"
__email__ = "mahdi.hamdani@gmail.com"
'''

import argparse
from pickle_io import load_obj
import pandas as pd

from feature_transformation import *

str_fields = ['product_name', 'menu_category', 'product_description']
list_fields = ['cuisine_characteristics', 'taste_characteristics', \
               'preparation_style_characteristics', 'dish_type_characteristics']
cat_fields = ['postcode', 'city_id']
num_fields = ['latitude', 'longitude']

feature_fields = str_fields+list_fields+cat_fields+num_fields

parser = argparse.ArgumentParser("Test script for price prediction")

parser.add_argument('--product_name', required=True, type=str, help='e.g. Pizza Margherita, Penne Arrabiata')
parser.add_argument('--product_description', default='', type=str, help='e.g. Tomatensauce und fettarmer Käse, Spaghetti Aglio')
parser.add_argument('--menu_category', default='', type=str, help='e.g. Fit Pizza, Dolce')
parser.add_argument('--cuisine_characteristics', default='', type=str, help='coma separeted: e.g. "Italienisch", "Amerikanisch, Gesundes Essen"')
parser.add_argument('--taste_characteristics',default='', type=str, help='coma separeted: e.g. "Natürlich", "Leicht scharf, Mild"')
parser.add_argument('--preparation_style_characteristics', default='', type=str, help='coma separeted: e.g. "Gekocht", "frittiert, Gebacken"')
parser.add_argument('--dish_type_characteristics', default='', type=str, help='coma separeted: e.g. "Pommes", "Fleisch, Gemüse"')
parser.add_argument('--postcode', type=str, default='', help='postal code, e.g 35043"')
parser.add_argument('--city_id', type=int, default=0, help='city, e.g. 9')
parser.add_argument('--latitude', default=0.0, type=float, help='latitude, e.g. 50.11352921')
parser.add_argument('--longitude', default=0.0, type=float, help='longitude, e.g. 8.68463135')
parser.add_argument('--config_pkl', type=str, default='best_config')
parser.add_argument('--models_pkl', type=str, default='best_models')
parser.add_argument('--models_dir', type=str, default='models')

args = parser.parse_args()


print 'Loading config and models'
best_solutions= load_obj(args.models_dir + '/' + args.config_pkl)
models = load_obj(args.models_dir + '/' + args.models_pkl)

#Preparing list columns
cuisine_characteristics = args.cuisine_characteristics.split(',')
taste_characteristics = args.taste_characteristics.split(',')
preparation_style_characteristics = args.preparation_style_characteristics.split(',')
dish_type_characteristics = args.dish_type_characteristics.split(',')

print 'Preparing data and tranforming features'
row = [args.product_name, args.menu_category, args.product_description, cuisine_characteristics, taste_characteristics, \
       preparation_style_characteristics, dish_type_characteristics, args.postcode, args.city_id, args.latitude, args.longitude]
x_test = pd.DataFrame([row], columns=feature_fields)


x_test_feats = None
'''
transforming the string fields, these are free text fields so we will use a text analyzer
Bag of n-grams can be used in this case: 
The context is important in this case, e.g. Pizza Salami vs Pizza al Tonno
'''

x_test_feats, _ = transform_strings(x_test, str_fields, best_solutions, x_test_feats, models['vectorizers'])

'''
1) Transforming lists to indicator features
2) Transforming indicator features to more sophisticated categorical features
'''

x_test_feats, _, _ = transform_lists(x_test, list_fields, best_solutions, x_test_feats, transformers=models['transformers_lists'], unique_cols=models['unique_cols'])

'''
Using sophisticated categorical features
'''

x_test_feats, _ = transform_categorical(x_test, cat_fields, best_solutions, x_test_feats, transformers=models['transformers_cat'])

'''
1) clustering the lattitude/longitude
2) constructing a feature with the cluster index
3) transforming to more sophisticated categorical features
'''

x_test_feats,_,_ = transform_geographic(x_test, best_solutions, x_test_feats, transformer=models['transformer_geo'], clusterer=models['clusterer'])

regressor = models['regressor']

predicted_price = regressor.predict(x_test_feats)

print 'predicted price', predicted_price
