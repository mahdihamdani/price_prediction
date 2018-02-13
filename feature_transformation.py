from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import hstack
from bisect import bisect_left
import numpy as np
from haversine import haversine
from sklearn.cluster import DBSCAN

def transform_strings(in_feats, fields, best_solutions, out_feats, vectorizers={}):
    '''
        Transforms the string columns to Bag of n-grams
        More details are here: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
        Args:
            in_feats (DataFrame): input dataframe
            fields (list): list of columns
            best_solutions (dict): dict with options
            out_feats: output array
        Results:
            out_feats: concatenate existing featues with compted
            vectorizer: model for fields vectorization
    '''
    for col in fields:
        n_gram_level = best_solutions[col]
        if col not in vectorizers:
            vectorizer = CountVectorizer(ngram_range=(1, n_gram_level), min_df=1, strip_accents='unicode')
            feats_col = vectorizer.fit_transform(in_feats[col].replace(np.nan, '', regex=True))
            vectorizers[col] = vectorizer
        else:
            feats_col = vectorizers[col].transform(in_feats[col].replace(np.nan, '', regex=True))
        if out_feats is None:
            out_feats = feats_col
        else:
            out_feats = hstack([out_feats, feats_col])
    return out_feats, vectorizers

def contains(a, v):
    '''
        Helper to search value x in array a using binary search
        Args:
            a: sorted array
            v: value to search
        Returns: Boolean: True if v exists in a
    '''
    i = bisect_left(a, v)
    if i != len(a) and a[i] == v:
        return i
    return -1

def transform_lists(in_feats, fields, best_solutions, out_feats, labels= None, transformers={}, unique_cols=None):
    '''
        Transforms the list columns to categorical features
        1) creates indicator features
        2) use a transformer to have final categorical features
        Args:
            in_feats (DataFrame): input dataframe
            fields (list): list of columns
            best_solutions (dict): dict with options
            out_feats: output array
            labels: training labels
            transformer: object for feature transformation
            unique_cols: created columns while creating indicator features for training
        Results:
            out_feats: concatenate existing featues with compted
            transformer: model forfeature transformation
            unique_cols: created columns while creating indicator features for training
    '''
    ########################################################
    def indicatorFeature(in_list):
        '''
            Helper for one hot encoding for the lists
            Args:
                x: input list of strings
            Returns: list(string): one hot encoding of list
        '''
        result = [0] * len(unique_cols)
        if not isinstance(in_list, list): return result
        for value in in_list:
            i = contains(unique_cols, value)
            if i >= 0:
                result[i] = 1
        return result
    ########################################################

    for col in fields:
        if not unique_cols:
            #transform all possible values to a unique list
            values = []
            for v in in_feats[col].values.tolist():
                if v is np.nan: continue
                values += [s for s in v]
            unique_cols = sorted(set(values))

        #add indicator features: 1: exists, 0: no
        feats_col = in_feats[col].apply(indicatorFeature).apply(pd.Series)

        #apply categorical feature transformation
        if col not in transformers:
            transformer = best_solutions[col]
            transformer.fit(feats_col, labels)
            transformers[col] = transformer

        out_feats_transform = transformers[col].transform(feats_col)
        
        if out_feats is None:
            out_feats = out_feats_transform
        else:
            out_feats = hstack([out_feats,out_feats_transform])
    return out_feats, transformers, unique_cols


def transform_categorical(in_feats, fields, best_solutions, out_feats, labels= None, transformers={}):
    '''
        Transforms categorical columns to features
        Args:
            in_feats (DataFrame): input dataframe
            fields (list): list of columns
            best_solutions (dict): dict with options
            out_feats: output array
            labels: training labels
            transformer: object for feature transformation
        Results:
            out_feats: concatenate existing featues with compted
            transformer: model forfeature transformation
    '''
    for col in fields:
        if col not in transformers:
            transformer = best_solutions[col]
            transformer.fit(in_feats[col].to_frame(), labels)
            transformers[col] = transformer
        
        feats_col = transformers[col].transform(in_feats[col].to_frame())
        feats_col.fillna(0, inplace=True)

        if out_feats is None:
            out_feats = feats_col
        else:
            out_feats = hstack([out_feats, feats_col])

    return out_feats, transformers

def dbscan_predict(dbscan_model, points, metric=haversine):
    '''
        Helper to find closest cluster
        Args:
            dbscan_model: trained sklearn DBSCAN model
            X_new np.array: lat/long in radians
            metric: used metric for comparing points
        Returns: np.array: cluster values
    '''
    # Result is noise by default
    cluster_pred = np.ones(shape=len(points), dtype=int)*-1

    # Iterate all input samples for a label
    for j, point in enumerate(points):
    # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(point, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                cluster_pred[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return cluster_pred

def transform_geographic(in_feats, best_solutions, out_feats, labels=None, transformer=None, clusterer=None):
    geo_feats = np.radians(in_feats[['latitude', 'longitude']])
    kms_per_radian = 6371.0088
    epsilon = 1.0 / kms_per_radian

    if not transformer:
        clusterer = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
        geo_clusters = clusterer.fit_predict(geo_feats)
        geo_clusters = np.resize(geo_clusters, (geo_clusters.shape[0], 1))
        transformer = best_solutions['lat_long']
        transformer.fit(geo_clusters, labels)
    else:
        geo_clusters = dbscan_predict(clusterer, geo_feats.as_matrix())

    feats_col = transformer.transform(geo_clusters)
             
    if out_feats is None:
        out_feats = feats_col
    else:
        out_feats = hstack([out_feats, feats_col])
    return out_feats, transformer, clusterer
