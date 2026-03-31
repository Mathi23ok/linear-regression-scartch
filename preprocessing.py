import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

def load_data(path):
    df = pd.read_csv(path)
    return df

def handle_missing(df):
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    return df

def encode_features(df):
    df = pd.get_dummies(df,columns=['Neighborhood'],drop_first=True)
    return df

def feature_engineering(df):
    current_year = 2026
    df['Age'] = current_year - df['YearBuilt']
    df.drop('YearBuilt', axis=1, inplace=True)
    df['Area_per_room'] = df['SquareFeet'] / (df['Bedrooms'] + 1)
    df['Bedrooms_sq'] = df['Bedrooms'] ** 2
    df['Bathrooms_sq'] = df['Bathrooms'] ** 2
    df['Age_sq'] = df['Age'] ** 2
    df['Area_Bedrooms'] = df['SquareFeet'] * df['Bedrooms']
    df['Area_Age'] = df['SquareFeet'] * df['Age']
    #df['SquareFeet_sq'] = df['SquareFeet'] ** 2
    return df

def split_feature_target(df):
    y = df['Price'].values
    X = df.drop('Price', axis=1).astype(float).values
    return X, y

def train_test_split(X,y,test_size = 0.2):
    n = len(X)
    split_index = int(n * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def scale_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std