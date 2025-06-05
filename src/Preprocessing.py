import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def delete_unnamed_column(df):
    """Supprime la colonne Unnamed: 0"""
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

def prepare_features_labels(df):
    """Prépare X et y en supprimant 'price' et 'cut'"""
    X = df.drop(['price', 'cut'], axis=1)
    y = df['cut']
    return X, y

def split_data(df):
    """Split avec stratification sur 'cut'"""
    X, y = prepare_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=True, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def gestion_doublons(df):
    """Supprime les doublons"""
    return df.drop_duplicates()

def cat_encoding(df):
    """Encodage des variables catégorielles"""
    color_map = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}
    clarity_map = {'IF':1, 'VVS1':2, 'VVS2':3, 'VS1':4, 'VS2':5, 'SI1':6, 'SI2':7, 'I1':8}
    cut_map = {'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4}
    
    df['color'] = df['color'].map(color_map)
    df['clarity'] = df['clarity'].map(clarity_map)
    df['cut'] = df['cut'].map(cut_map)
    return df

def gestion_outliers(df):
    """Gestion des valeurs aberrantes"""
    df = df[(df['depth'] > 51.5) & (df['depth'] < 75) &
            (df['carat'] < 3.4) &
            (df['table'] > 48) & (df['table'] < 75) &
            (df['x'] > 2) & (df['y'] > 2) & (df['y'] < 15) &
            (df['z'] > 1) & (df['z'] < 10)]
    return df



def preprocessing(df, output_path='data/processed/diamonds_processed.csv'):
    """Pipeline complet de preprocessing"""
    df = delete_unnamed_column(df)
    df = gestion_doublons(df)
    df = gestion_outliers(df)
    df = cat_encoding(df)
    df.to_csv(output_path, index=False)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test
