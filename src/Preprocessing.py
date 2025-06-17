import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer



def delete_unnamed_column(df):
    """Supprime la colonne Unnamed: 0"""
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

def prepare_features_labels(df):
    """Prépare X et y en supprimant 'price' et 'cut'"""
    X = df.drop(['price', 'cut'], axis=1)
    y = df['cut']
    return X, y



def gestion_doublons(df):
    """Supprime les doublons"""
    return df.drop_duplicates()


def gestion_outliers(df):
    """Gestion des valeurs aberrantes"""
    df = df[(df['depth'] > 51.5) & (df['depth'] < 75) &
            (df['carat'] < 3.4) &
            (df['table'] > 48) & (df['table'] < 75) &
            (df['x'] > 2) & (df['y'] > 2) & (df['y'] < 15) &
            (df['z'] > 1) & (df['z'] < 10)]
    return df


# 1. Encodage personnalisé (clarity, color, cut)
class ManualCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.color_map = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}
        self.clarity_map = {'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7, 'I1': 8}
        self.cut_map = {'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4}

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['color'] = X['color'].map(self.color_map)
        X['clarity'] = X['clarity'].map(self.clarity_map)
        return X

# 2. Construction pipeline
def build_pipeline():
    return make_pipeline(ManualCategoricalEncoder())


def split_data(df):
    """Split avec stratification sur 'cut'"""
    X, y = prepare_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=True, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def preprocessing(df, output_path='data/processed/diamonds_processed.csv'):
    """Pipeline complet de preprocessing"""
    df = delete_unnamed_column(df)
    df = gestion_doublons(df)
    df = gestion_outliers(df)
    #df = cat_encoding(df)

    df.to_csv(output_path, index=False)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test


