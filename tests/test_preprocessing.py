import pytest
import pandas as pd
from src.Preprocessing import delete_unnamed_column, gestion_doublons, cat_encoding, gestion_outliers, preprocessing

def test_delete_unnamed_column():
    df = pd.DataFrame({'Unnamed: 0': [1,2], 'a': [3,4]})
    df2 = delete_unnamed_column(df.copy())
    assert 'Unnamed: 0' not in df2.columns

def test_gestion_doublons():
    df = pd.DataFrame({'a': [1,1,2]})
    df2 = gestion_doublons(df.copy())
    assert df2.shape[0] == 2

def test_cat_encoding():
    df = pd.DataFrame({
        'color': ['D', 'E'],
        'clarity': ['IF', 'VVS1'],
        'cut': ['Fair', 'Good']
    })
    df2 = cat_encoding(df.copy())
    assert df2['color'].iloc[0] == 1
    assert df2['clarity'].iloc[0] == 1
    assert df2['cut'].iloc[0] == 0

def test_gestion_outliers():
    df = pd.DataFrame({
        'depth': [52, 80],
        'carat': [1, 5],
        'table': [50, 80],
        'x': [3, 1],
        'y': [3, 20],
        'z': [2, 11]
    })
    df2 = gestion_outliers(df.copy())
    assert df2.shape[0] == 1

def test_preprocessing(tmp_path):
    df = pd.DataFrame({
        'Unnamed: 0': [0],
        'depth': [60],
        'carat': [1],
        'table': [55],
        'x': [5],
        'y': [5],
        'z': [5],
        'color': ['D'],
        'clarity': ['IF'],
        'cut': ['Fair']
    })
    output_path = tmp_path / "test.csv"
    X, y = preprocessing(df.copy(), output_path=str(output_path))
    assert 'Unnamed: 0' not in X.columns
    assert y.iloc[0] == 0
    assert output_path.exists()
