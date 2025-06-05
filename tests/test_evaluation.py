import numpy as np
import pandas as pd
from src.evaluation import evaluate_model
from sklearn.tree import DecisionTreeClassifier

def test_evaluate_model_classic():
    X_train = pd.DataFrame(np.random.rand(20, 3))
    y_train = np.random.randint(0, 2, 20)
    X_test = pd.DataFrame(np.random.rand(10, 3))
    y_test = np.random.randint(0, 2, 10)
    
    model = DecisionTreeClassifier()
    evaluate_model(model, X_train, y_train, X_test, y_test)

def test_evaluate_model_ann():
    from src.Modelling import create_ann_model
    import numpy as np
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 5, 20)
    X_test = np.random.rand(10, 5)
    y_test = np.random.randint(0, 5, 10)
    
    model = create_ann_model(input_shape=(5,))
    evaluate_model(model, X_train, y_train, X_test, y_test)
