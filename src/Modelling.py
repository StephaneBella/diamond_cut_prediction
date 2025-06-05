from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def initialize_models():
    """Initialise les différents modèles"""
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=2),
        'Balanced Random Forest': BalancedRandomForestClassifier(
            random_state=2, sampling_strategy='auto', replacement=False, bootstrap=True),
        'LightGBM': LGBMClassifier(objective='multiclass', random_state=2, verbose=-1, is_unbalance=True),
        'ANN': create_ann_model()
    }
    return models

def create_ann_model(input_shape=None):
    """Crée le modèle de réseau de neurones"""
    model = Sequential()
    model.add(Dense(88, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model