from src.data_loader import load_config, load_data
from src.Preprocessing import preprocessing, split_data
from src.Modelling import initialize_models
from src.Evaluation import evaluate_model, save_model, save_model_params
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
def main(encoder=encoder):
    config = load_config()
    df = load_data(config['data']['processed_path'])
    
    X_train, X_test, y_train, y_test = split_data(df)
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    models = initialize_models()
    model = models['LightGBM']  # LightGBM s'est avéré etre le meilleur modèle

    evaluate_model(model, X_train, y_train, X_test, y_test)
    save_model(model, config['model']['path'])
    #save_model_params(model)
if __name__ == "__main__":
    main()
