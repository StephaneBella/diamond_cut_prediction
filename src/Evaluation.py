from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, make_scorer
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import os
import json


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="model"):
    """Évalue un modèle, affiche et enregistre les métriques et figures"""
    
    # Crée les dossiers outputs s'ils n'existent pas
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    # Variables communes
    fig_cm_path = f"outputs/figures/confusion_matrix_{model_name}.png"
    fig_learning_path = f"outputs/figures/learning_curve_{model_name}.png"
    metrics_path = f"outputs/metrics/metrics_{model_name}.json"

    if hasattr(model, '_is_compiled') or model.__class__.__name__ == 'Sequential':
        # ANN
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=120,
            validation_split=0.1,
            verbose=0
        )

        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Confusion matrix
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
        plt.title('Matrice de confusion')
        plt.grid(False)
        plt.savefig(fig_cm_path)
        plt.close()

        # Courbe d'apprentissage
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.xlabel('epochs')
        plt.ylabel('crossentropy')
        plt.title("Courbe d'apprentissage")
        plt.legend()
        plt.savefig(fig_learning_path)
        plt.close()

    else:
        # Modèle classique
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Confusion matrix
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
        plt.title('Matrice de confusion')
        plt.grid(False)
        plt.savefig(fig_cm_path)
        plt.close()

        # Courbe d'apprentissage
        kappa_scorer = make_scorer(cohen_kappa_score, weights='quadratic')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            scoring=kappa_scorer,
            cv=cv,
            train_sizes=np.linspace(0.1, 1, 10)
        )
        plt.plot(train_sizes, train_scores.mean(axis=1), label='training')
        plt.plot(train_sizes, val_scores.mean(axis=1), label='validation')
        plt.ylabel('Quadratic Weighted Kappa')
        plt.title("Courbe d'apprentissage")
        plt.legend()
        plt.savefig(fig_learning_path)
        plt.close()

    # Sauvegarde des métriques
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    kappa = cohen_kappa_score(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    results = {
        "classification_report": report,
        "cohen_kappa": kappa,
        "quadratic_weighted_kappa": qwk
    }

    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"[✔] Résultats sauvegardés dans :\n- {fig_cm_path}\n- {fig_learning_path}\n- {metrics_path}")



def save_model(model, output_path='models/model_cut_prediction.pkl'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def save_model_params(model):
    output_path = 'C:/Users/DELL/Documents/VEMV/pycaret/work/Diamond_cut_prediction/config/model_params.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
    else:
        params = {'model': 'Keras model - parameters in source code'}
    
    with open(output_path, 'w') as f:
            json.dump(params, f, indent=4)

    print(f"✅ Paramètres enregistrés dans : {output_path}")
