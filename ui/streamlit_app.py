import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import base64
import os

import threading


# Fonction pour loader le csv
def load_data(dataset):
    df = pd.read_excel(dataset)
    return(df)

def main():
    
   # Appliquer le style CSS pour uniformiser les couleurs
    st.markdown(
        """
        <style>

            header[data-testid="stHeader"] {
            background-color: #808080 !important; 
            }
            /* Appliquer le fond bleu foncé à l'ensemble de l'application */
            .stApp {
                background-color: #808080 !important;
            }

            /* Appliquer le fond bleu foncé au sidebar */
            section[data-testid="stSidebar"] {
                background-color: #A9A9A9 !important;
            }

            /* Modifier le style du selectbox */
            div[data-baseweb="select"] {
                background-color: #A9A9A9 !important;
                color: white !important;
            }

            div[data-baseweb="select"] > div {
                background-color: #A9A9A9 !important;
            }

            div[data-baseweb="select"] * {
                color: white !important;
            }

             /* Modifier la couleur des titres pour plus de lisibilité */
            h1, h2, h3, h4, h5, h6 {
                color: #FFFFFF; /* Blanc */
            }

            /* Modifier la couleur du texte standard */
            .stMarkdown, .stText, .stAlert {
                color: #000000; /* Gris clair */
            }

            /* Modifier la couleur des widgets de saisie (input, select, etc.) */
            input, textarea, select {
                background-color: #003366; /* Bleu foncé */
                color: white;
            }
            
            /* Personnalisation des boutons */
            .stButton>button {
                background-color: #004080;
                color: white;
                border-radius: 8px;
                border: none;
            }

            .stButton>button:hover {
                background-color: #0055A4; /* Bleu plus clair au survol */
            }


            /* Assurer que les boutons suivent le style */
            .stButton>button {
                background-color: #1F2833 !important;
                color: white !important;
                border-radius: 10px;
                border: 1px solid white;
            }

            .stButton>button:hover {
                background-color: #45A29E !important;
                color: black !important;
            }
        </style>
         """,
            unsafe_allow_html=True
        )

    
    # Mettre la barre_latérale
    st.sidebar.image('C:/Users/DELL/Documents/VEMV/pycaret/work/Projets_professionnels/DiamondGradingModel/ui/diamant-couleur.JPG', width=400)
    
    st.markdown("<h1 style='text-align:center;color:brown;'>Application pour classer les diamants</h1>",
    unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:black;'>Informations sur les diamant </h2>",
    unsafe_allow_html=True)
    
    # Menu sur le sidebar
    menu = ['Accueil', 'Classer de nouveaux diamants']

    # Variable de selection de menus sur le sidebar
    choix = st.sidebar.selectbox('Menu',menu)

    if choix == "Accueil":
        col1, col2, cl3 = st.columns(3)
        with col2:
            st.image('C:/Users/DELL/Documents/VEMV/pycaret/work/Projets_professionnels/DiamondGradingModel/ui/diamant-image.jpg', width=500)

        st.info("Cette application vous permet de classer les diamants en fonction de leurs caractéristiques")

        st.markdown(
            """
            ####  Les critères de qualité d'un diamant expliqués par des diamantaires

            #####  Les critères de qualité d'un diamant

            Les diamants sont parmi les pierres précieuses les plus convoitées.  
            Pour bien choisir un diamant, il est essentiel de connaître ses caractéristiques essentielles.  

            ##### **Les 4 critères d’évaluation du diamant (4C)**

            * **Cut (Taille):** Détermine la manière dont le diamant reflète la lumière(la qualité de la coupe.

            * **Carat (Poids):** Indique le poids du diamant en carats.

            * **Color (Couleur)**

            * **Clarity (Pureté):** Évalue la présence d'inclusions et d'imperfections.
        """)

    if choix == "Classer de nouveaux diamants":
        t1, t2 = st.tabs([':bar_chart: classer une instance', ':clipboard:  classer un ensemble de diamants'])

        
        with t1:
            st.subheader('Classer une instance de diamant à partir de ses caractéristiques')
            st.markdown('Entrer les caractéristiques de votre nouveau diamant')

            with st.form("prediction_form"):
                carat = st.number_input("Poids (en carat)", min_value=0.0, format="%.2f")
                depth = st.number_input("Profondeur (Depth)", min_value=0.0, format="%.2f")
                table = st.number_input("Table", min_value=0.0, format="%.2f")
                x = st.number_input("Longueur (X)", min_value=0.0, format="%.2f")
                y = st.number_input("Largeur (Y)", min_value=0.0, format="%.2f")
                z = st.number_input("Hauteur (Z)", min_value=0.0, format="%.2f")
                color = st.selectbox("Couleur", options=["J", "I", "H", "G", "F", "E", "D"])
                clarity = st.selectbox("Clarté", options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
                

                submit = st.form_submit_button('Predire')
                

            if submit:
                url = "http://127.0.0.1:8000/predict"

                payload = {
                    "carat": carat, "depth": depth, "table": table,
                    "x": x, "y": y, "z": z,
                    "color": color, "clarity": clarity 
                }

                try:
                    response = requests.post(url, json=payload)

                    if response.status_code == 200:

                        result = response.json()

                        st.success(f"Ce diamant a été prédit de qualité : **{result['predicted_cut']}**")

                        # affichage des probabilites
                        proba_dict = result.get('probabilities', None)

                        if proba_dict:
                            classes = list(proba_dict.keys())
                            probas = list(proba_dict.values())

                            plt.style.use('dark_background')
                            plt.rcParams.update({
                                "figure.facecolor":  (0.12 , 0.12, 0.12, 1),
                                "axes.facecolor": (0.12 , 0.12, 0.12, 1),
                            })
                            fig, ax = plt.subplots()
                            ax.bar(classes, probas, color='skyblue') #,
                            ax.set_ylim([0, 1])
                            ax.set_ylabel('Probabilité')
                            ax.set_title('Distribution des probabilités')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)

                    else:

                        st.error(f"Erreur API : {response.status_code}")

                except Exception as e:

                    st.error(f"Erreur de connexion à l'API : {e}")
        with t2:
            
            st.subheader('Classer un ensemble de nouveux diamants à partir de leurs caractéristiques')

            uploaded_file = st.sidebar.file_uploader('Importer le fichier Excel contenant les données de vos nouveaux diamants', type=['xlsx', 'xls', 'csv'])
            
            if uploaded_file:
                # Lire le fichier excel dans un dataframe pandas
                try:
                    df = pd.read_excel(uploaded_file)
                    st.write('Appercu des données importées :')
                    st.dataframe(df)
                    
                    # Boutton pour lancer la prédiction sur l'ensemble
                    if st.button("Prédire l'ensemble de diamants"):

                        url = "http://127.0.0.1:8000/predict_batch"

                        # Convertir le DataFrame en liste de dicts (JSON serialisable)
                        payload = df.to_dict(orient='records')

                        try:
                            response = requests.post(url, json=payload)

                            if response.status_code == 200:

                                results = response.json()['predicted_cut'] 

                                # Ajouter les prédictions au DataFrame
                                df['predicted_cut'] = results

                                st.success("Vous trouverez les valeurs prédites à la dernière colonne")
                                st.dataframe(df)

                            else:
                                st.error(f"Erreur API:{response.status_code} - {response.text}")

                        except Exception as e:

                            st.error(f"Erreur de connection à l'API: {e}")

                except Exception as e:

                    st.error(f"Erreur lors de la lecture du fichier Excel: {e}")

if __name__ == '__main__':
    main()

    
    
    