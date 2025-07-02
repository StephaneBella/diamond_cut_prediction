# Utilisation d'une image officielle Python légère
FROM python:3.10-slim

# Définition du répertoire de travail dans le contener
WORKDIR / app

# Copie du fichier requirements.txt 
COPY requirements.txt .

# Installation les dépendances 
RUN pip install --no-cache-dir -r requirements.txt

# Copie des documents nécessaires pour l'app dans le container
# Copier le backend FastAPI
COPY app/ ./app/

# Copier le frontend Streamlit
COPY ui/ .ui/ 

# Copier les modèles ML
COPY models/ ./models/

# Définir PYTHONPATH pour que Python trouve le package src
ENV PYTHONPATH=/app/src

# Exposition du port streamlit attendu par Hugging Face
EXPOSE 7860

# lancement de l'API FastAPI et de l'app stremlit
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run streamlit_app/app.py --server.port 7860 --server.address 0.0.0.0