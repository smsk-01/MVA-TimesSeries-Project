import os
import requests
import zipfile
import io
import pandas as pd

def download_and_load_spikefinder():
    """
    Télécharge le ZIP officiel du challenge depuis les serveurs Amazon S3.
    Extrait les fichiers et charge le dataset d'entraînement n°1.
    """
    # URL directe vers le ZIP officiel des données d'entraînement
    url_zip = "https://s3.amazonaws.com/neuro.datasets/challenges/spikefinder/spikefinder.train.zip"
    
    # Dossier où l'on va tout extraire
    extract_folder = "spikefinder_data"
    
    # Fichiers cibles (Dataset 1)
    file_calcium = os.path.join(extract_folder, "spikefinder.train", "1.train.calcium.csv")
    file_spikes = os.path.join(extract_folder, "spikefinder.train", "1.train.spikes.csv")

    # 1. Téléchargement et Extraction
    if not os.path.exists(file_calcium):
        print("Téléchargement du ZIP officiel Spikefinder (Amazon S3)...")
        try:
            r = requests.get(url_zip)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(extract_folder)
                print("Extraction terminée.")
            else:
                print(f"Erreur de téléchargement : Code {r.status_code}")
                return None, None
        except Exception as e:
            print(f"Erreur critique : {e}")
            return None, None
    else:
        print("Données locales trouvées.")

    # 2. Chargement Pandas
    print(f"Chargement de {file_calcium}...")
    df_calcium = pd.read_csv(file_calcium)
    df_spikes = pd.read_csv(file_spikes)
    
    return df_calcium, df_spikes

# Exécution
df_calcium, df_spikes = download_and_load_spikefinder()

if df_calcium is not None:
    # Nettoyage des colonnes vides (NaN)
    valid_cols = df_calcium.notna().all()
    clean_calcium = df_calcium.loc[:, valid_cols]
    clean_spikes = df_spikes.loc[:, valid_cols]
    
    print(f"SUCCÈS : {clean_calcium.shape[1]} neurones chargés et prêts.")
else:
    print("ÉCHEC : Impossible de récupérer les données.")