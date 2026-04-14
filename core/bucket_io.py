"""
bucket_io.py — Lecture/écriture directe vers HuggingFace bucket
    Lors de la lecture des fichiers aucun fichier n'est téléchargé en local, tous sont lus depuis huggingface et seul les variables contenant
les informations necessaires sont retournées
    Lors de l'écriture, chaque fichier est tranferé sur huggingface depuis un repertoire temporaire, et détruit une fois le transfert éffectué
"""

import numpy as np
import json
import io
import os
import tempfile # pour la sauvegarde temporaire des fichiers en local avant son tranfert vers le bucket
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem

# Configuration
BUCKET_ID = "ktongue/DEM_MCM"
# BUCKET_PREFIX = "ResultsDtMCM"
BUCKET_PREFIX = "NewResultsMCM"
BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"

_fs = None
_api = None


def get_fs():
    global _fs
    if _fs is None:
        _fs = HfFileSystem()
    return _fs


def get_api():
    global _api
    if _api is None:
        _api = HfApi()
    return _api


# =============================================================================
# ÉCRITURE
# =============================================================================

def save_experiment_to_bucket(folder_name, matrix, stats, config, partitioner_data=None):
    """
    Sauvegarde tous les fichiers d'une expérience dans le bucket.
    """
    api = get_api()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        local_folder = Path(tmpdir)
        
        # Préparer tous les fichiers localement
        files_to_upload = []
        
        # Matrice
        matrix_path = local_folder / "transition_matrix.npy"
        np.save(matrix_path, matrix)
        files_to_upload.append((str(matrix_path), f"{BUCKET_PREFIX}/{folder_name}/transition_matrix.npy"))
        
        # Stats
        stats_path = local_folder / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        files_to_upload.append((str(stats_path), f"{BUCKET_PREFIX}/{folder_name}/stats.json"))
        
        # Config
        config_path = local_folder / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        files_to_upload.append((str(config_path), f"{BUCKET_PREFIX}/{folder_name}/config.json"))
        
        # Données du partitionneur
        if partitioner_data:
            for key, value in partitioner_data.items():
                if isinstance(value, np.ndarray):
                    file_path = local_folder / f"{key}.npy"
                    np.save(file_path, value)
                    files_to_upload.append((str(file_path), f"{BUCKET_PREFIX}/{folder_name}/{key}.npy"))
                else:
                    file_path = local_folder / f"{key}.json"
                    with open(file_path, "w") as f:
                        json.dump(value, f, indent=2)
                    files_to_upload.append((str(file_path), f"{BUCKET_PREFIX}/{folder_name}/{key}.json"))
        
        # Upload batch
        api.batch_bucket_files(
            bucket_id=BUCKET_ID,
            add=[(local_path, path_in_bucket) for local_path, path_in_bucket in files_to_upload],
        )


# =============================================================================
# LECTURE
# =============================================================================

def load_matrix_from_bucket(path):
    fs = get_fs()
    full_path = f"{BUCKET_BASE}/{path}"
    with fs.open(full_path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return np.load(buffer)


def load_json_from_bucket(path):
    fs = get_fs()
    full_path = f"{BUCKET_BASE}/{path}"
    with fs.open(full_path, "r") as f:
        return json.load(f)


def load_experiment_from_bucket(folder_name):
    return {
        "matrix": load_matrix_from_bucket(f"{folder_name}/transition_matrix.npy"),
        "stats": load_json_from_bucket(f"{folder_name}/stats.json"),
        "config": load_json_from_bucket(f"{folder_name}/config.json"),
    }


def list_experiments():
    fs = get_fs()
    try:
        items = fs.ls(BUCKET_BASE)
        return sorted([
            item["name"].split("/")[-1] 
            for item in items 
            if item["type"] == "directory"
        ])
    except FileNotFoundError:
        return []


def load_all_experiments():
    results = {}
    for folder in list_experiments():
        try:
            results[folder] = load_experiment_from_bucket(folder)
        except Exception as e:
            print(f"⚠️ {folder}: {e}")
    return results