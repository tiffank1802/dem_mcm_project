"""
===================================================================================
SWEEP MARKOVIEN — Lance les calculs pour un type de partitionnement donné
===================================================================================

Usage:
    python run_sweep.py --method voronoi
    python run_sweep.py --method cartesian
    python run_sweep.py --method all
    python run_sweep.py --method voronoi --list   # liste les configs sans lancer

Depuis Python:
    from run_sweep import run_markov_sweep
    run_markov_sweep("cylindrical")
===================================================================================
"""

import os
import json
import argparse
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from huggingface_hub import HfFileSystem

# from partitioners import create_partitioner, REGISTRY
# from bucket_io import save_experiment_to_bucket, BUCKET_BASE
from .partitioners import create_partitioner, REGISTRY
from .bucket_io import save_experiment_to_bucket, BUCKET_BASE



# =============================================================================
# CONFIGURATION GÉNÉRALE
# =============================================================================

# BASE_OUTPUT_DIR = "NewResultsMCM"
BASE_OUTPUT_DIR = "ResultsDtMCM"
HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
SAMPLE_RATE = 50  # pour le fit des partitionneurs


# =============================================================================
# DATACLASS EXPÉRIENCE
# =============================================================================


@dataclass # crée et ajoute automatiquement le constructeur de classe
class ExperimentConfig:
    """Configuration d'une expérience."""

    method: str = "cartesian"
    method_kwargs: dict = field(default_factory=dict) # type par defaut de la (dict vide) lors de l'instanciation de la classe ExperimentConfig sans passage explicite de method_kwargs
    nlt: int = 100
    dt:float=None
    step_size: int = 1 # pas de temps d'apprentissage telque le temps d'apprentissage soit T=nlt*step_size
    start_index: int = 250 # début de l'apprentissage

    def __post_init__(self):
        if self.method_kwargs is None:
            self.method_kwargs = {}
        if self.dt is None:
            # Par défaut: 1/10 du step_size, minimum 1
            # self.dt = min(1, self.step_size // 10)    
            self.dt=.1 
            
    def output_folder(self, base_dir=BASE_OUTPUT_DIR,sample_coords=None):
        part = create_partitioner(self.method, **self.method_kwargs)
        if sample_coords is not None:
            part.fit(sample_coords)
        return os.path.join(
            base_dir,
            f"{part.label}_NLT{self.nlt}_step{self.step_size}_start{self.start_index}_dt{self.dt}",
        )
   


# =============================================================================
# CONFIGURATIONS PAR MÉTHODE
# =============================================================================


def get_configs(method):
    """
    Retourne la liste de configs pour une méthode donnée.

    Axes de sweep:
      1. Paramètres de discrétisation (propres à chaque méthode)
      2. Nombre de pas de temps (NLT)
      3. Pas de sous-échantillonnage temporel (step_size)
      4. Index de départ (start_index)
      5. Pas de glissement (dt)
    """

    configs = []

    # ══════════════════════════════════════════════════════════════════════
    # Sweep de discrétisation spatiale
    # ══════════════════════════════════════════════════════════════════════

    if method == "cartesian":
        for n in [2, 3, 4, 5]:
            configs.append(
                ExperimentConfig(
                    method="cartesian",
                    method_kwargs={"nx": n, "ny": n, "nz": n},
                )
            )

    elif method == "cylindrical":
        # nr variable (axisymétrique pur)
        for nr in [3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": nr, "ntheta": 1, "nz": 1,
                        "radial_mode": "equal_area",
                    },
                )
            )
        # ntheta variable
        for nth in [1, 2, 3, 4]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 2, "ntheta": nth, "nz": 1,
                        "radial_mode": "equal_area",
                    },
                )
            )
        # nz variable
        for nz in [1,2]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 2, "ntheta": 2, "nz": nz,
                        "radial_mode": "equal_area",
                    },
                )
            )
        # equal_dr vs equal_area
        for mode in ["equal_dr", "equal_area"]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 2, "ntheta": 2, "nz": 1,
                        "radial_mode": mode,
                    },
                )
            )

    elif method == "voronoi":
        for nc in [8,10,12,14,16,18,20,24, 27,30, 64, 100]:
            configs.append(
                ExperimentConfig(
                    method="voronoi",
                    method_kwargs={"n_cells": nc},
                )
            )

    elif method == "quantile":
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            configs.append(
                ExperimentConfig(
                    method="quantile",
                    method_kwargs={"nx": n, "ny": n, "nz": 1},
                )
            )

    elif method == "octree":
        # max_particles variable
        for mp in [2, 4, 8, 16, 32, 64, 100]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": mp, "max_depth": 5},
                )
            )
        # max_depth variable
        for md in [3, 4, 5, 6, 7]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": 100, "max_depth": md},
                )
            )

    elif method == "physics":
        for nc in [2, 4, 8, 16, 32, 64, 100]:
            configs.append(
                ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": nc},
                )
            )

    elif method == "adaptive":
        # ── Sweep z_split (quantile) ─────────────────────────────────
        for z_q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": z_q,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 3, "ntheta": 3, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Sweep finesse zone basse (nr) ────────────────────────────
        for nr in [3, 5, 8, 10, 15]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": nr, "ntheta": 2, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Sweep finesse zone basse (nz) ────────────────────────────
        for nz in [1,4, 2, 3]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 2, "ntheta": 2, "nz": nz,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Sweep ntheta zone basse ──────────────────────────────────
        for nth in [1, 4, 8, 12, 16]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 2, "ntheta": nth, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Zone haute avec quelques cellules ────────────────────────
        for n_top in [1, 2, 4,3]:
            top_method = "single" if n_top == 1 else "cylindrical"
            top_kwargs = {} if n_top == 1 else {
                "nr": 1, "ntheta": n_top, "nz": 1,
                "radial_mode": "equal_area",
            }
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": n_top,
                        "top_method": top_method,
                        "top_kwargs": top_kwargs,
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 2, "ntheta": 2, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Voronoï en bas au lieu de cylindrique ────────────────────
        for nc in [64, 125, 250, 500]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "voronoi",
                        "bottom_kwargs": {"n_cells": nc},
                    },
                )
            )

    elif method == "multizone":
        # ── 2 zones: fin en bas, grossier en haut ────────────────────
        configs.append(
            ExperimentConfig(
                method="multizone",
                method_kwargs={
                    "z_mode": "quantile",
                    "zones": [
                        {
                            "z_min": 0.0, "z_max": 0.8,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 2, "ntheta": 2, "nz": 1,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.8, "z_max": 1.0,
                            "method": "single",
                            "kwargs": {},
                        },
                    ],
                },
            )
        )

        # ── 3 zones: gradient de finesse ─────────────────────────────
        for split1, split2 in [(0.5, 0.8), (0.6, 0.85), (0.7, 0.9)]:
            configs.append(
                ExperimentConfig(
                    method="multizone",
                    method_kwargs={
                        "z_mode": "quantile",
                        "zones": [
                            {
                                "z_min": 0.0, "z_max": split1,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": 2, "ntheta": 2, "nz": 1,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": split1, "z_max": split2,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": 2, "ntheta": 2, "nz": 1,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": split2, "z_max": 1.0,
                                "method": "single",
                                "kwargs": {},
                            },
                        ],
                    },
                )
            )

        # ── 3 zones avec Voronoï en bas ──────────────────────────────
        for nc_bottom in [2, 4, 8, 16, 32, 64]:
            configs.append(
                ExperimentConfig(
                    method="multizone",
                    method_kwargs={
                        "z_mode": "quantile",
                        "zones": [
                            {
                                "z_min": 0.0, "z_max": 0.6,
                                "method": "voronoi",
                                "kwargs": {"n_cells": nc_bottom},
                            },
                            {
                                "z_min": 0.6, "z_max": 0.85,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": 2, "ntheta": 2, "nz": 1,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": 0.85, "z_max": 1.0,
                                "method": "single",
                                "kwargs": {},
                            },
                        ],
                    },
                )
            )

        # ── 4 zones (très gradué) ────────────────────────────────────
        configs.append(
            ExperimentConfig(
                method="multizone",
                method_kwargs={
                    "z_mode": "quantile",
                    "zones": [
                        {
                            "z_min": 0.0, "z_max": 0.4,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 8, "ntheta": 16, "nz": 10,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.4, "z_max": 0.7,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 5, "ntheta": 10, "nz": 6,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.7, "z_max": 0.9,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 3, "ntheta": 6, "nz": 3,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.9, "z_max": 1.0,
                            "method": "single",
                            "kwargs": {},
                        },
                    ],
                },
            )
        )

        # ── Sweep nb cellules zone basse ─────────────────────────────
        for nr, nz in [(3, 5), (5, 8), (8, 10), (10, 12)]:
            configs.append(
                ExperimentConfig(
                    method="multizone",
                    method_kwargs={
                        "z_mode": "quantile",
                        "zones": [
                            {
                                "z_min": 0.0, "z_max": 0.75,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": nr, "ntheta": 8, "nz": nz,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": 0.75, "z_max": 1.0,
                                "method": "single",
                                "kwargs": {},
                            },
                        ],
                    },
                )
            )

    elif method == "single":
        configs.append(
            ExperimentConfig(
                method="single",
                method_kwargs={},
            )
        )

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    # ══════════════════════════════════════════════════════════════════════
    # Sweeps temporels (avec discrétisation spatiale par défaut)
    # ══════════════════════════════════════════════════════════════════════

    default_kwargs = _get_default_kwargs(method)

    # ── Sweep NLT ────────────────────────────────────────────────────────
    for nlt in [10, 20, 50, 100, 150, 200, 300, 500]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                nlt=nlt,
            )
        )

    # ── Sweep step_size ──────────────────────────────────────────────────
    for step in [1, 2, 3, 5, 8, 10, 15, 20]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                step_size=step,
            )
        )

    # ── Sweep start_index ────────────────────────────────────────────────
    for start in [250, 500, 1000, 2000, 3000, 5000]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                start_index=start,
            )
        )

    # ── Sweep dt ─────────────────────────────────────────────────────────
    for dt in [.01, .05, .10, .25, .50]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                dt=dt,
            )
        )

    # ── Dédoublonner ─────────────────────────────────────────────────────
    seen = set()
    unique = []
    for c in configs:
        key = c.output_folder()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def _get_default_kwargs(method):
    """Paramètres de discrétisation par défaut pour les sweeps temporels."""
    defaults = {
        "cartesian": {"nx": 5, "ny": 5, "nz": 5},
        "cylindrical": {
            "nr": 5, "ntheta": 8, "nz": 5,
            "radial_mode": "equal_area",
        },
        "voronoi": {"n_cells": 125},
        "quantile": {"nx": 5, "ny": 5, "nz": 5},
        "octree": {"max_particles": 100, "max_depth": 5},
        "physics": {"n_cells": 125},
        "adaptive": {
            "z_split": 0.75,
            "z_split_mode": "quantile",
            "n_cells_top": 1,
            "top_method": "single",
            "top_kwargs": {},
            "bottom_method": "cylindrical",
            "bottom_kwargs": {
                "nr": 2, "ntheta": 2, "nz": 1,
                "radial_mode": "equal_area",
            },
        },
        "multizone": {
            "z_mode": "quantile",
            "zones": [
                {
                    "z_min": 0.0, "z_max": 0.75,
                    "method": "cylindrical",
                    "kwargs": {
                        "nr": 2, "ntheta": 2, "nz": 1,
                        "radial_mode": "equal_area",
                    },
                },
                {
                    "z_min": 0.75, "z_max": 1.0,
                    "method": "single",
                    "kwargs": {},
                },
            ],
        },
        "single": {},
    }
    return defaults.get(method, {})

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================


def sample_coordinates(files, fs, sample_rate=SAMPLE_RATE):
    """
    Échantillonne des coordonnées pour le fit des partitionneurs.

    Returns:
        np.ndarray shape (N, 3)
    """
    all_coords = []
    for f in tqdm(files[::sample_rate], desc="   Échantillonnage", leave=False):
        with fs.open(f, "rb") as fh:
            df = pl.read_csv(fh)
        coords = np.column_stack(
            [
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ]
        )
        all_coords.append(coords)
    return np.vstack(all_coords)


# =============================================================================
# CALCUL MATRICE DE TRANSITION
# =============================================================================




# def phi_particule(state: int, partition: int) -> bool:
#     """Vérifie si une particule est bien dans une partition"""
#     return 1 if state == partition else 0

# def phi_sum_partition(states, partition: int) -> int:
#     """Somme les particules qui sont dans une partition"""
#     phi_s = 0
#     for i in range(len(states)):
#         phi_s += phi_particule(states[i], partition=partition)
#     return phi_s

# def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu"):
#     """
#     Calcule P_n pour un timestep en utilisant phi_particule et phi_sum_partition.
#     Normalisation par colonnes (somme des colonnes = 1).
#     """
#     # Conversion en tensor si nécessaire
#     if isinstance(states_curr, np.ndarray):
#         states_curr = torch.from_numpy(states_curr)
#     if isinstance(states_prev, np.ndarray):
#         states_prev = torch.from_numpy(states_prev)
    
#     s_prev = states_prev.to(device).long()
#     s_curr = states_curr.to(device).long()
    
#     # Initialisation de la matrice de transition
#     P = torch.zeros((n_states, n_states), device=device, dtype=torch.float64)
    
#     # Calcul des transitions P[i,j] = probabilité d'aller de i à j
#     for i in range(n_states):
#         for j in range(n_states):
#             # Compte les transitions de i vers j
#             inter = 0
#             n = min(len(s_prev), len(s_curr))
#             for p in range(n):
#                 inter += phi_particule(state=s_prev[p].item(), partition=i) * phi_particule(state=s_curr[p].item(), partition=j)
            
#             # Normalisation par le nombre de particules dans l'état i au temps précédent
#             denominator = phi_sum_partition(s_prev.cpu().numpy(), i)
#             P[i, j] = inter / denominator if denominator > 0 else 0.0
    
#     # Transposition pour avoir les états courants en lignes, précédents en colonnes
#     P = P.T
    
#     # # Normalisation par colonnes (somme des colonnes = 1) avec torch.sum(dim=0)
#     # col_sums = torch.sum(P, dim=0)
    
#     # P = torch.where(col_sums > 0, P / col_sums, torch.zeros_like(P))
    
#     return P

import torch

def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu"):
    """
    Calcule P_n pour un timestep - version entièrement vectorisée.
    P[j,i] = probabilité de transition de l'état i vers l'état j
    """
    # Conversion en tensor si nécessaire
    if isinstance(states_prev, np.ndarray):
        states_prev = torch.from_numpy(states_prev)
    if isinstance(states_curr, np.ndarray):
        states_curr = torch.from_numpy(states_curr)
    
    s_prev = states_prev.to(device).long()
    s_curr = states_curr.to(device).long()
    
    n = min(len(s_prev), len(s_curr))
    s_prev = s_prev[:n]
    s_curr = s_curr[:n]
    
    # Création des masques one-hot pour chaque particule
    # phi_prev[p, i] = 1 si particule p était dans état i
    # phi_curr[p, j] = 1 si particule p est dans état j
    phi_prev = (s_prev.unsqueeze(1) == torch.arange(n_states, device=device)).float()  # (n, n_states)
    phi_curr = (s_curr.unsqueeze(1) == torch.arange(n_states, device=device)).float()  # (n, n_states)
    
    # Matrice de co-occurrence : transitions[i, j] = nombre de transitions i → j
    # Somme sur toutes les particules de phi_prev[:, i] * phi_curr[:, j]
    transitions = phi_prev.T @ phi_curr  # (n_states, n_states)
    
    # Dénominateur : nombre de particules dans chaque état au temps précédent
    denominator = phi_prev.sum(dim=0)  # (n_states,)
    
    # P[i, j] = transitions[i, j] / denominator[i]
    # P = transitions.T / denominator.unsqueeze(1).clamp(min=1e-10)
    P=transitions.T/denominator

    
    # Mettre à zéro les lignes sans particules
    P[denominator == 0] = 0.0
    
    # Transposition pour avoir P[j, i] = prob(i → j)
    # P = P.T
    
    return P.to(torch.float64)





# =============================================================================
# EXPÉRIENCE
# =============================================================================

def run_experiment(config, partitioner, files, fs, device):
    """
    Exécute une expérience complète avec fenêtre glissante.

    Logique temporelle:
        Chaque paire utilise deux snapshots séparés de step_size.
        La fenêtre glisse de dt fichiers entre chaque paire.
        dt << step_size → beaucoup de paires avec fort recouvrement.

    Exemple: step_size=50, dt=5, start=0, nlt=6

        Fichiers:  0    5   10   15   20   25   ...  50   55   60   ...  75
                   │    │    │    │    │    │         │    │    │         │
        Paire 0:   ●────────────────────────────────→●
                   0                                 50
        Paire 1:        ●────────────────────────────────→●
                        5                                 55
        Paire 2:             ●────────────────────────────────→●
                            10                                 60
        Paire 3:                  ●────────────────────────────────→●
                                 15                                 65
        Paire 4:                       ●────────────────────────────────→●
                                      20                                 70
        Paire 5:                            ●────────────────────────────────→●
                                           25                                 75

        Toutes les paires ont le MÊME écart temporel (step_size=50).
        Le glissement de dt=5 raffine l'échantillonnage statistique.

        P_final = (P_0 + P_1 + P_2 + P_3 + P_4 + P_5) / 6
    """
    n_states = partitioner.n_cells
    step = config.step_size
    dt = config.dt

    # ── Vérifier la faisabilité ──
    # Dernier fichier nécessaire : start + (nlt-1)*dt + step
    last_needed = config.start_index + (config.nlt - 1) * dt + step

    if last_needed >= len(files):
        max_nlt = (len(files) - 1 - config.start_index - 1*step) // dt + 1
        max_nlt = max(max_nlt, 0)
        print(
            f"   ⚠️  Seulement {max_nlt} paires possibles "
            f"(demandé: {config.nlt}, step={step}, dt={dt})"
        )
        actual_nlt = max_nlt
    else:
        actual_nlt = config.nlt

    if actual_nlt <= 0:
        raise ValueError(
            f"Aucune paire possible: start={config.start_index}, "
            f"step={step}, dt={dt}, fichiers={len(files)}"
        )

    # ── Construire les paires (fenêtre glissante) ──
    #
    #   Paire k : (start + k*dt,  start + k*dt + step)
    #
    #   L'écart entre les deux snapshots d'une paire est TOUJOURS = step
    #   Le décalage entre paires successives est = dt
    #
    pairs: list[tuple[int, int]] = []
    for k in range(actual_nlt):
        idx_prev = config.start_index + k 
        idx_curr = idx_prev + step
        pairs.append((idx_prev, idx_curr))

    # ── Diagnostic ──
    ratio = dt / step
    if ratio < 1:
        overlap_pct = round((1 - ratio) * 100, 1)
        n_paires_par_step = int(step / dt)
        print(f"   🔄 Recouvrement: {overlap_pct}% "
              f"({n_paires_par_step} paires par intervalle de step_size)")
    elif ratio == 1:
        print(f"   ▶️  Paires bout à bout (dt == step)")
    else:
        gap = dt - step
        print(f"   ⏭️  Trou de {gap} fichiers entre paires")

    plage_temporelle = pairs[-1][1] - pairs[0][0]
    print(
        f"   📐 {actual_nlt} paires | step={step} | dt={dt}\n"
        f"   📂 Paire 0:   files[{int(pairs[0][0]):5d}] → files[{int(pairs[0][1]):5d}]\n"
        f"   📂 Paire {actual_nlt-1}:  files[{int(pairs[-1][0]):5d}] → files[{int(pairs[-1][1]):5d}]\n"
        f"   📏 Plage temporelle couverte: {plage_temporelle} fichiers"
    )

    # ── Accumulateur ──
    P_acc = torch.zeros(
        (n_states, n_states), dtype=torch.float64, device=device
    )

    for idx_prev, idx_curr in tqdm(pairs, desc="   Paires", leave=False):
        # Lecture des deux snapshots séparés de step_size
        with fs.open(files[idx_prev], "rb") as f:
            df_prev = pl.read_csv(f)
        with fs.open(files[idx_curr], "rb") as f:
            df_curr = pl.read_csv(f)

        # Assignation des états
        states_prev = partitioner.compute_states(
            df_prev["coordinates:0"],
            df_prev["coordinates:1"],
            df_prev["coordinates:2"],
        )
        states_curr = partitioner.compute_states(
            df_curr["coordinates:0"],
            df_curr["coordinates:1"],
            df_curr["coordinates:2"],
        )

        # Vers le device
        sp = torch.from_numpy(states_prev).to(device)
        sc = torch.from_numpy(states_curr).to(device)

        P_acc [:]+= compute_P_matrix_torch(sp, sc, n_states, device)

    # ── Moyenne sur les nlt évaluations ──
    P = P_acc / actual_nlt
    P_np = P.cpu().numpy()

    # ── Statistiques ──
    column_sums = P_np.sum(axis=0)
    visited = column_sums > 0
    diag = np.diag(P_np)

    stats = {
        "n_timesteps_used": actual_nlt,
        "n_states": n_states,
        "n_states_visited": int(visited.sum()),
        "n_states_empty": int((~visited).sum()),
        "fraction_visited": round(float(visited.sum()) / n_states, 4),
        "column_sum_min": float(column_sums[visited].min()) if visited.any() else 0,
        "column_sum_max": float(column_sums[visited].max()) if visited.any() else 0,
        "column_sum_mean": float(column_sums[visited].mean()) if visited.any() else 0,
        "diagonal_mean": float(diag.mean()),
        "diagonal_std": float(diag.std()),
        "method": config.method,
        "step_size": step,
        "dt": dt,
        "overlap_ratio": round(max(0, 1 - dt / step), 4),
        "n_paires_par_step": step // dt if dt > 0 else 0,
        "plage_temporelle": int(pairs[-1][1] - pairs[0][0]),
        "start_index": config.start_index,
        "first_pair": list(pairs[0]),
        "last_pair": list(pairs[-1]),
    }

    return P_np, stats

def save_results(config, partitioner, P, stats, output_dir):
    """Sauvegarde les résultats dans le bucket HuggingFace."""
    
    folder_name = os.path.basename(output_dir)
    
    # Préparer les données du partitionneur
    partitioner_data = {}
    if hasattr(partitioner, 'centroids') and partitioner.centroids is not None:
        partitioner_data["centroids"] = partitioner.centroids
    if hasattr(partitioner, '_r_edges') and partitioner._r_edges is not None:
        partitioner_data["r_edges"] = partitioner._r_edges
    if hasattr(partitioner, '_leaves') and partitioner._leaves:
        partitioner_data["leaves"] = np.array(partitioner._leaves)
    if hasattr(partitioner, '_x_edges') and partitioner._x_edges is not None:
        partitioner_data["x_edges"] = partitioner._x_edges
        partitioner_data["y_edges"] = partitioner._y_edges
        partitioner_data["z_edges"] = partitioner._z_edges
    
    # Métadonnées du partitionneur
    partitioner_data["partitioner_meta"] = {
        "type": type(partitioner).__name__,
        "label": partitioner.label,
        "n_cells": partitioner.n_cells,
    }
    
    # Sauvegarder dans le bucket
    save_experiment_to_bucket(
        folder_name=folder_name,
        matrix=P,
        stats=stats,
        config=asdict(config),
        partitioner_data=partitioner_data,
    )
    
    print(f"   💾 Bucket: {BUCKET_BASE}/{folder_name}/")
# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def run_markov_sweep(method:str, configs:list[ExperimentConfig]=None, base_dir=BASE_OUTPUT_DIR)-> list[dict]:
    """
    Lance le sweep Markovien pour une méthode de partitionnement.

    Args:
        method: str — "cartesian", "cylindrical", "voronoi",
                       "quantile", "octree", "physics", ou "all"
        configs: liste de ExperimentConfig (None = configs par défaut)
        base_dir: dossier de sortie

    Exemple:
        run_markov_sweep("voronoi")
        run_markov_sweep("cylindrical", configs=[
            ExperimentConfig(method="cylindrical",
                             method_kwargs={"nr":10, "ntheta":8, "nz":10}),
        ])
    """

    print("=" * 70)
    print(f"  SWEEP MARKOVIEN — méthode: {method.upper()}")
    print("=" * 70)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Fichiers ──
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))
    print(f"📁 Fichiers disponibles: {len(files)}")

    # ── Coordonnées pour fit ──
    print("\n🔍 Échantillonnage des coordonnées pour le fit...")
    sample_coords = sample_coordinates(files, fs)
    print(f"   {len(sample_coords)} points échantillonnés")

    # ── Configs ──
    if method == "all":
        methods = list(REGISTRY.keys())
    else:
        methods = [method]

    if configs is None:
        all_configs = []
        for m in methods:
            all_configs.extend(get_configs(m))
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} expériences à lancer:")
    print("-" * 70)
    for i, c in enumerate(all_configs):
        part = create_partitioner(c.method, **c.method_kwargs)
        part.fit(sample_coords)
        print(
            f"  {i + 1:3d}. [{c.method:12s}] {part.label:40s} "
            f"NLT={c.nlt:4d} step={c.step_size:2d} start={c.start_index}"
        )
    print("-" * 70)

    # ── Cache des partitionneurs fittés ──
    fitted_cache = {}

    # ── Boucle principale ──
    results = []
    for i, config in enumerate(all_configs):
        if config.method=="adaptive" or config.method=="multizone":
            output_dir=config.output_folder(base_dir=base_dir,sample_coords=sample_coords)
        else :
            output_dir = config.output_folder(base_dir)
        print(f"\n[{i + 1}/{len(all_configs)}] {os.path.basename(output_dir)}")

        try:
            # Créer ou récupérer le partitionneur
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            cache_key = partitioner.label

            if cache_key in fitted_cache:
                partitioner = fitted_cache[cache_key]
                print(f"   ♻️  Partitionneur en cache: {cache_key}")
            else:
                print(f"   🔧 Fit: {cache_key}...")
                partitioner.fit(sample_coords)
                fitted_cache[cache_key] = partitioner

                # Diagnostics
                diag = partitioner.diagnostics(sample_coords)
                print(
                    f"   📊 {partitioner.n_cells} cellules | "
                    f"{diag['n_visited']} visitées | "
                    f"pop: [{diag['pop_min']}, {diag['pop_max']}] "
                    f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}"
                )

            # Lancer l'expérience
            P, stats = run_experiment(config, partitioner, files, fs, device)

            # Sauvegarder
            save_results(config, partitioner, P, stats, output_dir)

            results.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                f"P(rester)={stats['diagonal_mean']:.4f} | "
                f"Σrow=[{stats['column_sum_min']:.4f}, {stats['column_sum_max']:.4f}]"
            )

        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # ── Résumé ──
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    ok = [r for r in results if r["success"]]
    ko = [r for r in results if not r["success"]]
    print(f"\n✅ Réussies: {len(ok)}/{len(results)}")
    if ko:
        print(f"❌ Échouées: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    # Sauvegarder le résumé
    summary_path = os.path.join(base_dir, f"summary_{method}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résumé: {summary_path}")
    print("✨ Terminé!")

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Markovien multi-partitionnement"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cartesian",
        choices=list(REGISTRY.keys()) + ["all"],  # ← inclut automatiquement adaptive, multizone, single
        help="Type de partitionnement (default: cartesian)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=BASE_OUTPUT_DIR,
        help=f"Dossier de sortie (default: {BASE_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les configurations sans lancer les calculs",
    )
    args = parser.parse_args()

    if args.list:
        if args.method == "all":
            for m in REGISTRY:
                configs = get_configs(m)
                print(f"\n{m.upper()} ({len(configs)} configs):")
                for c in configs:
                    p = create_partitioner(c.method, **c.method_kwargs)
                    print(f"  {p.label} NLT={c.nlt} step={c.step_size} dt={c.dt}")
        else:
            configs = get_configs(args.method)
            print(f"{args.method.upper()} ({len(configs)} configs):")
            for c in configs:
                p = create_partitioner(c.method, **c.method_kwargs)
                print(f"  {p.label} NLT={c.nlt} step={c.step_size} dt={c.dt}")
        return

    run_markov_sweep(args.method, base_dir=args.output)


if __name__ == "__main__":
    main()