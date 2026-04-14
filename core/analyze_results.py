"""
===================================================================================
ANALYSE MARKOVIENNE — Chargement et visualisation depuis le bucket HuggingFace
===================================================================================

Charge automatiquement toutes les expériences (voronoi, cartesian, cylindrical,
quantile, octree, physics) et propose des visualisations comparatives.

Usage:
    python analyze_results.py
    
Depuis un notebook:
    from analyze_results import MarkovAnalyzer
    analyzer = MarkovAnalyzer()
    analyzer.load_all()
    analyzer.compare_methods()
===================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import json
import io
from collections import defaultdict
from huggingface_hub import HfFileSystem
import src.bucket_io as b_io
# =============================================================================
# CONFIGURATION
# =============================================================================

# BUCKET_ID = "ktongue/DEM_MCM"
# BUCKET_PREFIX = "markov_results"
# BUCKET_PREFIX = "ResultsDtMCM"
# BUCKET_PREFIX = "ResultsDtMCM"
BUCKET_ID = b_io.BUCKET_ID
BUCKET_PREFIX = b_io.BUCKET_PREFIX
BUCKET_BASE = b_io.BUCKET_BASE

# Anciennes données cartésiennes (dossier séparé)
# OLD_BUCKET_PREFIX = "markov_sweep_results"
# OLD_BUCKET_PREFIX = "NewResultsMCM"
OLD_BUCKET_PREFIX = BUCKET_PREFIX
OLD_BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{OLD_BUCKET_PREFIX}"


BUCKET_ID=b_io.BUCKET_ID
BUCKET_PREFIX=b_io.BUCKET_PREFIX
BUCKET_BASE=b_io.BUCKET_BASE
# Méthodes connues et leurs préfixes
METHOD_PREFIXES = {
    "cartesian": ["cartesian_", "NLT_"],   # NLT_ = ancien format cartésien
    "cylindrical": ["cylindrical_"],
    "voronoi": ["voronoi_"],
    "quantile": ["quantile_"],
    "octree": ["octree_"],
    "physics": ["physics_"],
    "adaptive":["adaptive_"],
    "mutlizone":["multizone_"],
    "single":["single_"],
}

# Couleurs par méthode
METHOD_COLORS = {
    "cartesian": "#1f77b4",
    "cylindrical": "#ff7f0e",
    "voronoi": "#2ca02c",
    "quantile": "#d62728",
    "octree": "#9467bd",
    "physics": "#8c564b",
    "adaptive":"#af6c3c",
    "multizone":"#4b5d4c",
    "single":"#2b3e4ba8",
    "unknown": "#7f7f7f",
}


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class MarkovAnalyzer:
    """
    Chargeur et analyseur universel de résultats Markoviens.
    
    Gère tous les types de partitionnement et les deux formats
    (ancien cartésien + nouveau multi-méthode).
    """
    
    def __init__(self):
        self.fs = HfFileSystem()
        self.results = {}           # {folder_name: {matrix, params, stats, method}}
        self.by_method = defaultdict(dict)  # {method: {folder_name: data}}
    
    # ─────────────────────────────────────────────────────────────────────
    # DÉTECTION DE MÉTHODE
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_method(self, folder_name, params=None):
        """
        Détecte la méthode de partitionnement depuis le nom du dossier ou les params.
        
        Args:
            folder_name: nom du dossier
            params: dict de paramètres (optionnel)
        Returns:
            str: nom de la méthode
        """
        # Depuis les params/config
        if params:
            if "method" in params:
                return params["method"]
            # Ancien format cartésien (a nx/ny/nz mais pas de "method")
            if "nx" in params and "method" not in params:
                return "cartesian"
        
        # Depuis le nom du dossier
        for method, prefixes in METHOD_PREFIXES.items():
            for prefix in prefixes:
                if folder_name.startswith(prefix):
                    return method
        
        return "unknown"
    
    def _parse_experiment_info(self, folder_name, params, stats):
        """
        Extrait les infos clés d'une expérience de manière uniforme.
        
        Returns:
            dict avec n_states, nlt, step_size, start_index, description
        """
        info = {
            "folder": folder_name,
            "n_states": None,
            "nlt": None,
            "step_size": None,
            "start_index": None,
            "description": "",
        }
        
        # Depuis stats
        if stats:
            info["n_states"] = stats.get("n_states")
            info["nlt"] = stats.get("n_timesteps_used")
        
        # Depuis params/config
        if params:
            # Nouveau format (config.json)
            if "method_kwargs" in params:
                kwargs = params["method_kwargs"]
                info["description"] = str(kwargs)
            
            info["nlt"] = info["nlt"] or params.get("nlt") or params.get("NLT")
            info["step_size"] = params.get("step_size")
            info["start_index"] = params.get("start_index")
            
            # Ancien format cartésien
            if "nx" in params and "method" not in params:
                nx = params.get("nx", "?")
                ny = params.get("ny", "?")
                nz = params.get("nz", "?")
                info["description"] = f"nx={nx}, ny={ny}, nz={nz}"
                if info["n_states"] is None:
                    try:
                        info["n_states"] = int(nx) * int(ny) * int(nz)
                    except:
                        pass
        
        # Fallback depuis la matrice
        return info
    
    # ─────────────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────────────────
    
    def _load_npy(self, full_path):
        """Charge un .npy depuis le bucket."""
        with self.fs.open(full_path, "rb") as f:
            return np.load(io.BytesIO(f.read()))
    
    def _load_json(self, full_path):
        """Charge un .json depuis le bucket."""
        with self.fs.open(full_path, "r") as f:
            return json.load(f)
    
    def _load_partitioner_data(self, partitioner_path):
        """
        Charge les données du partitionnement depuis le bucket.
        
        Args:
            partitioner_path: chemin du dossier /partitioner
        
        Returns:
            dict avec métadonnées et données spécifiques (e.g., r_edges pour cylindrique)
        """
        # Charger les métadonnées
        meta_file = f"{partitioner_path}/partitioner_meta.json"
        meta = self._load_json(meta_file)
        
        partitioner_data = {
            "type": meta.get("type"),
            "label": meta.get("label"),
            "n_cells": meta.get("n_cells"),
        }
        
        # Charger les données spécifiques selon le type
        partitioner_type = meta.get("type")
        
        if partitioner_type == "CylindricalPartitioner":
            try:
                # Charger les paramètres cylindriques
                cyl_params = self._load_json(f"{partitioner_path}/cylindrical_params.json")
                partitioner_data.update(cyl_params)
                
                # Charger les edges des rayons
                r_edges = self._load_npy(f"{partitioner_path}/r_edges.npy")
                partitioner_data["r_edges"] = r_edges
            except Exception as e:
                print(f"⚠️  Impossible de charger les données cylindriques: {e}")
        
        elif partitioner_type == "CartesianPartitioner":
            try:
                cart_params = self._load_json(f"{partitioner_path}/cartesian_params.json")
                partitioner_data.update(cart_params)
            except Exception as e:
                print(f"⚠️  Impossible de charger les données cartésiennes: {e}")
        
        elif partitioner_type == "VoronoiPartitioner":
            try:
                vor_params = self._load_json(f"{partitioner_path}/voronoi_params.json")
                partitioner_data.update(vor_params)
                
                centroids = self._load_npy(f"{partitioner_path}/centroids.npy")
                partitioner_data["centroids"] = centroids
            except Exception as e:
                print(f"⚠️  Impossible de charger les données Voronoï: {e}")
        
        return partitioner_data
    
    def _list_folders(self, base_path=BUCKET_BASE):
        """Liste les sous-dossiers d'un chemin."""
        try:
            items = self.fs.ls(base_path)
            return sorted([
                item["name"].split("/")[-1]
                for item in items
                if item["type"] == "directory"
            ])
        except FileNotFoundError:
            return []
    
    def _load_experiment(self, base_path=BUCKET_BASE , folder_name=None):
        """
        Charge une expérience depuis un dossier du bucket.
        
        Gère les deux formats:
        - Ancien: params.json + stats.json + transition_matrix.npy
        - Nouveau: config.json + stats.json + transition_matrix.npy
        
        Essaie aussi de charger les données du partitionnement si disponibles.
        """
        prefix = f"{base_path}/{folder_name}"
        
        # Matrice (obligatoire)
        matrix = self._load_npy(f"{prefix}/transition_matrix.npy")
        
        # Params (essayer config.json puis params.json)
        params = {}
        for fname in ["config.json", "params.json"]:
            try:
                params = self._load_json(f"{prefix}/{fname}")
                break
            except:
                continue
        
        # Stats
        stats = {}
        try:
            stats = self._load_json(f"{prefix}/stats.json")
        except:
            pass
        
        # Centroïdes (voronoi)
        centroids = None
        try:
            centroids = self._load_npy(f"{prefix}/centroids.npy")
        except:
            pass
        
        # Données de partitionnement
        partitioner_data = None
        try:
            partitioner_data = self._load_partitioner_data(f"{prefix}/partitioner")
        except:
            pass
        
        # Méthode
        method = self._detect_method(folder_name, params)
        
        # Infos
        info = self._parse_experiment_info(folder_name, params, stats)
        if info["n_states"] is None:
            info["n_states"] = matrix.shape[0]
        
        return {
            "matrix": matrix,
            "params": params,
            "stats": stats,
            "method": method,
            "info": info,
            "centroids": centroids,
            "partitioner_data": partitioner_data,
        }
    
    def load_all(self, include_old=True):
        """
        Charge toutes les expériences depuis le bucket.
        
        Args:
            include_old: inclure les anciennes données cartésiennes
        """
        self.results = {}
        self.by_method = defaultdict(dict)
        
        # ── Nouveau format ──
        print(f"📂 Chargement depuis {BUCKET_BASE}...")
        new_folders = self._list_folders(BUCKET_BASE)
        print(f"   {len(new_folders)} dossiers trouvés")
        
        for folder in new_folders:
            try:
                data = self._load_experiment(BUCKET_BASE, folder)
                self.results[folder] = data
                self.by_method[data["method"]][folder] = data
                print(f"   ✅ [{data['method']:12s}] {folder}: "
                      f"shape={data['matrix'].shape}")
            except Exception as e:
                print(f"   ⚠️  {folder}: {e}")
        
        # ── Ancien format cartésien ──
        if include_old:
            print(f"\n📂 Chargement depuis {OLD_BUCKET_BASE}...")
            old_folders = self._list_folders(OLD_BUCKET_BASE)
            print(f"   {len(old_folders)} dossiers trouvés")
            
            for folder in old_folders:
                if folder in self.results:
                    continue  # déjà chargé
                try:
                    data = self._load_experiment(OLD_BUCKET_BASE, folder)
                    self.results[folder] = data
                    self.by_method[data["method"]][folder] = data
                    print(f"   ✅ [{data['method']:12s}] {folder}: "
                          f"shape={data['matrix'].shape}")
                except Exception as e:
                    print(f"   ⚠️  {folder}: {e}")
        
        # Résumé
        print(f"\n{'='*60}")
        print(f"RÉSUMÉ: {len(self.results)} expériences chargées")
        print(f"{'='*60}")
        for method, exps in sorted(self.by_method.items()):
            print(f"   {method:15s}: {len(exps):3d} expériences")
        print()
    
    def load_method(self, method):
        """Charge uniquement les expériences d'une méthode."""
        self.results = {}
        self.by_method = defaultdict(dict)
        
        for base_path in [BUCKET_BASE, OLD_BUCKET_BASE]:
            folders = self._list_folders(base_path)
            for folder in folders:
                detected = self._detect_method(folder)
                if detected == method:
                    try:
                        data = self._load_experiment(base_path, folder)
                        self.results[folder] = data
                        self.by_method[method][folder] = data
                        print(f"   ✅ {folder}: shape={data['matrix'].shape}")
                    except Exception as e:
                        print(f"   ⚠️  {folder}: {e}")
        
        print(f"\n{len(self.results)} expériences {method} chargées")
    
    # ─────────────────────────────────────────────────────────────────────
    # ACCÈS AUX DONNÉES
    # ─────────────────────────────────────────────────────────────────────
    
    def get_methods(self):
        """Retourne la liste des méthodes disponibles."""
        return list(self.by_method.keys())
    
    def get_experiments(self, method=None):
        """
        Retourne les expériences, optionnellement filtrées par méthode.
        
        Args:
            method: str ou None (toutes)
        Returns:
            dict {folder_name: data}
        """
        if method is None:
            return self.results
        return dict(self.by_method.get(method, {}))
    
    def get_matrix(self, folder_name):
        """Accès rapide à une matrice."""
        return self.results[folder_name]["matrix"]
    
    def get_matrices_by_method(self, method):
        """Retourne {folder_name: matrix} pour une méthode."""
        return {
            name: data["matrix"]
            for name, data in self.by_method.get(method, {}).items()
        }
    
    def summary_table(self):
        """Retourne un tableau récapitulatif de toutes les expériences."""
        rows = []
        for name, data in self.results.items():
            M = data["matrix"]
            diag = np.diag(M)
            row_sums = M.sum(axis=0)
            visited = row_sums > 0
            
            rows.append({
                "name": name,
                "method": data["method"],
                "n_states": M.shape[0],
                "n_visited": int(visited.sum()),
                "nlt": data["info"]["nlt"],
                "step": data["info"]["step_size"],
                "start": data["info"]["start_index"],
                "diag_mean": float(diag.mean()),
                "diag_std": float(diag.std()),
                "row_sum_min": float(row_sums[visited].min()) if visited.any() else 0,
                "row_sum_max": float(row_sums[visited].max()) if visited.any() else 0,
            })
        
        rows.sort(key=lambda r: (r["method"], r["n_states"]))
        return rows
    
    def print_summary(self):
        """Affiche le résumé formaté."""
        rows = self.summary_table()
        
        print(f"\n{'Method':>12s} | {'Name':40s} | {'States':>6s} | {'Visit':>5s} | "
              f"{'NLT':>4s} | {'P(stay)':>8s} | {'ΣRow':>12s}")
        print("-" * 110)
        
        current_method = None
        for r in rows:
            if r["method"] != current_method:
                current_method = r["method"]
                print(f"{'─'*12}─┼{'─'*52}┼{'─'*8}┼{'─'*7}┼{'─'*6}┼{'─'*10}┼{'─'*14}")
            
            nlt_str = str(r["nlt"]) if r["nlt"] else "?"
            print(f"{r['method']:>12s} | {r['name'][:50]:50s} | {r['n_states']:6d} | "
                  f"{r['n_visited']:5d} | {nlt_str:>4s} | {r['diag_mean']:8.4f} | "
                  f"[{r['row_sum_min']:.3f}, {r['row_sum_max']:.3f}]")
    
    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ─────────────────────────────────────────────────────────────────────
    
    def simulate_mixing(self, folder_name, n_steps=100, initial_split=0.5):
        """
        Simule le mélange à partir d'une matrice de transition.
        
        Args:
            folder_name: nom de l'expérience
            n_steps: nombre de pas de temps
            initial_split: fraction de la frontière initiale
        
        Returns:
            S_history: array (n_steps, n_states)
        """
        M = self.get_matrix(folder_name)
        n_states = M.shape[0]
        
        # État initial: séparation binaire
        S = np.zeros(n_states)
        mid = int(n_states * initial_split)
        S[:mid] = 1.0
        S[mid:] = 0.0
        S = S / S.sum() if S.sum() > 0 else S
        
        S_history = np.zeros((n_steps, n_states))
        for i in range(n_steps):
            S = S @ M
            S_history[i] = S
        
        return S_history
    
    # ─────────────────────────────────────────────────────────────────────
    # VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────
    
    def plot_matrix(self, folder_name, log_scale=False, figsize=(8, 7)):
        """Affiche la matrice de transition en heatmap."""
        data = self.results[folder_name]
        M = data["matrix"]
        method = data["method"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        kwargs = {"cmap": "viridis", "aspect": "auto"}
        if log_scale:
            kwargs["norm"] = LogNorm(vmin=max(M[M > 0].min(), 1e-6), vmax=M.max())
        
        im = ax.imshow(M, **kwargs)
        ax.set_xlabel("État destination")
        ax.set_ylabel("État source")
        ax.set_title(f"Matrice P — {method}\n{folder_name}")
        plt.colorbar(im, ax=ax, label="Probabilité de transition")
        plt.tight_layout()
        plt.show()
    
    # def plot_experiment(self, folder_name, n_steps=100, figsize=(16, 10)):
    #     """Visualisation complète d'une expérience."""
    #     data = self.results[folder_name]
    #     M = data["matrix"]
    #     method = data["method"]
    #     n_states = M.shape[0]
        
    #     # Simulation
    #     S_history = self.simulate_mixing(folder_name, n_steps)
        
    #     fig = plt.figure(figsize=figsize)
    #     fig.suptitle(f"{method.upper()} — {folder_name}", fontsize=14, fontweight="bold")
    #     gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
    #     # 1. Matrice de transition
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     im = ax1.imshow(M, cmap="viridis", aspect="auto")
    #     ax1.set_xlabel("Dest")
    #     ax1.set_ylabel("Source")
    #     ax1.set_title("Matrice P")
    #     plt.colorbar(im, ax=ax1, fraction=0.046)
        
    #     # 2. Diagonale
    #     ax2 = fig.add_subplot(gs[0, 1])
    #     diag = np.diag(M)
    #     ax2.bar(range(n_states), diag, color=METHOD_COLORS.get(method, "#333"), alpha=0.8)
    #     ax2.axhline(diag.mean(), color="red", ls="--", label=f"μ={diag.mean():.3f}")
    #     ax2.set_xlabel("État")
    #     ax2.set_ylabel("P(rester)")
    #     ax2.set_title("Diagonale de P")
    #     ax2.legend()
        
    #     # 3. Somme des lignes
    #     ax3 = fig.add_subplot(gs[0, 2])
    #     row_sums = M.sum(axis=1)
    #     ax3.bar(range(n_states), row_sums, color="steelblue", alpha=0.8)
    #     ax3.axhline(1.0, color="red", ls="--", alpha=0.5)
    #     ax3.set_xlabel("État")
    #     ax3.set_ylabel("ΣP")
    #     ax3.set_title(f"Somme des lignes\n[{row_sums[row_sums>0].min():.3f}, {row_sums.max():.3f}]")
        
    #     # 4. Évolution temporelle
    #     ax4 = fig.add_subplot(gs[1, 0:2])
    #     step = max(1, n_states // 10)
    #     for j in range(0, n_states, step):
    #         ax4.plot(range(n_steps), S_history[:, j], label=f"État {j}")
    #     ax4.set_xlabel("Pas de temps")
    #     ax4.set_ylabel("Probabilité")
    #     ax4.set_title("Évolution temporelle")
    #     ax4.legend(fontsize=7, ncol=2)
    #     ax4.grid(True, alpha=0.3)
        
    #     # 5. Distribution finale vs initiale
    #     ax5 = fig.add_subplot(gs[1, 2])
    #     mid = n_states // 2
    #     S0 = np.zeros(n_states)
    #     S0[:mid] = 1.0
    #     S0 = S0 / S0.sum()
        
    #     ax5.bar(range(n_states), S0, alpha=0.4, label="Initial", color="blue")
    #     ax5.bar(range(n_states), S_history[-1], alpha=0.4, label=f"t={n_steps}", color="red")
    #     ax5.set_xlabel("État")
    #     ax5.set_ylabel("Probabilité")
    #     ax5.set_title("Initiale vs Finale")
    #     ax5.legend()
        
    #     plt.savefig(f"analysis_{folder_name[:50]}.png", dpi=150, bbox_inches="tight")
    #     plt.show()
    
    def compare_methods(self, metric="diag_mean", figsize=(14, 6)):
        """
        Compare toutes les méthodes sur une métrique.
        
        Args:
            metric: "diag_mean", "n_states", "row_sum_range"
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x_offset = 0
        tick_positions = []
        tick_labels = []
        method_spans = []
        
        for method in sorted(self.by_method.keys()):
            exps = self.by_method[method]
            if not exps:
                continue
            
            start_x = x_offset
            color = METHOD_COLORS.get(method, "#333")
            
            # Trier par nombre d'états
            sorted_exps = sorted(exps.items(), key=lambda x: x[1]["matrix"].shape[0])
            
            for name, data in sorted_exps:
                M = data["matrix"]
                diag = np.diag(M)
                row_sums = M.sum(axis=1)
                visited = row_sums > 0
                
                if metric == "diag_mean":
                    value = diag.mean()
                elif metric == "n_states":
                    value = M.shape[0]
                elif metric == "row_sum_range":
                    value = row_sums[visited].max() - row_sums[visited].min() if visited.any() else 0
                elif metric == "n_visited":
                    value = visited.sum()
                else:
                    value = 0
                
                ax.bar(x_offset, value, color=color, alpha=0.8, width=0.8)
                
                # Label court
                short = name.replace(f"{method}_", "").replace("_NLT", "\nNLT")[:25]
                tick_positions.append(x_offset)
                tick_labels.append(short)
                x_offset += 1
            
            method_spans.append((start_x, x_offset - 1, method))
            x_offset += 1  # espace entre méthodes
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"Comparaison inter-méthodes: {metric}")
        
        # Légende des méthodes
        for start, end, method in method_spans:
            mid = (start + end) / 2
            ax.annotate(method.upper(), xy=(mid, ax.get_ylim()[1]),
                       ha="center", va="bottom", fontsize=10, fontweight="bold",
                       color=METHOD_COLORS.get(method, "#333"))
        
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        plt.savefig(f"compare_methods_{metric}.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    def compare_within_method(self, method, sweep_param="n_states", figsize=(12, 8)):
        """
        Compare les expériences au sein d'une même méthode.
        
        Args:
            method: "voronoi", "cartesian", etc.
            sweep_param: "n_states", "nlt", "step_size"
        """
        exps = self.get_experiments(method)
        if not exps:
            print(f"Aucune expérience pour {method}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"{method.upper()} — Sweep sur {sweep_param}", fontsize=14)
        color = METHOD_COLORS.get(method, "#333")
        
        # Collecter les données
        data_points = []
        for name, data in exps.items():
            M = data["matrix"]
            info = data["info"]
            diag = np.diag(M)
            row_sums = M.sum(axis=1)
            visited = row_sums > 0
            
            if sweep_param == "n_states":
                x_val = M.shape[0]
            elif sweep_param == "nlt":
                x_val = info.get("nlt") or 0
            elif sweep_param == "step_size":
                x_val = info.get("step_size") or 0
            elif sweep_param == "start_index":
                x_val = info.get("start_index") or 0
            else:
                x_val = M.shape[0]
            
            data_points.append({
                "x": x_val,
                "name": name,
                "diag_mean": diag.mean(),
                "diag_std": diag.std(),
                "n_visited": int(visited.sum()),
                "n_states": M.shape[0],
                "row_sum_min": float(row_sums[visited].min()) if visited.any() else 0,
                "row_sum_max": float(row_sums[visited].max()) if visited.any() else 0,
            })
        
        data_points.sort(key=lambda d: d["x"])
        xs = [d["x"] for d in data_points]
        
        # 1. Diagonale moyenne
        ax = axes[0, 0]
        ax.plot(xs, [d["diag_mean"] for d in data_points], "o-", color=color)
        ax.fill_between(
            xs,
            [d["diag_mean"] - d["diag_std"] for d in data_points],
            [d["diag_mean"] + d["diag_std"] for d in data_points],
            alpha=0.2, color=color,
        )
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("P(rester)")
        ax.set_title("Diagonale moyenne ± σ")
        ax.grid(True, alpha=0.3)
        
        # 2. Fraction visitée
        ax = axes[0, 1]
        fracs = [d["n_visited"] / d["n_states"] for d in data_points]
        ax.plot(xs, fracs, "o-", color=color)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("Fraction visitée")
        ax.set_title("États visités / total")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # 3. Somme des lignes (min/max)
        ax = axes[1, 0]
        ax.fill_between(
            xs,
            [d["row_sum_min"] for d in data_points],
            [d["row_sum_max"] for d in data_points],
            alpha=0.3, color=color, label="[min, max]",
        )
        ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="Idéal = 1")
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("Σ lignes")
        ax.set_title("Somme des lignes [min, max]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Nombre d'états visités
        ax = axes[1, 1]
        ax.plot(xs, [d["n_visited"] for d in data_points], "s-", color=color, label="Visités")
        ax.plot(xs, [d["n_states"] for d in data_points], "x--", color="gray", label="Total")
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("Nombre d'états")
        ax.set_title("États visités vs total")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"sweep_{method}_{sweep_param}.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    
    # Ajoutez ces méthodes à la classe MarkovAnalyzer dans analyze_results.py

    def compute_rsd(self, folder_name, n_steps=200, initial_split=0.5):
        """
        Calcule le RSD (Relative Standard Deviation) des particules
        dans chaque partition au cours du temps.

        Le RSD mesure l'homogénéité du mélange:
            RSD = 0%   → mélange parfait (distribution uniforme)
            RSD = 100% → ségrégation totale

        Formule: RSD(t) = σ(C_i(t)) / μ(C_i(t))
        où C_i(t) est la concentration (fraction de particules) dans la cellule i.

        Args:
            folder_name: nom de l'expérience
            n_steps: nombre de pas de simulation
            initial_split: fraction de la frontière initiale (0.5 = moitié/moitié)

        Returns:
            dict avec:
                - rsd: array (n_steps,) — RSD à chaque pas
                - rsd_percent: array (n_steps,) — RSD en pourcentage
                - concentration_history: array (n_steps, n_states) — C_i(t)
                - entropy: array (n_steps,) — entropie normalisée
                - rsd_initial: float — RSD initial
                - rsd_final: float — RSD final
                - mixing_time_50: int ou None — pas où RSD < 50% du RSD initial
                - mixing_time_90: int ou None — pas où RSD < 10% du RSD initial
        """
        M = self.get_matrix(folder_name)
        n_states = M.shape[0]

        # ── État initial ségrégé ──
        # Concentration initiale: espèce A dans la moitié gauche,
        # espèce B dans la moitié droite
        C = np.zeros(n_states)
        mid = int(n_states * initial_split)
        C[:mid] = 1.0    # 100% d'espèce A dans les cellules 0..mid
        C[mid:] = 0.0    # 0% d'espèce A dans les cellules mid..n

        # ── Simulation ──
        concentration_history = np.zeros((n_steps, n_states))
        rsd = np.zeros(n_steps)
        entropy = np.zeros(n_steps)

        for t in range(n_steps):
            C = C @ M

            # Stocker
            concentration_history[t] = C

            # RSD: σ/μ sur les cellules visitées (P > 0)
            visited = C > 1e-12
            if visited.sum() > 1:
                mean_c = C[visited].mean()
                std_c = C[visited].std()
                rsd[t] = std_c / mean_c if mean_c > 0 else 0
            else:
                rsd[t] = 0

            # Entropie normalisée
            C_pos = C[C > 1e-12]
            if len(C_pos) > 0 and n_states > 1:
                entropy[t] = -np.sum(C_pos * np.log(C_pos)) / np.log(n_states)
            else:
                entropy[t] = 0

        # ── Temps de mélange ──
        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0

        mixing_time_50 = None
        mixing_time_90 = None
        for t in range(n_steps):
            if mixing_time_50 is None and rsd[t] < 0.5 * rsd_0:
                mixing_time_50 = t
            if mixing_time_90 is None and rsd[t] < 0.1 * rsd_0:
                mixing_time_90 = t

        return {
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentration_history": concentration_history,
            "entropy": entropy,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
        }

    """
Ajoutez ces méthodes à la classe MarkovAnalyzer dans analyze_results.py
"""

# ═══════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES DEM
# ═══════════════════════════════════════════════════════════════════

    def load_dem_snapshots(self, file_indices=None, sample_every=1):
        """
        Charge les positions des particules DEM à plusieurs instants.

        Les particules conservent leur index (ligne) entre les fichiers:
        la particule i dans le fichier t est la même particule physique
        que la particule i dans le fichier t+1.

        Args:
            file_indices: liste d'indices de fichiers (None = auto)
            sample_every: sous-échantillonnage spatial (1 = toutes)

        Returns:
            list de dict {t, coords} stocké dans self.dem_snapshots
        """
        import polars as pl

        if not hasattr(self, '_dem_fs'):
            self._dem_fs = HfFileSystem()
            self._dem_files = sorted(
                self._dem_fs.glob("hf://buckets/ktongue/DEM_MCM/Output Paraview/*.csv")
            )
            print(f"📁 {len(self._dem_files)} fichiers DEM disponibles")

        if file_indices is None:
            file_indices = list(range(0, min(len(self._dem_files), 500), 10))

        self.dem_snapshots = []
        self.dem_file_indices = file_indices

        print(f"📂 Chargement de {len(file_indices)} snapshots DEM...")

        for i, idx in enumerate(file_indices):
            with self._dem_fs.open(self._dem_files[idx], "rb") as f:
                df = pl.read_csv(f)

            coords = np.column_stack([
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ])[::sample_every]

            self.dem_snapshots.append({"t": idx, "coords": coords})

            if (i + 1) % 10 == 0 or i == len(file_indices) - 1:
                print(f"   [{i+1}/{len(file_indices)}] t={idx}: {len(coords)} particules")

        self.n_particles = len(self.dem_snapshots[0]["coords"])
        print(f"✅ {len(self.dem_snapshots)} snapshots | {self.n_particles} particules/snapshot")

        return self.dem_snapshots


    def label_species(self, criterion="z_median", custom_labels=None):
        """
        Étiquette chaque particule comme espèce A (1) ou B (0) à t=0.

        L'étiquette est PERMANENTE: une particule garde son espèce
        quel que soit son déplacement.

        Args:
            criterion: méthode d'étiquetage:
                - "z_median": z > médiane(z) → espèce A
                - "z_half":   z > (zmin+zmax)/2 → espèce A
                - "x_median": x > médiane(x) → espèce A
                - "y_median": y > médiane(y) → espèce A
                - "r_median": r > médiane(r) → espèce A (radial)
                - "random":   50/50 aléatoire
                - "quadrant": haut-gauche vs bas-droite
            custom_labels: array bool de taille n_particles (override)

        Returns:
            np.array bool de taille n_particles (True = espèce A)
        """
        if custom_labels is not None:
            self.species_labels = np.asarray(custom_labels, dtype=bool)
            n_a = self.species_labels.sum()
            print(f"🏷️  Labels custom: {n_a} A / {len(self.species_labels) - n_a} B")
            return self.species_labels

        coords_t0 = self.dem_snapshots[0]["coords"]
        x, y, z = coords_t0[:, 0], coords_t0[:, 1], coords_t0[:, 2]

        if criterion == "z_median":
            labels = z > np.median(z)
        elif criterion == "z_half":
            labels = z > (z.min() + z.max()) / 2
        elif criterion == "x_median":
            labels = x > np.median(x)
        elif criterion == "y_median":
            labels = y > np.median(y)
        elif criterion == "r_median":
            xc, yc = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
            r = np.sqrt((x - xc)**2 + (y - yc)**2)
            labels = r > np.median(r)
        elif criterion == "random":
            rng = np.random.RandomState(42)
            labels = rng.rand(len(x)) > 0.5
        elif criterion == "quadrant":
            labels = (z > np.median(z)) & (x > np.median(x))
        else:
            raise ValueError(f"Critère inconnu: {criterion}")

        self.species_labels = labels
        n_a = labels.sum()
        print(f"🏷️  Espèces ({criterion}): {n_a} A / {len(labels) - n_a} B")

        return self.species_labels


    def create_partitioner_for_comparison(self, method, method_kwargs):
        """
        Crée et fit un partitionneur sur les données DEM.

        Args:
            method: "cartesian", "voronoi", "cylindrical", "quantile"
            method_kwargs: dict de paramètres

        Returns:
            partitioner fitté
        """
        from src.partitioners import create_partitioner

        # Agréger les données pour le fit
        all_coords = np.vstack([s["coords"] for s in self.dem_snapshots])

        part = create_partitioner(method, **method_kwargs)
        part.fit(all_coords)

        diag = part.diagnostics(all_coords)
        print(f"🔧 {part.label}: {part.n_cells} cellules | "
            f"{diag['n_visited']} visitées | "
            f"pop μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}")

        return part


    # ═══════════════════════════════════════════════════════════════════
    # CALCUL DU RSD — DONNÉES DEM RÉELLES
    # ═══════════════════════════════════════════════════════════════════

    def compute_dem_rsd(self, partitioner, species_labels=None):
        """
        Calcule le RSD à partir des données DEM réelles.

        À chaque instant t:
        1. Assigner chaque particule à sa cellule
        2. Pour chaque cellule i:
            C_i(t) = n_A(i,t) / n_total(i,t)
            (concentration de l'espèce A dans la cellule i)
        3. RSD(t) = std(C_i) / mean(C_i)  sur les cellules non-vides

        Args:
            partitioner: partitionneur fitté
            species_labels: array bool (None = self.species_labels)

        Returns:
            dict avec:
                - times: array des temps
                - rsd: array des RSD
                - rsd_percent: idem en %
                - concentrations: list de arrays C_i(t)
                - n_particles_per_cell: list de arrays
                - entropy: entropie normalisée
                - intensity_of_segregation: I(t) = σ²(C) / (C̄(1-C̄))
        """
        if species_labels is None:
            species_labels = self.species_labels

        n_states = partitioner.n_cells
        n_snaps = len(self.dem_snapshots)

        times = np.zeros(n_snaps)
        rsd = np.zeros(n_snaps)
        entropy = np.zeros(n_snaps)
        intensity_seg = np.zeros(n_snaps)
        concentrations = []
        populations = []

        for k, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            times[k] = snap["t"]

            # Assigner les particules aux cellules
            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )

            # Compter par cellule: total et espèce A
            n_total = np.bincount(states, minlength=n_states).astype(float)
            n_A = np.bincount(states[species_labels], minlength=n_states).astype(float)

            # Concentration C_i = n_A / n_total
            C = np.zeros(n_states)
            mask = n_total > 0
            C[mask] = n_A[mask] / n_total[mask]

            concentrations.append(C.copy())
            populations.append(n_total.copy())

            # RSD sur cellules non-vides
            C_active = C[mask]
            if len(C_active) > 1 and C_active.mean() > 0:
                rsd[k] = C_active.std() / C_active.mean()
            else:
                rsd[k] = 0

            # Entropie de mélange normalisée
            # H = -Σ [C_i ln(C_i) + (1-C_i) ln(1-C_i)] / N_cells
            C_clip = np.clip(C_active, 1e-10, 1 - 1e-10)
            H = -np.mean(C_clip * np.log(C_clip) + (1 - C_clip) * np.log(1 - C_clip))
            H_max = np.log(2)  # entropie max pour distribution binaire
            entropy[k] = H / H_max if H_max > 0 else 0

            # Intensité de ségrégation: I = σ²(C) / (C̄(1-C̄))
            C_bar = C_active.mean()
            if C_bar > 0 and C_bar < 1:
                intensity_seg[k] = C_active.var() / (C_bar * (1 - C_bar))
            else:
                intensity_seg[k] = 0

        # Temps de mélange
        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
        mixing_time_50 = None
        mixing_time_90 = None
        for k in range(n_snaps):
            if mixing_time_50 is None and rsd[k] < 0.5 * rsd_0:
                mixing_time_50 = int(times[k])
            if mixing_time_90 is None and rsd[k] < 0.1 * rsd_0:
                mixing_time_90 = int(times[k])

        return {
            "times": times,
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentrations": concentrations,
            "populations": populations,
            "entropy": entropy,
            "intensity_of_segregation": intensity_seg,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
            "source": "DEM",
        }


# ═══════════════════════════════════════════════════════════════════
# CALCUL DU RSD — PRÉDICTION MARKOV
# ═══════════════════════════════════════════════════════════════════

    def compute_markov_rsd_from_dem(self, P, partitioner, species_labels=None):
        """
        Calcule le RSD prédit par la chaîne de Markov à partir
        de la condition initiale DEM réelle.

        Principe:
        1. À t=0: compter φ_A(i,0) et φ_total(i,0) depuis le DEM
        2. Prédire: φ_A(i,t+1) = φ_A(t) @ P
                    φ_total(i,t+1) = φ_total(t) @ P
        3. C_i(t) = φ_A(i,t) / φ_total(i,t)
        4. RSD(t) = std(C) / mean(C)

        Args:
            P: matrice de transition
            partitioner: partitionneur fitté
            species_labels: array bool

        Returns:
            dict similaire à compute_dem_rsd
        """
        if species_labels is None:
            species_labels = self.species_labels

        n_states = partitioner.n_cells
        n_snaps = len(self.dem_snapshots)

        # ── Condition initiale depuis les données DEM ──
        coords_t0 = self.dem_snapshots[0]["coords"]
        states_t0 = partitioner.compute_states(
            coords_t0[:, 0], coords_t0[:, 1], coords_t0[:, 2]
        )

        # Distribution initiale des particules A et totales par cellule
        phi_total = np.bincount(states_t0, minlength=n_states).astype(float)
        phi_A = np.bincount(states_t0[species_labels], minlength=n_states).astype(float)

        # ── Prédiction Markov ──
        times = np.zeros(n_snaps)
        rsd = np.zeros(n_snaps)
        entropy = np.zeros(n_snaps)
        intensity_seg = np.zeros(n_snaps)
        concentrations = []

        # État courant
        current_phi_A = phi_A.copy()
        current_phi_total = phi_total.copy()

        for k in range(n_snaps):
            times[k] = self.dem_snapshots[k]["t"]

            if k > 0:
                # Nombre de pas Markov entre deux snapshots
                dt = self.dem_snapshots[k]["t"] - self.dem_snapshots[k - 1]["t"]
                for _ in range(int(dt)):
                    current_phi_A = current_phi_A @ P
                    current_phi_total = current_phi_total @ P

            # Concentration prédite
            C = np.zeros(n_states)
            mask = current_phi_total > 1e-10
            C[mask] = current_phi_A[mask] / current_phi_total[mask]
            C = np.clip(C, 0, 1)

            concentrations.append(C.copy())

            # RSD
            C_active = C[mask]
            if len(C_active) > 1 and C_active.mean() > 0:
                rsd[k] = C_active.std() / C_active.mean()
            else:
                rsd[k] = 0

            # Entropie
            C_clip = np.clip(C_active, 1e-10, 1 - 1e-10)
            H = -np.mean(C_clip * np.log(C_clip) + (1 - C_clip) * np.log(1 - C_clip))
            H_max = np.log(2)
            entropy[k] = H / H_max if H_max > 0 else 0

            # Intensité de ségrégation
            C_bar = C_active.mean()
            if 0 < C_bar < 1:
                intensity_seg[k] = C_active.var() / (C_bar * (1 - C_bar))
            else:
                intensity_seg[k] = 0

        # Temps de mélange
        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
        mixing_time_50 = None
        mixing_time_90 = None
        for k in range(n_snaps):
            if mixing_time_50 is None and rsd[k] < 0.5 * rsd_0:
                mixing_time_50 = int(times[k])
            if mixing_time_90 is None and rsd[k] < 0.1 * rsd_0:
                mixing_time_90 = int(times[k])

        return {
            "times": times,
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentrations": concentrations,
            "entropy": entropy,
            "intensity_of_segregation": intensity_seg,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
            "source": "Markov",
        }


# ═══════════════════════════════════════════════════════════════════
# COMPARAISON DEM vs MARKOV
# ═══════════════════════════════════════════════════════════════════

    def compare_dem_vs_markov(self, method, method_kwargs,
                            folder_name=None,
                            species_criterion="z_median",
                            file_indices=None,
                            figsize=(20, 16)):
        """
        Comparaison complète DEM vs Markov pour un partitionnement donné.

        1. Charge les snapshots DEM (si pas déjà fait)
        2. Crée le partitionneur et calcule le RSD DEM
        3. Charge (ou calcule) la matrice P
        4. Calcule le RSD Markov depuis la même condition initiale
        5. Affiche la comparaison

        Args:
            method: "cartesian", "voronoi", "cylindrical", "quantile"
            method_kwargs: paramètres du partitionneur
            folder_name: nom de l'expérience dans le bucket (None = recalculer P)
            species_criterion: critère d'étiquetage des espèces
            file_indices: indices des fichiers DEM à charger
            figsize: taille de la figure

        Returns:
            dict avec dem_rsd, markov_rsd
        """
        from src.partitioners import create_partitioner

        # ── 1. Charger les snapshots DEM ──
        if not hasattr(self, 'dem_snapshots') or not self.dem_snapshots:
            if file_indices is None:
                file_indices = list(range(0, 500, 5))
            self.load_dem_snapshots(file_indices)

        # ── 2. Étiqueter les espèces ──
        self.label_species(species_criterion)

        # ── 3. Créer le partitionneur ──
        partitioner = self.create_partitioner_for_comparison(method, method_kwargs)

        # ── 4. RSD DEM ──
        print("\n📊 Calcul RSD DEM...")
        dem_rsd = self.compute_dem_rsd(partitioner)
        print(f"   RSD DEM: {dem_rsd['rsd_initial']*100:.1f}% → {dem_rsd['rsd_final']*100:.1f}%")

        # ── 5. Matrice P ──
        if folder_name and folder_name in self.results:
            P = self.results[folder_name]["matrix"]
            print(f"   Matrice P chargée: {folder_name}")
        else:
            print("   Calcul de la matrice P depuis les données DEM...")
            P = self._compute_P_from_dem(partitioner)

        # ── 6. RSD Markov ──
        print("📊 Calcul RSD Markov...")
        markov_rsd = self.compute_markov_rsd_from_dem(P, partitioner)
        print(f"   RSD Markov: {markov_rsd['rsd_initial']*100:.1f}% → {markov_rsd['rsd_final']*100:.1f}%")

        # ── 7. Visualisation ──
        self._plot_dem_vs_markov_comparison(
            dem_rsd, markov_rsd, partitioner, method, figsize
        )

        return {"dem": dem_rsd, "markov": markov_rsd, "partitioner": partitioner, "P": P}


    def _compute_P_from_dem(self, partitioner):
        """
        Calcule la matrice P directement depuis les snapshots DEM chargés.
        """
        n_states = partitioner.n_cells
        T = np.zeros((n_states, n_states))

        for k in range(len(self.dem_snapshots) - 1):
            coords_prev = self.dem_snapshots[k]["coords"]
            coords_curr = self.dem_snapshots[k + 1]["coords"]

            states_prev = partitioner.compute_states(
                coords_prev[:, 0], coords_prev[:, 1], coords_prev[:, 2]
            )
            states_curr = partitioner.compute_states(
                coords_curr[:, 0], coords_curr[:, 1], coords_curr[:, 2]
            )

            n = min(len(states_prev), len(states_curr))
            for i in range(n):
                T[states_prev[i], states_curr[i]] += 1

        # Normaliser
        row_sums = T.sum(axis=1, keepdims=True)
        P = np.divide(T, row_sums, where=row_sums > 0, out=np.zeros_like(T))

        print(f"   P calculée: {n_states}×{n_states}, diag_mean={np.diag(P).mean():.3f}")
        return P


    def _plot_dem_vs_markov_comparison(self, dem_rsd, markov_rsd,
                                        partitioner, method, figsize=(20, 16)):
        """Affiche la comparaison complète DEM vs Markov."""
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"COMPARAISON DEM vs MARKOV — {method.upper()}\n"
            f"{partitioner.label} | {partitioner.n_cells} cellules",
            fontsize=15, fontweight="bold",
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

        times_dem = dem_rsd["times"]
        times_mkv = markov_rsd["times"]

        # ── 1. RSD: DEM vs Markov ──
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(times_dem, dem_rsd["rsd_percent"], "o-", color="#1f77b4",
                lw=2, markersize=4, label="DEM (réel)")
        ax.plot(times_mkv, markov_rsd["rsd_percent"], "s--", color="#ff7f0e",
                lw=2, markersize=4, label="Markov (prédit)")
        ax.set_xlabel("Temps (index fichier)")
        ax.set_ylabel("RSD (%)")
        ax.set_title("RSD: DEM vs Markov")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # ── 2. RSD en échelle log ──
        ax = fig.add_subplot(gs[0, 1])
        rsd_dem_pos = np.clip(dem_rsd["rsd_percent"], 1e-3, None)
        rsd_mkv_pos = np.clip(markov_rsd["rsd_percent"], 1e-3, None)
        ax.semilogy(times_dem, rsd_dem_pos, "o-", color="#1f77b4",
                    lw=2, markersize=4, label="DEM")
        ax.semilogy(times_mkv, rsd_mkv_pos, "s--", color="#ff7f0e",
                    lw=2, markersize=4, label="Markov")
        ax.set_xlabel("Temps")
        ax.set_ylabel("RSD (%) — log")
        ax.set_title("RSD (échelle logarithmique)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 3. Intensité de ségrégation ──
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(times_dem, dem_rsd["intensity_of_segregation"], "o-",
                color="#1f77b4", lw=2, markersize=4, label="DEM")
        ax.plot(times_mkv, markov_rsd["intensity_of_segregation"], "s--",
                color="#ff7f0e", lw=2, markersize=4, label="Markov")
        ax.set_xlabel("Temps")
        ax.set_ylabel("I(t)")
        ax.set_title("Intensité de ségrégation I(t) = σ²(C) / C̄(1-C̄)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # ── 4. Entropie ──
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(times_dem, dem_rsd["entropy"], "o-", color="#1f77b4",
                lw=2, markersize=4, label="DEM")
        ax.plot(times_mkv, markov_rsd["entropy"], "s--", color="#ff7f0e",
                lw=2, markersize=4, label="Markov")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Mélange parfait")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Entropie normalisée")
        ax.set_title("Entropie de mélange")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 5. Erreur relative ──
        ax = fig.add_subplot(gs[2, 0])
        rsd_dem = dem_rsd["rsd"]
        rsd_mkv = markov_rsd["rsd"]
        n = min(len(rsd_dem), len(rsd_mkv))

        abs_error = np.abs(rsd_dem[:n] - rsd_mkv[:n]) * 100
        rel_error = np.zeros(n)
        for k in range(n):
            if rsd_dem[k] > 1e-6:
                rel_error[k] = abs(rsd_dem[k] - rsd_mkv[k]) / rsd_dem[k] * 100

        ax.bar(times_dem[:n], abs_error, width=times_dem[1] - times_dem[0] if n > 1 else 1,
            color="#d62728", alpha=0.7, label="Erreur absolue (%)")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Erreur RSD (%)")
        ax.set_title(f"Erreur |RSD_DEM - RSD_Markov| — "
                    f"moyenne={abs_error.mean():.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 6. Tableau récapitulatif ──
        ax = fig.add_subplot(gs[2, 1])
        ax.axis("off")

        # Corrélation
        corr = np.corrcoef(rsd_dem[:n], rsd_mkv[:n])[0, 1] if n > 2 else 0
        rmse = np.sqrt(np.mean((rsd_dem[:n] - rsd_mkv[:n])**2)) * 100

        table_data = [
            ["", "DEM", "Markov"],
            ["RSD initial (%)", f"{dem_rsd['rsd_initial']*100:.1f}", f"{markov_rsd['rsd_initial']*100:.1f}"],
            ["RSD final (%)", f"{dem_rsd['rsd_final']*100:.1f}", f"{markov_rsd['rsd_final']*100:.1f}"],
            ["t₅₀", f"{dem_rsd['mixing_time_50'] or 'N/A'}", f"{markov_rsd['mixing_time_50'] or 'N/A'}"],
            ["t₉₀", f"{dem_rsd['mixing_time_90'] or 'N/A'}", f"{markov_rsd['mixing_time_90'] or 'N/A'}"],
            ["", "", ""],
            ["Corrélation", f"{corr:.4f}", ""],
            ["RMSE (%)", f"{rmse:.2f}", ""],
            ["Erreur moy (%)", f"{abs_error.mean():.2f}", ""],
            ["Erreur max (%)", f"{abs_error.max():.2f}", ""],
        ]

        table = ax.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.4, 0.3, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style
        for j in range(3):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(table_data)):
            for j in range(3):
                if i == 5:
                    table[i, j].set_height(0.02)
                if i >= 6:
                    table[i, j].set_facecolor("#E2EFDA")

        ax.set_title("Résumé", fontsize=12, fontweight="bold", pad=20)

        plt.savefig(f"dem_vs_markov_{method}.png", dpi=200, bbox_inches="tight")
        plt.show()


    def compare_all_methods_dem_vs_markov(self, species_criterion="z_median",
                                        file_indices=None, figsize=(16, 10)):
        """
        Compare DEM vs Markov pour TOUTES les méthodes sur un seul graphique.

        Args:
            species_criterion: critère de labeling
            file_indices: indices des fichiers DEM
        """
        from src.partitioners import create_partitioner

        # Charger les données
        if not hasattr(self, 'dem_snapshots') or not self.dem_snapshots:
            if file_indices is None:
                file_indices = list(range(0, 500, 5))
            self.load_dem_snapshots(file_indices)

        self.label_species(species_criterion)

        # Configurations à tester
        configs = {
            "Cartésien (5³)": {"method": "cartesian", "kwargs": {"nx": 15, "ny": 15, "nz": 15}},
            "Cylindrique": {"method": "cylindrical", "kwargs": {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_area"}},
            "Voronoï (125)": {"method": "voronoi", "kwargs": {"n_cells": 125}},
            "Quantile (5³)": {"method": "quantile", "kwargs": {"nx": 5, "ny": 5, "nz": 5}},
        }

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"DEM vs MARKOV — Toutes les méthodes\n"
            f"(espèces: {species_criterion} | {len(self.dem_snapshots)} snapshots)",
            fontsize=14, fontweight="bold",
        )

        all_results = {}

        colors_dem = "#1f77b4"
        colors_mkv = "#ff7f0e"

        for idx, (name, config) in enumerate(configs.items()):
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            print(f"\n{'─'*50}")
            print(f"📐 {name}")

            # Créer partitionneur
            part = self.create_partitioner_for_comparison(config["method"], config["kwargs"])

            # RSD DEM
            dem_rsd = self.compute_dem_rsd(part)

            # Matrice P
            P = self._compute_P_from_dem(part)

            # RSD Markov
            mkv_rsd = self.compute_markov_rsd_from_dem(P, part)

            all_results[name] = {"dem": dem_rsd, "markov": mkv_rsd}

            # Plot
            t = dem_rsd["times"]
            ax.plot(t, dem_rsd["rsd_percent"], "o-", color=colors_dem,
                    lw=2, markersize=3, label="DEM", alpha=0.8)
            ax.plot(t, mkv_rsd["rsd_percent"], "s--", color=colors_mkv,
                    lw=2, markersize=3, label="Markov", alpha=0.8)

            # Corrélation
            n = min(len(dem_rsd["rsd"]), len(mkv_rsd["rsd"]))
            corr = np.corrcoef(dem_rsd["rsd"][:n], mkv_rsd["rsd"][:n])[0, 1] if n > 2 else 0

            ax.set_title(f"{name}\nCorr={corr:.3f}", fontsize=11)
            ax.set_xlabel("Temps")
            ax.set_ylabel("RSD (%)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig("dem_vs_markov_all_methods.png", dpi=200, bbox_inches="tight")
        plt.show()

        return all_results
        
    def plot_experiment(self, folder_name, n_steps=200, figsize=(20, 16)):
        """
        Visualisation complète d'une expérience incluant le RSD.

        6 subplots:
            1. Matrice de transition P (heatmap)
            2. Diagonale de P
            3. Somme des lignes
            4. Évolution temporelle des concentrations
            5. RSD + Entropie au cours du temps
            6. Distribution initiale vs finale + RSD annoté
        """
        from matplotlib.colors import LogNorm
        import matplotlib.gridspec as gridspec

        data = self.results[folder_name]
        M = data["matrix"]
        method = data["method"]
        n_states = M.shape[0]

        # ── Calcul du RSD ──
        rsd_data = self.compute_rsd(folder_name, n_steps)
        C_history = rsd_data["concentration_history"]
        rsd_vals = rsd_data["rsd_percent"]
        entropy_vals = rsd_data["entropy"]

        # ── Figure ──
        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"{method.upper()} — {folder_name}\n"
            f"{n_states} états | RSD initial={rsd_data['rsd_initial']*100:.1f}% → "
            f"final={rsd_data['rsd_final']*100:.1f}%",
            fontsize=14, fontweight="bold",
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

        # ── 1. Matrice P ──
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(M, cmap="viridis", aspect="auto")
        ax1.set_xlabel("Destination")
        ax1.set_ylabel("Source")
        ax1.set_title("Matrice de transition P")
        plt.colorbar(im, ax=ax1, fraction=0.046, label="Probabilité")

        # ── 2. Diagonale ──
        ax2 = fig.add_subplot(gs[0, 1])
        diag = np.diag(M)
        color = METHOD_COLORS.get(method, "#333")
        ax2.bar(range(n_states), diag, color=color, alpha=0.8, width=1.0)
        ax2.axhline(diag.mean(), color="red", ls="--", lw=2,
                    label=f"μ={diag.mean():.3f}")
        ax2.axhline(diag.mean() + diag.std(), color="red", ls=":", alpha=0.5)
        ax2.axhline(diag.mean() - diag.std(), color="red", ls=":", alpha=0.5)
        ax2.set_xlabel("État")
        ax2.set_ylabel("P(rester)")
        ax2.set_title("Diagonale de P")
        ax2.legend()

        # ── 3. Évolution des concentrations ──
        ax3 = fig.add_subplot(gs[1, 0])
        step = max(1, n_states // 10)
        for j in range(0, n_states, step):
            ax3.plot(range(n_steps), C_history[:, j], label=f"Cellule {j}", alpha=0.8)

        # Ligne de concentration uniforme
        ax3.axhline(1.0 / n_states, color="gray", ls=":", alpha=0.5,
                    label=f"Uniforme={1/n_states:.4f}")
        ax3.set_xlabel("Pas de temps")
        ax3.set_ylabel("Concentration C_i(t)")
        ax3.set_title("Évolution de la concentration par cellule")
        ax3.legend(fontsize=7, ncol=2, loc="upper right")
        ax3.grid(True, alpha=0.3)

        # ── 4. RSD + Entropie ──
        ax4 = fig.add_subplot(gs[1, 1])

        color_rsd = "#d62728"
        color_entropy = "#2ca02c"

        ax4_twin = ax4.twinx()

        # RSD
        ax4.plot(range(n_steps), rsd_vals, color=color_rsd, lw=2.5, label="RSD (%)")
        ax4.fill_between(range(n_steps), rsd_vals, alpha=0.1, color=color_rsd)
        ax4.set_xlabel("Pas de temps")
        ax4.set_ylabel("RSD (%)", color=color_rsd)
        ax4.tick_params(axis="y", labelcolor=color_rsd)

        # Entropie
        ax4_twin.plot(range(n_steps), entropy_vals, color=color_entropy, lw=2.5,
                    ls="--", label="Entropie norm.")
        ax4_twin.set_ylabel("Entropie normalisée", color=color_entropy)
        ax4_twin.tick_params(axis="y", labelcolor=color_entropy)
        ax4_twin.set_ylim(0, 1.05)

        # Temps de mélange
        if rsd_data["mixing_time_50"] is not None:
            t50 = rsd_data["mixing_time_50"]
            ax4.axvline(t50, color="orange", ls="--", alpha=0.7,
                        label=f"t₅₀={t50} (RSD÷2)")
        if rsd_data["mixing_time_90"] is not None:
            t90 = rsd_data["mixing_time_90"]
            ax4.axvline(t90, color="purple", ls="--", alpha=0.7,
                        label=f"t₉₀={t90} (RSD÷10)")

        ax4.set_title("RSD et Entropie au cours du mélange")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, alpha=0.3)

        # ── 5. Distribution initiale vs finale ──
        ax5 = fig.add_subplot(gs[2, 0])

        C_initial = np.zeros(n_states)
        mid = n_states // 2
        C_initial[:mid] = 1.0

        ax5.bar(range(n_states), C_initial, alpha=0.4, label="Initial (ségrégé)",
                color="blue", width=1.0)
        ax5.bar(range(n_states), C_history[-1], alpha=0.4,
                label=f"Final (t={n_steps})", color="red", width=1.0)
        ax5.axhline(1.0 / n_states, color="gray", ls=":", alpha=0.7,
                    label=f"Uniforme={1/n_states:.4f}")
        ax5.set_xlabel("Cellule")
        ax5.set_ylabel("Concentration")
        ax5.set_title(f"Distribution: initiale → finale | "
                    f"RSD={rsd_data['rsd_final']*100:.1f}%")
        ax5.legend(fontsize=8)

        # ── 6. Somme des lignes + annotation RSD ──
        ax6 = fig.add_subplot(gs[2, 1])
        row_sums = M.sum(axis=1)
        visited = row_sums > 0
        ax6.bar(range(n_states), row_sums, color="steelblue", alpha=0.8, width=1.0)
        ax6.axhline(1.0, color="red", ls="--", alpha=0.5, label="Idéal = 1")
        ax6.set_xlabel("État")
        ax6.set_ylabel("Σ P(i→j)")
        ax6.set_title(f"Somme des lignes\n"
                    f"[{row_sums[visited].min():.3f}, {row_sums[visited].max():.3f}]")
        ax6.legend()

        # Annotation avec les métriques RSD
        textstr = (
            f"━━━ Métriques de mélange ━━━\n"
            f"RSD initial:  {rsd_data['rsd_initial']*100:.1f}%\n"
            f"RSD final:    {rsd_data['rsd_final']*100:.1f}%\n"
            f"Réduction:    {(1 - rsd_data['rsd_final']/max(rsd_data['rsd_initial'], 1e-10))*100:.1f}%\n"
            f"t₅₀ (RSD÷2): {rsd_data['mixing_time_50'] or 'N/A'}\n"
            f"t₉₀ (RSD÷10):{rsd_data['mixing_time_90'] or 'N/A'}\n"
            f"Entropie fin: {entropy_vals[-1]:.4f}"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                    edgecolor="gray", alpha=0.9)
        ax6.text(0.95, 0.95, textstr, transform=ax6.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=props, family="monospace")

        plt.savefig(f"experiment_{folder_name[:50]}.png", dpi=200, bbox_inches="tight")
        plt.show()


    def plot_rsd_comparison(self, folder_names=None, n_steps=200, figsize=(14, 10)):
        """
        Compare le RSD entre plusieurs expériences.

        Args:
            folder_names: liste de noms (None = une par méthode)
            n_steps: nombre de pas de simulation
        """
        if folder_names is None:
            folder_names = []
            for method in sorted(self.by_method.keys()):
                exps = sorted(
                    self.by_method[method].items(),
                    key=lambda x: x[1]["matrix"].shape[0],
                )
                if exps:
                    folder_names.append(exps[len(exps) // 2][0])

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Comparaison du RSD entre méthodes", fontsize=14, fontweight="bold")

        all_rsd_data = {}

        for name in folder_names:
            if name not in self.results:
                print(f"⚠️ {name} non trouvé")
                continue

            method = self.results[name]["method"]
            color = METHOD_COLORS.get(method, "#333")
            n_states = self.results[name]["matrix"].shape[0]

            rsd_data = self.compute_rsd(name, n_steps)
            all_rsd_data[name] = rsd_data
            label = f"{method} ({n_states})"

            # 1. RSD vs temps
            axes[0, 0].plot(range(n_steps), rsd_data["rsd_percent"],
                            # color=color,
                            lw=2, label=label)

            # 2. Entropie vs temps
            axes[0, 1].plot(range(n_steps), rsd_data["entropy"],
                            # color=color,
                             lw=2, label=label)

            # 3. RSD en log
            rsd_pos = rsd_data["rsd_percent"].copy()
            rsd_pos[rsd_pos < 1e-6] = 1e-6
            axes[1, 0].semilogy(range(n_steps), rsd_pos,
                                # color=color,
                                lw=2, label=label)

        # 1. RSD
        ax = axes[0, 0]
        ax.set_xlabel("Pas de temps")
        ax.set_ylabel("RSD (%)")
        ax.set_title("Décroissance du RSD")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", ls=":", alpha=0.3)

        # 2. Entropie
        ax = axes[0, 1]
        ax.set_xlabel("Pas de temps")
        ax.set_ylabel("Entropie normalisée")
        ax.set_title("Convergence entropique")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Mélange parfait")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. RSD log
        ax = axes[1, 0]
        ax.set_xlabel("Pas de temps")
        ax.set_ylabel("RSD (%) — échelle log")
        ax.set_title("RSD (échelle logarithmique)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Tableau récapitulatif
        ax = axes[1, 1]
        ax.axis("off")

        table_data = []
        headers = ["Méthode", "N états", "RSD₀ %", "RSD_f %", "t₅₀", "t₉₀", "Entropie_f"]

        for name in folder_names:
            if name not in all_rsd_data:
                continue
            rd = all_rsd_data[name]
            method = self.results[name]["method"]
            n_st = rd["n_states"]
            table_data.append([
                f"{method}",
                f"{n_st}",
                f"{rd['rsd_initial']*100:.1f}",
                f"{rd['rsd_final']*100:.1f}",
                f"{rd['mixing_time_50'] or 'N/A'}",
                f"{rd['mixing_time_90'] or 'N/A'}",
                f"{rd['entropy'][-1]:.3f}",
            ])

        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=headers,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            # Coloriser les en-têtes
            for j, header in enumerate(headers):
                table[0, j].set_facecolor("#4472C4")
                table[0, j].set_text_props(color="white", fontweight="bold")

            # Coloriser la meilleure valeur de RSD final
            rsd_finals = [float(row[3]) for row in table_data]
            best_idx = np.argmin(rsd_finals)
            for j in range(len(headers)):
                table[best_idx + 1, j].set_facecolor("#E2EFDA")

        ax.set_title("Résumé", fontsize=12, fontweight="bold", pad=20)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig("rsd_comparison.png", dpi=200, bbox_inches="tight")
        plt.show()

        return all_rsd_data


    def plot_rsd_vs_resolution(self, method, n_steps=200, figsize=(12, 5)):
        """
        RSD final en fonction de la résolution (nombre d'états) pour une méthode.

        Args:
            method: "cartesian", "voronoi", etc.
            n_steps: nombre de pas de simulation
        """
        exps = self.get_experiments(method)
        if not exps:
            print(f"Aucune expérience pour {method}")
            return

        data_points = []
        for name, exp_data in exps.items():
            rsd_data = self.compute_rsd(name, n_steps)
            data_points.append({
                "n_states": rsd_data["n_states"],
                "rsd_final": rsd_data["rsd_final"] * 100,
                "rsd_initial": rsd_data["rsd_initial"] * 100,
                "mixing_time_50": rsd_data["mixing_time_50"],
                "mixing_time_90": rsd_data["mixing_time_90"],
                "entropy_final": rsd_data["entropy"][-1],
                "name": name,
            })

        data_points.sort(key=lambda d: d["n_states"])

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"{method.upper()} — RSD vs Résolution", fontsize=14)
        color = METHOD_COLORS.get(method, "#333")

        xs = [d["n_states"] for d in data_points]

        # 1. RSD final
        ax = axes[0]
        ax.plot(xs, [d["rsd_final"] for d in data_points], "o-", color=color, lw=2)
        ax.set_xlabel("Nombre d'états")
        ax.set_ylabel("RSD final (%)")
        ax.set_title("RSD final")
        ax.grid(True, alpha=0.3)

        # 2. Temps de mélange
        ax = axes[1]
        t50s = [d["mixing_time_50"] if d["mixing_time_50"] else n_steps for d in data_points]
        t90s = [d["mixing_time_90"] if d["mixing_time_90"] else n_steps for d in data_points]
        ax.plot(xs, t50s, "o-", color="orange", lw=2, label="t₅₀")
        ax.plot(xs, t90s, "s-", color="purple", lw=2, label="t₉₀")
        ax.set_xlabel("Nombre d'états")
        ax.set_ylabel("Temps de mélange")
        ax.set_title("Temps de mélange")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Entropie finale
        ax = axes[2]
        ax.plot(xs, [d["entropy_final"] for d in data_points], "o-", color=color, lw=2)
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Nombre d'états")
        ax.set_ylabel("Entropie normalisée")
        ax.set_title("Entropie finale")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"rsd_vs_resolution_{method}.png", dpi=200, bbox_inches="tight")
        plt.show()
    
    
    def plot_mixing_comparison(self, folder_names=None, n_steps=200, figsize=(14, 6)):
        """
        Compare la convergence du mélange entre plusieurs expériences.
        
        Args:
            folder_names: liste de noms (None = une par méthode)
            n_steps: nombre de pas de simulation
        """
        if folder_names is None:
            # Prendre une expérience par méthode (la plus petite)
            folder_names = []
            for method in sorted(self.by_method.keys()):
                exps = sorted(
                    self.by_method[method].items(),
                    key=lambda x: x[1]["matrix"].shape[0],
                )
                if exps:
                    folder_names.append(exps[len(exps) // 2][0])  # taille médiane
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for name in folder_names:
            if name not in self.results:
                print(f"⚠️ {name} non trouvé")
                continue
            
            data = self.results[name]
            method = data["method"]
            color = METHOD_COLORS.get(method, "#333")
            
            S_history = self.simulate_mixing(name, n_steps)
            n_states = S_history.shape[1]
            
            # Entropie normalisée (mesure de mélange)
            entropy = np.zeros(n_steps)
            for t in range(n_steps):
                S = S_history[t]
                S_pos = S[S > 0]
                if len(S_pos) > 0:
                    entropy[t] = -np.sum(S_pos * np.log(S_pos)) / np.log(n_states)
            
            # Variance (mesure de ségrégation)
            variance = S_history.var(axis=1)
            
            label = f"{method} ({n_states} états)"
            axes[0].plot(range(n_steps), entropy, label=label, color=color, linewidth=2)
            axes[1].plot(range(n_steps), variance, label=label, color=color, linewidth=2)
        
        axes[0].set_xlabel("Pas de temps")
        axes[0].set_ylabel("Entropie normalisée")
        axes[0].set_title("Convergence du mélange (entropie)")
        axes[0].axhline(1.0, color="gray", ls=":", alpha=0.5, label="Mélange parfait")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel("Pas de temps")
        axes[1].set_ylabel("Variance")
        axes[1].set_title("Décroissance de la ségrégation")
        axes[1].set_yscale("log")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("mixing_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    def plot_eigenvalues(self, folder_names=None, n_eigenvalues=20, figsize=(12, 5)):
        """
        Compare les valeurs propres des matrices de transition.
        
        Le 2ème plus grand eigenvalue contrôle la vitesse de mélange.
        """
        if folder_names is None:
            folder_names = []
            for method in sorted(self.by_method.keys()):
                exps = sorted(
                    self.by_method[method].items(),
                    key=lambda x: x[1]["matrix"].shape[0],
                )
                if exps:
                    folder_names.append(exps[len(exps) // 2][0])
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        lambda2_data = []
        
        for name in folder_names:
            if name not in self.results:
                continue
            
            data = self.results[name]
            method = data["method"]
            color = METHOD_COLORS.get(method, "#333")
            M = data["matrix"]
            
            # Valeurs propres (les n plus grandes)
            n_eig = min(n_eigenvalues, M.shape[0])
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(M)))[::-1][:n_eig]
            
            label = f"{method} ({M.shape[0]})"
            axes[0].plot(range(len(eigenvalues)), eigenvalues, "o-",
                        label=label, color=color, markersize=4)
            
            if len(eigenvalues) > 1:
                lambda2_data.append({
                    "name": name, "method": method,
                    "lambda2": eigenvalues[1],
                    "n_states": M.shape[0],
                })
        
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("|λ|")
        axes[0].set_title("Spectre des valeurs propres")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        
        # 2ème eigenvalue
        if lambda2_data:
            methods = [d["method"] for d in lambda2_data]
            l2s = [d["lambda2"] for d in lambda2_data]
            colors = [METHOD_COLORS.get(m, "#333") for m in methods]
            labels = [f"{d['method']}\n({d['n_states']})" for d in lambda2_data]
            
            axes[1].bar(range(len(l2s)), l2s, color=colors, alpha=0.8)
            axes[1].set_xticks(range(len(l2s)))
            axes[1].set_xticklabels(labels, fontsize=8)
            axes[1].set_ylabel("|λ₂|")
            axes[1].set_title("2ème valeur propre\n(plus petit = mélange plus rapide)")
            axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig("eigenvalues_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()
        
    def visualize_partitioner(self, experiment_name=None, size=700):
        """
        Visualise le partitionnement d'une expérience chargée.
        
        1. Essaie d'utiliser les données du partitionneur chargées depuis HF
        2. Si absent, le recrée depuis les paramètres de l'expérience
        
        Args:
            experiment_name: nom du dossier d'expérience (None = le premier chargé)
            size: taille du canvas
        
        Returns:
            HTML object pour affichage Jupyter
        """
        # Sélectionner l'expérience
        if experiment_name is None:
            if not self.results:
                raise ValueError("❌ Aucune expérience chargée. Utilisez load_all() d'abord.")
            experiment_name = list(self.results.keys())[0]
        
        if experiment_name not in self.results:
            raise ValueError(f"❌ Expérience '{experiment_name}' non trouvée. "
                            f"Disponibles : {list(self.results.keys())[:5]}...")
        
        exp_data = self.results[experiment_name]
        method = exp_data.get("method", "unknown")
        params = exp_data.get("params", {})
        partitioner_data = exp_data.get("partitioner_data")
        
        print(f"🔍 Visualisation du partitionnement pour '{experiment_name}'...")
        print(f"   Méthode: {method}")
        
        # ══════════════════════════════════════════════════════════════
        # ÉTAPE 1 : Utiliser les données du partitionneur si disponibles
        # ══════════════════════════════════════════════════════════════
        
        if partitioner_data is not None:
            print(f"   ✅ Données du partitionneur chargées: {partitioner_data.get('type')}")
            partitioner_type = partitioner_data.get("type")
            n_cells = partitioner_data.get("n_cells")
            print(f"      Type : {partitioner_type} | Cellules : {n_cells}")
            
            # Visualiser directement avec les données chargées
            if partitioner_type == "AdaptiveZPartitioner":
                # Récréer l'objet pour la visualisation
                partitioner = self._recreate_partitioner_from_params(exp_data)
                return self._visualize_adaptive(partitioner, size=size)
            elif partitioner_type == "CylindricalPartitioner":
                return self._visualize_cylindrical_with_data(partitioner_data, size=size)
            elif partitioner_type == "CartesianPartitioner":
                return self._visualize_cartesian_with_data(partitioner_data, size=size)
            elif partitioner_type == "MultiZonePartitioner":
                partitioner = self._recreate_partitioner_from_params(exp_data)
                return self._visualize_multizone(partitioner, size=size)
            elif partitioner_type == "VoronoiPartitioner":
                return self._visualize_voronoi_with_data(partitioner_data, size=size)
            elif partitioner_type == "SingleCellPartitioner":
                partitioner = self._recreate_partitioner_from_params(exp_data)
                return self._visualize_single(partitioner, size=size)
        
        # ══════════════════════════════════════════════════════════════
        # ÉTAPE 2 : Recréer depuis les paramètres
        # ══════════════════════════════════════════════════════════════
        
        print(f"   🔧 Recréation depuis les paramètres...")
        partitioner = self._recreate_partitioner_from_params(exp_data)
        
        # ══════════════════════════════════════════════════════════════
        # VISUALISATION
        # ══════════════════════════════════════════════════════════════
        
        partitioner_type = type(partitioner).__name__
        
        if partitioner_type == "AdaptiveZPartitioner":
            return self._visualize_adaptive(partitioner, size=size)
        elif partitioner_type == "MultiZonePartitioner":
            return self._visualize_multizone(partitioner, size=size)
        elif partitioner_type == "CylindricalPartitioner":
            return self._visualize_cylindrical(partitioner, size=size)
        elif partitioner_type == "CartesianPartitioner":
            return self._visualize_cartesian(partitioner, size=size)
        elif partitioner_type == "VoronoiPartitioner":
            return self._visualize_voronoi(partitioner, size=size)
        elif partitioner_type == "SingleCellPartitioner":
            return self._visualize_single(partitioner, size=size)
        else:
            return self._visualize_generic(partitioner, size=size)


    def _recreate_partitioner_from_params(self, exp_data):
        """
        Recrée un partitionneur depuis les paramètres de l'expérience.
        
        Args:
            exp_data: données de l'expérience depuis self.results
        
        Returns:
            partitioner configuré avec les bons paramètres
        """
        from src.partitioners import create_partitioner
        
        method = exp_data["method"]
        params = exp_data.get("params", {})
        matrix = exp_data["matrix"]
        n_states = matrix.shape[0]
        
        print(f"   🔧 Recréation du partitionneur {method}:")
        
        # ── Extraire les paramètres selon le format ──
        if "method_kwargs" in params:
            # Nouveau format
            method_kwargs = params["method_kwargs"]
            print(f"      Paramètres (nouveau format): {method_kwargs}")
        else:
            # Ancien format : déduire les paramètres
            method_kwargs = self._deduce_old_format_params(method, params, n_states)
            print(f"      Paramètres (déduits): {method_kwargs}")
        
        # ── Créer le partitionneur ──
        partitioner = create_partitioner(method, **method_kwargs)
        
        # ── Simuler un fit basique pour adaptive/multizone ──
        if method in ["adaptive", "multizone"]:
            # Ces méthodes ont besoin de z_min, z_max pour la visualisation
            # On utilise des valeurs par défaut raisonnables
            partitioner = self._simulate_basic_fit(partitioner, method_kwargs)
        
        print(f"      ✅ Partitionneur recréé: {partitioner.n_cells} cellules")
        
        return partitioner


    def _deduce_old_format_params(self, method, params, n_states):
        """
        Déduit les paramètres depuis l'ancien format ou les infos disponibles.
        
        Args:
            method: nom de la méthode
            params: paramètres de l'expérience
            n_states: nombre d'états depuis la matrice
        
        Returns:
            dict de paramètres pour create_partitioner
        """
        if method == "cartesian":
            # Ancien format : nx, ny, nz directement dans params
            nx = params.get("nx", 5)
            ny = params.get("ny", 5) 
            nz = params.get("nz", 5)
            
            # Vérifier cohérence avec n_states
            if nx * ny * nz != n_states:
                # Essayer de déduire depuis n_states (supposer cube)
                cube_root = round(n_states ** (1/3))
                if cube_root ** 3 == n_states:
                    nx = ny = nz = cube_root
                    print(f"      ⚠️  Paramètres incohérents, déduit cube {cube_root}³")
            
            return {"nx": nx, "ny": ny, "nz": nz}
        
        elif method == "cylindrical":
            # Essayer de déduire nr, ntheta, nz depuis n_states
            # Supposer nz=1 par défaut (cas fréquent)
            nz = 1
            remaining = n_states // nz
            
            # Essayer quelques combinaisons courantes
            for nr in [3, 4, 5, 6, 8, 10]:
                if remaining % nr == 0:
                    ntheta = remaining // nr
                    if ntheta >= 1 and ntheta <= 16:  # valeurs raisonnables
                        return {"nr": nr, "ntheta": ntheta, "nz": nz, "radial_mode": "equal_area"}
            
            # Fallback
            return {"nr": 5, "ntheta": 8, "nz": 1, "radial_mode": "equal_area"}
        
        elif method == "voronoi":
            return {"n_cells": n_states}
        
        elif method == "quantile":
            # Supposer cube
            cube_root = round(n_states ** (1/3))
            if cube_root ** 3 == n_states:
                return {"nx": cube_root, "ny": cube_root, "nz": cube_root}
            else:
                return {"nx": 5, "ny": 5, "nz": 5}
        
        elif method == "octree":
            return {"max_particles": 100, "max_depth": 5}
        
        elif method == "physics":
            return {"n_cells": n_states}
        
        elif method == "adaptive":
            # Paramètres par défaut
            return {
                "z_split": 0.75,
                "z_split_mode": "quantile",
                "n_cells_top": 1,
                "top_method": "single",
                "top_kwargs": {},
                "bottom_method": "cylindrical", 
                "bottom_kwargs": {"nr": 3, "ntheta": 4, "nz": 1, "radial_mode": "equal_area"}
            }
        
        elif method == "single":
            return {}
        
        else:
            print(f"      ⚠️  Méthode {method} inconnue, paramètres par défaut")
            return {}


    def _simulate_basic_fit(self, partitioner, method_kwargs):
        """
        Simule un fit basique pour les partitionneurs qui en ont besoin.
        
        Args:
            partitioner: instance du partitionneur
            method_kwargs: paramètres utilisés
        
        Returns:
            partitioner avec attributs basiques remplis
        """
        if type(partitioner).__name__ == "AdaptiveZPartitioner":
            # Remplir les attributs nécessaires pour la visualisation
            partitioner._z_min = 0.0
            partitioner._z_max = 1.0
            
            if partitioner.z_split_mode == "quantile":
                partitioner._z_split = partitioner.z_split_input or 0.75
            else:
                partitioner._z_split = partitioner.z_split_input or 0.5
            
            # Créer les sous-partitionneurs
            from src.partitioners import create_partitioner
            
            partitioner._bottom_partitioner = create_partitioner(
                partitioner.bottom_method, 
                **partitioner.bottom_kwargs
            )
            partitioner._n_cells_bottom = partitioner._bottom_partitioner.n_cells
            
            if partitioner.top_method == "single":
                partitioner._top_partitioner = None
                partitioner._n_cells_top = 1
            else:
                partitioner._top_partitioner = create_partitioner(
                    partitioner.top_method,
                    **partitioner.top_kwargs
                )
                partitioner._n_cells_top = partitioner._top_partitioner.n_cells
            
            print(f"      Zone basse: {partitioner._n_cells_bottom} cellules")
            print(f"      Zone haute: {partitioner._n_cells_top} cellules")
            print(f"      Z-split: {partitioner._z_split}")
        
        return partitioner


    def _visualize_cylindrical_with_data(self, partitioner_data, size=700):
        """
        Visualise un partitionnement cylindrique : le mélangeur vu de dessus.
        Le cercle représente le mélangeur qui est complètement partitionné.
        
        Args:
            partitioner_data: dict avec type, n_cells, r_max, r_edges, etc.
            size: taille du canvas
        
        Returns:
            HTML object pour Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        cid = f"cyl_data_viz_{uuid.uuid4().hex}"
        
        # Extraire les données
        n_cells = partitioner_data.get("n_cells", 0)
        r_max = partitioner_data.get("r_max", 1.0)
        r_edges = partitioner_data.get("r_edges")
        
        if r_edges is None:
            print("⚠️  r_edges non disponibles")
            return self._visualize_generic(None, size=size)
        
        # Convertir en liste
        r_edges_list = r_edges.tolist() if hasattr(r_edges, 'tolist') else list(r_edges)
        nr = len(r_edges_list) - 1
        nz = 1
        ntheta = max(1, n_cells // nr)
        
        # Palette de couleurs
        colors = [f"hsl({360 * i / max(1, n_cells)}, 75%, 50%)" for i in range(n_cells)]
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Partitionnement Cylindrique — Vue de dessus du mélangeur
        </h3>
        
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                <div>
                    <div style="font-size:13px; opacity:0.9;">Rayons (nr)</div>
                    <div style="font-size:18px; font-weight:bold;">{nr}</div>
                </div>
                <div>
                    <div style="font-size:13px; opacity:0.9;">Angles (nθ)</div>
                    <div style="font-size:18px; font-weight:bold;">{ntheta}</div>
                </div>
                <div>
                    <div style="font-size:13px; opacity:0.9;">Hauteurs (nz)</div>
                    <div style="font-size:18px; font-weight:bold;">{nz}</div>
                </div>
            </div>
            <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.3); 
                        text-align:center; font-size:16px; font-weight:bold;">
                Total : {n_cells} cellules | Rayon : {r_max:.4f}
            </div>
        </div>
        
        <canvas id="{cid}" width="{size}" height="{size}"
                style="border:2px solid #555; border-radius:10px; 
                    box-shadow:0 8px 16px rgba(0,0,0,0.15); display:block; margin:20px auto;
                    background:#f5f5f5;"></canvas>
        
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            
            const W = canvas.width;
            const H = canvas.height;
            const cx = W / 2;
            const cy = H / 2;
            const R_display = Math.min(W, H) * 0.35;
            
            const nr = {nr};
            const ntheta = {ntheta};
            const r_max_real = {r_max};
            const r_edges = {json.dumps(r_edges_list)};
            const colors = {json.dumps(colors)};
            
            // Normaliser les rayons pour l'affichage
            const r_edges_norm = r_edges.map(r => (r / r_max_real) * R_display);
            
            // ═════════════════════════════════════════════════════════
            // FONCTION POUR DESSINER UN SECTEUR ANNULAIRE
            // ═════════════════════════════════════════════════════════
            function drawAnnularSector(r_inner, r_outer, theta_start, theta_end, fillColor, strokeColor, lineWidth = 1) {{
                ctx.beginPath();
                
                // Arc intérieur
                ctx.arc(cx, cy, r_inner, theta_start, theta_end);
                
                // Ligne radiale de fin
                ctx.lineTo(cx + r_outer * Math.cos(theta_end), cy + r_outer * Math.sin(theta_end));
                
                // Arc extérieur (inverse)
                ctx.arc(cx, cy, r_outer, theta_end, theta_start, true);
                
                // Fermer le chemin
                ctx.closePath();
                
                // Remplir et tracer
                ctx.fillStyle = fillColor;
                ctx.fill();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth;
                ctx.stroke();
            }}
            
            // ═════════════════════════════════════════════════════════
            // DESSINER TOUTES LES PARTITIONS
            // ═════════════════════════════════════════════════════════
            for (let ir = 0; ir < nr; ir++) {{
                for (let itheta = 0; itheta < ntheta; itheta++) {{
                    const cell_id = ir + itheta * nr;
                    
                    const r_inner = r_edges_norm[ir];
                    const r_outer = r_edges_norm[ir + 1];
                    
                    // IMPORTANT: Couvrir 360° complètement
                    const theta_start = (2 * Math.PI * itheta) / ntheta;
                    const theta_end = (2 * Math.PI * (itheta + 1)) / ntheta;
                    
                    // Dessiner le secteur
                    drawAnnularSector(
                        r_inner, 
                        r_outer, 
                        theta_start, 
                        theta_end,
                        colors[cell_id],    // Couleur de remplissage
                        "#333",              // Couleur de bordure
                        0.5                  // Épaisseur bordure
                    );
                }}
            }}
            
            // ═════════════════════════════════════════════════════════
            // CONTOUR CIRCULAIRE DU MÉLANGEUR
            // ═════════════════════════════════════════════════════════
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, r_edges_norm[nr], 0, 2 * Math.PI);
            ctx.stroke();
            
            // ═════════════════════════════════════════════════════════
            // CERCLES RADIAUX (séparateurs de rayons)
            // ═════════════════════════════════════════════════════════
            ctx.strokeStyle = "rgba(0, 0, 0, 0.4)";
            ctx.lineWidth = 1;
            for (let ir = 1; ir < nr; ir++) {{
                ctx.beginPath();
                ctx.arc(cx, cy, r_edges_norm[ir], 0, 2 * Math.PI);
                ctx.stroke();
            }}
            
            // ═════════════════════════════════════════════════════════
            // LIGNES ANGULAIRES (séparateurs d'angles)
            // ═════════════════════════════════════════════════════════
            ctx.strokeStyle = "rgba(0, 0, 0, 0.4)";
            ctx.lineWidth = 1;
            for (let itheta = 0; itheta < ntheta; itheta++) {{
                const theta = (2 * Math.PI * itheta) / ntheta;
                const x_start = cx + r_edges_norm[0] * Math.cos(theta);
                const y_start = cy + r_edges_norm[0] * Math.sin(theta);
                const x_end = cx + r_edges_norm[nr] * Math.cos(theta);
                const y_end = cy + r_edges_norm[nr] * Math.sin(theta);
                
                ctx.beginPath();
                ctx.moveTo(x_start, y_start);
                ctx.lineTo(x_end, y_end);
                ctx.stroke();
            }}
            
            // ═════════════════════════════════════════════════════════
            // LABELS DES CELLULES
            // ═════════════════════════════════════════════════════════
            ctx.fillStyle = "#000";
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            
            for (let ir = 0; ir < nr; ir++) {{
                for (let itheta = 0; itheta < ntheta; itheta++) {{
                    const cell_id = ir + itheta * nr;
                    
                    // Position au centre du secteur
                    const r_mid = (r_edges_norm[ir] + r_edges_norm[ir + 1]) / 2;
                    const theta_mid = (2 * Math.PI * (itheta + 0.5)) / ntheta;
                    
                    const x = cx + r_mid * Math.cos(theta_mid);
                    const y = cy + r_mid * Math.sin(theta_mid);
                    
                    // Ombre de texte pour lisibilité
                    ctx.shadowColor = "rgba(255,255,255,0.8)";
                    ctx.shadowBlur = 3;
                    ctx.fillText(cell_id, x, y);
                    ctx.shadowColor = "transparent";
                }}
            }}
            
            // ═════════════════════════════════════════════════════════
            // CENTRE DU MÉLANGEUR
            // ═════════════════════════════════════════════════════════
            ctx.fillStyle = "#666";
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
            ctx.fill();
        }})();
        </script>
        
        <div style="margin-top:16px; padding:12px; background:#ecf0f1; border-radius:6px; font-size:13px; color:#34495e;">
            <strong>💡 Interprétation :</strong>
            <ul style="margin:8px 0; padding-left:20px;">
                <li><strong>Cercle extérieur</strong> : limite du mélangeur (rayon = {r_max:.4f})</li>
                <li><strong>Anneaux</strong> : {nr} couches radiales (nr = {nr})</li>
                <li><strong>Secteurs</strong> : {ntheta} divisions angulaires (ntheta = {ntheta})</li>
                <li><strong>Chaque couleur</strong> = une partition (cellule d'état)</li>
                <li><strong>Le mélangeur est ENTIÈREMENT partitionné</strong> (pas d'espace vide)</li>
            </ul>
        </div>
        
        </div>
        """
        return HTML(html)


    def _visualize_cartesian_with_data(self, partitioner_data, size=700):
        """
        Visualise un partitionnement cartésien avec données réelles.
        
        Args:
            partitioner_data: dict avec type, n_cells, nx, ny, nz
            size: taille du canvas
        
        Returns:
            HTML object pour Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        cid = f"cart_data_viz_{uuid.uuid4().hex}"
        n_cells = partitioner_data.get("n_cells", 0)
        
        # Déduire grille
        nx = partitioner_data.get("nx")
        ny = partitioner_data.get("ny")
        nz = partitioner_data.get("nz")
        
        if nx is None or ny is None or nz is None:
            cube_root = round(n_cells ** (1/3))
            if cube_root ** 3 == n_cells:
                nx = ny = nz = cube_root
            else:
                nx = ny = round(np.sqrt(n_cells))
                nz = max(1, n_cells // (nx * ny))
        
        colors = [f"hsl({360 * i / max(1, n_cells)}, 70%, 50%)" for i in range(n_cells)]
        display_z = 0
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Partitionnement Cartésien (Données réelles)
        </h3>
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                <div><div style="font-size:13px; opacity:0.9;">X (nx)</div><div style="font-size:18px; font-weight:bold;">{nx}</div></div>
                <div><div style="font-size:13px; opacity:0.9;">Y (ny)</div><div style="font-size:18px; font-weight:bold;">{ny}</div></div>
                <div><div style="font-size:13px; opacity:0.9;">Z (nz)</div><div style="font-size:18px; font-weight:bold;">{nz}</div></div>
            </div>
            <div style="margin-top:12px; text-align:center; font-weight:bold;">
                Total : {n_cells} cellules
            </div>
        </div>
        <canvas id="{cid}" width="{size}" height="{size}" style="border:2px solid #bdc3c7; border-radius:10px; 
                    box-shadow:0 8px 16px rgba(0,0,0,0.15); display:block; margin:20px auto; background:white;"></canvas>
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            const W = canvas.width, H = canvas.height, nx = {nx}, ny = {ny};
            const cell_w = (W - 40) / nx, cell_h = (H - 40) / ny, margin = 20;
            const colors = {json.dumps(colors)};
            for (let ix = 0; ix < nx; ix++) {{
                for (let iy = 0; iy < ny; iy++) {{
                    const cell_id = ix + iy * nx;
                    const x = margin + ix * cell_w, y = margin + iy * cell_h;
                    ctx.fillStyle = colors[cell_id];
                    ctx.fillRect(x, y, cell_w, cell_h);
                    ctx.strokeStyle = "#333";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x, y, cell_w, cell_h);
                    ctx.fillStyle = "#000";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(cell_id, x + cell_w/2, y + cell_h/2);
                }}
            }}
        }})();
        </script>
        </div>
        """
        return HTML(html)


    def _visualize_voronoi_with_data(self, partitioner_data, size=700):
        """
        Visualise un partitionnement Voronoï avec centroids.
        
        Args:
            partitioner_data: dict avec type, n_cells, centroids
            size: taille du canvas
        
        Returns:
            HTML object pour Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        cid = f"vor_data_viz_{uuid.uuid4().hex}"
        n_cells = partitioner_data.get("n_cells", 0)
        centroids = partitioner_data.get("centroids")
        
        if centroids is None:
            return self._visualize_generic(None, size=size)
        
        # Extraire coordonnées 2D (projection X-Y)
        if len(centroids.shape) > 1 and centroids.shape[1] >= 2:
            cents_2d = centroids[:, :2]
        else:
            cents_2d = np.array([[i % 10, i // 10] for i in range(n_cells)])
        
        colors = [f"hsl({360 * i / max(1, n_cells)}, 70%, 50%)" for i in range(n_cells)]
        cents_2d_list = cents_2d.tolist()
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Partitionnement Voronoï (Données réelles)
        </h3>
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; text-align:center; font-weight:bold;">
            {n_cells} cellules (K-means)
        </div>
        <canvas id="{cid}" width="{size}" height="{size}" style="border:2px solid #bdc3c7; border-radius:10px; 
                    box-shadow:0 8px 16px rgba(0,0,0,0.15); display:block; margin:20px auto; background:white;"></canvas>
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            const W = canvas.width, H = canvas.height, margin = 20;
            const centroids = {json.dumps(cents_2d_list)};
            const colors = {json.dumps(colors)};
            
            let x_min = Math.min(...centroids.map(c => c[0])),
                x_max = Math.max(...centroids.map(c => c[0])),
                y_min = Math.min(...centroids.map(c => c[1])),
                y_max = Math.max(...centroids.map(c => c[1]));
            
            const x_range = x_max - x_min || 1, y_range = y_max - y_min || 1;
            
            centroids.forEach((c, i) => {{
                const cx = margin + (c[0] - x_min) / x_range * (W - 2*margin);
                const cy = margin + (c[1] - y_min) / y_range * (H - 2*margin);
                ctx.fillStyle = colors[i];
                ctx.beginPath();
                ctx.arc(cx, cy, 6, 0, 2*Math.PI);
                ctx.fill();
                ctx.strokeStyle = "#333";
                ctx.lineWidth = 2;
                ctx.stroke();
            }});
        }})();
        </script>
        </div>
        """
        return HTML(html)


    def _visualize_cylindrical(self, partitioner, size=700):
        """
        Visualise un partitionnement cylindrique en coordonnées polaires (r, θ).
        
        Args:
            partitioner: CylindricalPartitioner instance
            size: taille du canvas
        
        Returns:
            HTML object pour Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        cid = f"cyl_viz_{uuid.uuid4().hex}"
        
        # Paramètres cylindriques
        nr = partitioner.nr
        ntheta = partitioner.ntheta
        nz = partitioner.nz
        r_max = partitioner._r_max if partitioner._r_max else 1.0
        r_edges = partitioner._r_edges if partitioner._r_edges is not None else np.linspace(0, r_max, nr + 1)
        radial_mode = partitioner.radial_mode
        
        # Palette de couleurs
        n_cells = partitioner.n_cells
        colors = [f"hsl({360 * i / n_cells}, 70%, 50%)" for i in range(n_cells)]
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Partitionnement Cylindrique (Vue de dessus)
        </h3>
        
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                <div>
                    <div style="font-size:13px; opacity:0.9;">Rayons (nr)</div>
                    <div style="font-size:18px; font-weight:bold;">{nr}</div>
                </div>
                <div>
                    <div style="font-size:13px; opacity:0.9;">Angles (nθ)</div>
                    <div style="font-size:18px; font-weight:bold;">{ntheta}</div>
                </div>
                <div>
                    <div style="font-size:13px; opacity:0.9;">Hauteurs (nz)</div>
                    <div style="font-size:18px; font-weight:bold;">{nz}</div>
                </div>
            </div>
            <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.3); 
                        text-align:center; font-size:16px; font-weight:bold;">
                Total : {n_cells} cellules | Mode radial : {radial_mode}
            </div>
        </div>
        
        <canvas id="{cid}" width="{size}" height="{size}"
                style="border:2px solid #bdc3c7; border-radius:10px; 
                    box-shadow:0 8px 16px rgba(0,0,0,0.15); display:block; margin:20px auto;
                    background:white;"></canvas>
        
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            
            const W = canvas.width;
            const H = canvas.height;
            const cx = W / 2;
            const cy = H / 2;
            const R_display = Math.min(W, H) * 0.4;
            
            const nr = {nr};
            const ntheta = {ntheta};
            const nz = {nz};
            const r_max = {r_max};
            const radial_mode = "{radial_mode}";
            const r_edges = {json.dumps(r_edges.tolist())};
            const colors = {json.dumps(colors)};
            
            // Dessiner les partitions
            for (let iz = 0; iz < nz; iz++) {{
                for (let ir = 0; ir < nr; ir++) {{
                    for (let itheta = 0; itheta < ntheta; itheta++) {{
                        const cell_id = ir + itheta * nr + iz * nr * ntheta;
                        
                        const r_inner = r_edges[ir] / r_max * R_display;
                        const r_outer = r_edges[ir + 1] / r_max * R_display;
                        
                        const theta_start = (2 * Math.PI * itheta) / ntheta;
                        const theta_end = (2 * Math.PI * (itheta + 1)) / ntheta;
                        
                        // Dessiner le secteur annulaire
                        ctx.fillStyle = colors[cell_id];
                        ctx.strokeStyle = "#333";
                        ctx.lineWidth = 1;
                        
                        ctx.beginPath();
                        for (let r = r_inner; r <= r_outer; r += (r_outer - r_inner) / 20) {{
                            const x = cx + r * Math.cos(theta_start);
                            const y = cy + r * Math.sin(theta_start);
                            if (r === r_inner) ctx.moveTo(x, y);
                            else ctx.lineTo(x, y);
                        }}
                        
                        ctx.arc(cx, cy, r_outer, theta_start, theta_end);
                        
                        for (let r = r_outer; r >= r_inner; r -= (r_outer - r_inner) / 20) {{
                            const x = cx + r * Math.cos(theta_end);
                            const y = cy + r * Math.sin(theta_end);
                            ctx.lineTo(x, y);
                        }}
                        
                        ctx.arc(cx, cy, r_inner, theta_end, theta_start, true);
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                    }}
                }}
            }}
            
            // Dessiner les cercles de rayon
            ctx.strokeStyle = "rgba(0,0,0,0.3)";
            ctx.lineWidth = 1;
            for (let ir = 1; ir < nr; ir++) {{
                const r = r_edges[ir] / r_max * R_display;
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, 2 * Math.PI);
                ctx.stroke();
            }}
            
            // Labels de cellule
            ctx.fillStyle = "#000";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            
            for (let itheta = 0; itheta < ntheta; itheta++) {{
                for (let ir = 0; ir < nr; ir++) {{
                    const cell_id = ir + itheta * nr;
                    const r_mid = (r_edges[ir] + r_edges[ir + 1]) / 2 / r_max * R_display;
                    const theta_mid = (2 * Math.PI * (itheta + 0.5)) / ntheta;
                    const x = cx + r_mid * Math.cos(theta_mid);
                    const y = cy + r_mid * Math.sin(theta_mid);
                    ctx.fillText(cell_id, x, y);
                }}
            }}
        }})();
        </script>
        
        </div>
        """
        return HTML(html)


    def _visualize_cartesian(self, partitioner, size=700):
        """
        Visualise un partitionnement cartésien en grille 3D (projection 2D).
        
        Args:
            partitioner: CartesianPartitioner instance
            size: taille du canvas
        
        Returns:
            HTML object pour Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        cid = f"cart_viz_{uuid.uuid4().hex}"
        
        nx = partitioner.nx
        ny = partitioner.ny
        nz = partitioner.nz
        n_cells = partitioner.n_cells
        
        # Palette de couleurs
        colors = [f"hsl({360 * i / n_cells}, 70%, 50%)" for i in range(n_cells)]
        
        # On affiche une tranche (z=0) en 2D
        display_z = 0
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Partitionnement Cartésien (Grille régulière)
        </h3>
        
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
                <div>
                    <div style="font-size:13px; opacity:0.9;">X (nx)</div>
                    <div style="font-size:18px; font-weight:bold;">{nx}</div>
                </div>
                <div>
                    <div style="font-size:13px; opacity:0.9;">Y (ny)</div>
                    <div style="font-size:18px; font-weight:bold;">{ny}</div>
                </div>
                <div>
                    <div style="font-size:13px; opacity:0.9;">Z (nz)</div>
                    <div style="font-size:18px; font-weight:bold;">{nz}</div>
                </div>
            </div>
            <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.3); 
                        text-align:center; font-size:16px; font-weight:bold;">
                Total : {n_cells} cellules
            </div>
        </div>
        
        <canvas id="{cid}" width="{size}" height="{size}"
                style="border:2px solid #bdc3c7; border-radius:10px; 
                    box-shadow:0 8px 16px rgba(0,0,0,0.15); display:block; margin:20px auto;
                    background:white;"></canvas>
        
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            
            const W = canvas.width;
            const H = canvas.height;
            const nx = {nx};
            const ny = {ny};
            const nz = {nz};
            const iz_display = {display_z};
            
            const cell_w = (W - 40) / nx;
            const cell_h = (H - 40) / ny;
            const margin = 20;
            
            const colors = {json.dumps(colors)};
            
            // Dessiner la grille
            for (let ix = 0; ix < nx; ix++) {{
                for (let iy = 0; iy < ny; iy++) {{
                    const cell_id = ix + iy * nx + iz_display * nx * ny;
                    const x = margin + ix * cell_w;
                    const y = margin + iy * cell_h;
                    
                    // Rectangle de la cellule
                    ctx.fillStyle = colors[cell_id];
                    ctx.fillRect(x, y, cell_w, cell_h);
                    
                    // Bordure
                    ctx.strokeStyle = "#333";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x, y, cell_w, cell_h);
                    
                    // Numéro de cellule
                    ctx.fillStyle = "#000";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(cell_id, x + cell_w/2, y + cell_h/2);
                }}
            }}
        }})();
        </script>
        
        </div>
        """
        return HTML(html)


    def _visualize_multizone(self, partitioner, size=700):
        """
        Visualise un partitionnement MultiZonePartitioner.
        
        Affiche N zones horizontales, chaque zone pouvant avoir une architecture différente.
        """
        import uuid
        import json
        from IPython.display import HTML
        
        cid = f"mz_viz_{uuid.uuid4().hex}"
        n_zones = len(partitioner._zones)
        n_total = partitioner.n_cells
        
        zones_info = []
        for i, (z_min, z_max, part) in enumerate(partitioner._zones):
            zones_info.append({
                "z_min": z_min,
                "z_max": z_max,
                "n_cells": part.n_cells,
                "method": type(part).__name__,
            })
        
        zone_colors = [f"hsl({360 * i / max(1, n_zones)}, 60%, 40%)" for i in range(n_zones)]
        
        all_colors = []
        for i, info in enumerate(zones_info):
            n_c = info["n_cells"]
            hue_base = 360 * i / max(1, n_zones)
            for j in range(n_c):
                hue = (hue_base + 360 * j / max(1, n_c)) % 360
                all_colors.append(f"hsl({hue}, 70%, 50%)")
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="margin-bottom:12px; color:#2c3e50;">🔍 Partitionnement MULTI-ZONE</h3>
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:8px; color:white;">
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:12px;">
        """
        for i in range(n_zones):
            info = zones_info[i]
            html += f'<div style="background:rgba(255,255,255,0.2); padding:8px; border-radius:4px;"><div style="font-size:12px;">Zone {i}</div><div style="font-size:14px; font-weight:bold;">{info["n_cells"]} cellules</div></div>'
        html += f'</div><div style="text-align:center; margin-top:8px; font-weight:bold;">Total : {n_total} cellules</div></div>'
        html += f'<canvas id="{cid}" width="{size}" height="{size}" style="border:2px solid #bdc3c7; border-radius:10px; display:block; margin:20px auto; background:white;"></canvas>'
        html += f'''<script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            const W = canvas.width, H = canvas.height, margin = 40, plot_w = W - 2*margin, plot_h = H - 2*margin;
            ctx.strokeStyle = "#333"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(margin, H - margin); ctx.lineTo(margin, margin); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(margin, H - margin); ctx.lineTo(W - margin, H - margin); ctx.stroke();
            const n_zones = {n_zones}, zones_info = {json.dumps(zones_info)}, all_colors = {json.dumps(all_colors)};
            let cell_offset = 0, z_offset = 0;
            for (let zone_idx = 0; zone_idx < n_zones; zone_idx++) {{
                const info = zones_info[zone_idx];
                const n_cells_in_zone = info.n_cells;
                const h_zone = plot_h / n_zones, y_zone = margin + z_offset;
                const w_cell = plot_w / Math.max(...zones_info.map(z => z.n_cells));
                for (let i = 0; i < n_cells_in_zone; i++) {{
                    const x = margin + i * w_cell;
                    ctx.fillStyle = all_colors[cell_offset]; ctx.fillRect(x, y_zone, w_cell, h_zone);
                    ctx.strokeStyle = "#333"; ctx.lineWidth = 1; ctx.strokeRect(x, y_zone, w_cell, h_zone);
                    ctx.fillStyle = "#000"; ctx.font = "bold 11px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText(cell_offset, x + w_cell/2, y_zone + h_zone/2);
                    cell_offset++;
                }}
                ctx.strokeStyle = "rgba(0, 0, 0, 0.4)"; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(margin, y_zone + h_zone); ctx.lineTo(W - margin, y_zone + h_zone); ctx.stroke();
                z_offset += h_zone;
            }}
        }})();
        </script>'''
        html += '<div style="margin-top:16px; padding:12px; background:#ecf0f1; border-radius:6px; font-size:12px; color:#34495e;">'
        html += '<strong>💡 Zone Details:</strong><table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:11px;">'
        html += '<tr style="background:#d0d0d0; font-weight:bold;"><th style="padding:4px; border:1px solid #999;">Zone</th><th style="padding:4px; border:1px solid #999;">z_min-z_max</th><th style="padding:4px; border:1px solid #999;">Cellules</th><th style="padding:4px; border:1px solid #999;">Méthode</th></tr>'
        for i, info in enumerate(zones_info):
            html += f'<tr style="border:1px solid #bbb;"><td style="padding:4px; border:1px solid #999;">Zone {i}</td><td style="padding:4px; border:1px solid #999;">[{info["z_min"]:.3f}, {info["z_max"]:.3f}]</td><td style="padding:4px; border:1px solid #999;">{info["n_cells"]}</td><td style="padding:4px; border:1px solid #999;">{info["method"]}</td></tr>'
        html += '</table></div></div>'
        return HTML(html)


    def _visualize_voronoi(self, partitioner, size=700):
        """
        Visualise un partitionnement Voronoï.
        
        Args:
            partitioner: VoronoiPartitioner instance
            size: taille du canvas
        
        Returns:
            HTML object pour Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        cid = f"vor_viz_{uuid.uuid4().hex}"
        n_cells = partitioner.n_cells
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Partitionnement Voronoï
        </h3>
        
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="text-align:center; font-size:16px; font-weight:bold;">
                Total : {n_cells} cellules (Clustering K-means)
            </div>
        </div>
        
        <p>Visualisation Voronoï — {n_cells} centroids K-means</p>
        
        </div>
        """
        return HTML(html)


    def _visualize_adaptive(self, partitioner, size=700):
        """
        Visualise un partitionnement AdaptiveZPartitioner.
        
        Affiche deux zones côte-à-côte:
        - Zone basse (70-80% du mixer): partitionnement FINE
        - Zone haute (20-30% du mixer): partitionnement GROSSIER (généralement 1 cellule)
        """
        import uuid
        import json
        from IPython.display import HTML
        
        cid = f"adapt_viz_{uuid.uuid4().hex}"
        
        n_top = partitioner.n_cells_top
        n_bot = partitioner.n_cells_bottom
        n_total = partitioner.n_cells
        z_split = partitioner._z_split
        z_min = partitioner._z_min
        z_max = partitioner._z_max
        
        if z_max - z_min > 0:
            pct_bot = 100 * (z_split - z_min) / (z_max - z_min)
            pct_top = 100 * (z_max - z_split) / (z_max - z_min)
        else:
            pct_bot = pct_top = 50
        
        colors_bot = [f"hsl({360 * i / max(1, n_bot)}, 70%, 50%)" for i in range(n_bot)]
        colors_top = [f"hsl(200, 40%, 50%)"] * n_top
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="margin-bottom:12px; color:#2c3e50;">🔍 Partitionnement ADAPTATIF (Zone Haute / Basse)</h3>
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:8px; color:white;">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px;">
                <div><div style="font-size:13px; opacity:0.9;">Zone Basse</div><div style="font-size:18px; font-weight:bold;">{n_bot} cellules</div></div>
                <div><div style="font-size:13px; opacity:0.9;">Zone Haute</div><div style="font-size:18px; font-weight:bold;">{n_top} cellule(s)</div></div>
                <div><div style="font-size:13px; opacity:0.9;">Limite (z)</div><div style="font-size:18px; font-weight:bold;">{z_split:.3f}</div></div>
                <div><div style="font-size:13px; opacity:0.9;">Total</div><div style="font-size:18px; font-weight:bold;">{n_total} cellules</div></div>
            </div>
        </div>
        <canvas id="{cid}" width="{size}" height="{size}" style="border:2px solid #bdc3c7; border-radius:10px; display:block; margin:20px auto; background:white;"></canvas>
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            const W = canvas.width, H = canvas.height;
            const margin = 40, plot_w = W - 2*margin, plot_h = H - 2*margin;
            ctx.strokeStyle = "#333"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(margin, H - margin); ctx.lineTo(margin, margin); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(margin, H - margin); ctx.lineTo(W - margin, H - margin); ctx.stroke();
            const h_bot = plot_h * {pct_bot} / 100, h_top = plot_h * {pct_top} / 100;
            const colors_bot = {json.dumps(colors_bot)}, colors_top = {json.dumps(colors_top)};
            const n_bot = {n_bot}, n_top = {n_top}, w_cell_bot = plot_w / n_bot, w_cell_top = plot_w / n_top;
            for (let i = 0; i < n_bot; i++) {{
                const x = margin + i * w_cell_bot, y = margin + h_top;
                ctx.fillStyle = colors_bot[i]; ctx.fillRect(x, y, w_cell_bot, h_bot);
                ctx.strokeStyle = "#333"; ctx.lineWidth = 1; ctx.strokeRect(x, y, w_cell_bot, h_bot);
                ctx.fillStyle = "#000"; ctx.font = "bold 12px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(i, x + w_cell_bot/2, y + h_bot/2);
            }}
            for (let i = 0; i < n_top; i++) {{
                const x = margin + i * w_cell_top, y = margin;
                ctx.fillStyle = colors_top[i]; ctx.fillRect(x, y, w_cell_top, h_top);
                ctx.strokeStyle = "#888"; ctx.lineWidth = 2; ctx.strokeRect(x, y, w_cell_top, h_top);
                ctx.fillStyle = "#fff"; ctx.font = "bold 12px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText({n_bot} + i, x + w_cell_top/2, y + h_top/2);
            }}
            ctx.strokeStyle = "red"; ctx.lineWidth = 3; ctx.setLineDash([5, 5]);
            ctx.beginPath(); ctx.moveTo(margin, margin + h_top); ctx.lineTo(W - margin, margin + h_top); ctx.stroke(); ctx.setLineDash([]);
        }})();
        </script>
        <div style="margin-top:16px; padding:12px; background:#ecf0f1; border-radius:6px; font-size:13px; color:#34495e;">
            <strong>💡 Interprétation :</strong>
            <ul style="margin:8px 0; padding-left:20px;">
            <li><strong>Zone BASSE</strong>: {pct_bot:.0f}% du mixer — {n_bot} cellules</li>
            <li><strong>Zone HAUTE</strong>: {pct_top:.0f}% du mixer — {n_top} cellule(s)</li>
            <li><strong>Numérotation</strong>: cellules 0···{n_bot-1} en bas, puis {n_bot}···{n_total-1} en haut</li>
            </ul>
        </div>
        </div>
        """
        return HTML(html)

    def _visualize_single(self, partitioner, size=700):
        """
        Visualise un partitionnement SingleCellPartitioner.
        
        Trivial: une seule cellule pour tout le domaine.
        """
        import uuid
        from IPython.display import HTML
        
        cid = f"single_viz_{uuid.uuid4().hex}"
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="margin-bottom:12px; color:#2c3e50;">🔍 Partitionnement SINGLE CELL</h3>
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:8px; color:white;">
            <div style="text-align:center; font-size:20px; font-weight:bold;">⚪ 1 SEULE CELLULE</div>
            <div style="text-align:center; margin-top:8px; font-size:14px; opacity:0.9;">Tout le domaine = une seule partition</div>
        </div>
        <canvas id="{cid}" width="{size}" height="300" style="border:2px solid #bdc3c7; border-radius:10px; display:block; margin:20px auto; background:white;"></canvas>
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            const W = canvas.width, H = canvas.height, margin = 50, rect_w = W - 2*margin, rect_h = H - 2*margin, x = margin, y = margin;
            ctx.fillStyle = "#4CAF50"; ctx.fillRect(x, y, rect_w, rect_h);
            ctx.strokeStyle = "#333"; ctx.lineWidth = 4; ctx.strokeRect(x, y, rect_w, rect_h);
            ctx.fillStyle = "#fff"; ctx.font = "bold 48px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("Cellule 0", W/2, H/2);
            ctx.fillStyle = "#333"; ctx.font = "14px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "top";
            ctx.fillText("Toutes les particules sont assignées à la même partition", W/2, y + rect_h + 20);
        }})();
        </script>
        <div style="margin-top:24px; padding:16px; background:#fff3cd; border-left:4px solid #ffc107; border-radius:6px; font-size:13px; color:#333;">
            <strong>⚠️ Note :</strong>
            <p style="margin:8px 0;">Le partitionnement SINGLE CELL n'a pas d'intérêt pour l'étude du mélange:</p>
            <ul style="margin:8px 0; padding-left:20px;">
            <li>Toutes les particules sont dans la même cellule</li>
            <li>Matrice de transition = [[1.0]] triviale</li>
            <li>Utilisé comme baseline ou pour les zones vides d'un partitionnement adaptatif</li>
            </ul>
        </div>
        </div>
        """
        return HTML(html)

    def _visualize_generic(self, partitioner, size=700):
        """
        Visualise un partitionneur générique.
        
        Args:
            partitioner: BasePartitioner instance
            size: taille du canvas
        
        Returns:
            HTML object pour Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        partitioner_type = type(partitioner).__name__
        n_cells = partitioner.n_cells
        label = partitioner.label if hasattr(partitioner, 'label') else "Unknown"
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Partitionnement: {partitioner_type}
        </h3>
        
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="text-align:center;">
                <div style="font-size:16px; font-weight:bold; margin-bottom:8px;">
                    {partitioner_type}
                </div>
                <div style="font-size:14px;">
                    Type : {label}
                </div>
                <div style="font-size:16px; font-weight:bold; margin-top:8px;">
                    Total : {n_cells} cellules
                </div>
            </div>
        </div>
        
        </div>
        """
        return HTML(html)


    def list_available_visualizations(self):
        """
        Liste les expériences pour lesquelles on peut faire une visualisation.
        
        Returns:
            dict: {experiment_name: {"method": str, "can_visualize": bool, "reason": str}}
        """
        result = {}
        
        print("🔍 Vérification des visualisations possibles...")
        
        for name, exp_data in self.results.items():
            method = exp_data.get("method", "unknown")
            can_viz = True
            reason = "OK"
            
            # Vérifier si la méthode est supportée
            if method in ["unknown"]:
                can_viz = False
                reason = "Méthode inconnue"
            
            result[name] = {
                "method": method,
                "can_visualize": can_viz, 
                "reason": reason
            }
        
        # Résumé par méthode
        by_method = defaultdict(list)
        for name, info in result.items():
            by_method[info["method"]].append(name)
        
        print(f"\n📊 Résumé par méthode:")
        for method, names in sorted(by_method.items()):
            can_viz_count = sum(1 for name in names if result[name]["can_visualize"])
            print(f"   {method:15s}: {can_viz_count}/{len(names)} visualisables")
        
        return result
# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    analyzer = MarkovAnalyzer()
    
    # Charger tout
    analyzer.load_all()
    
    # Résumé
    analyzer.print_summary()
    
    # Comparaison inter-méthodes
    if len(analyzer.get_methods()) > 1:
        analyzer.compare_methods(metric="diag_mean")
    
    # Analyse par méthode
    for method in analyzer.get_methods():
        n_exps = len(analyzer.get_experiments(method))
        if n_exps > 2:
            print(f"\n📊 Sweep {method.upper()} ({n_exps} expériences):")
            analyzer.compare_within_method(method, sweep_param="n_states")
    
    # Comparaison du mélange
    analyzer.plot_mixing_comparison(n_steps=200)
    
    # Spectre des eigenvalues
    analyzer.plot_eigenvalues()
    
    # Visualisation détaillée d'une expérience
    if analyzer.results:
        first = list(analyzer.results.keys())[0]
        analyzer.plot_experiment(first, n_steps=100)
    
    print("\n✨ Analyse terminée!")
