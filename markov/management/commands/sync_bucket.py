"""
Commande Django pour synchroniser la base de données
avec le bucket HuggingFace.

Usage:
    python manage.py sync_bucket
    python manage.py sync_bucket --method voronoi
    python manage.py sync_bucket --recompute-spectra
"""

import numpy as np
import json
import io
from django.core.management.base import BaseCommand
from django.conf import settings
from huggingface_hub import HfFileSystem

from markov.models import (
    PartitionMethod, Experiment, TransitionMatrix, RSDResult
)


class Command(BaseCommand):
    help = "Synchronise les expériences depuis le bucket HuggingFace"

    def add_arguments(self, parser):
        parser.add_argument(
            "--method", type=str, default=None,
            help="Filtrer par méthode (cartesian, voronoi, ...)"
        )
        parser.add_argument(
            "--recompute-spectra", action="store_true",
            help="Recalculer les valeurs propres"
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Lister sans importer"
        )

    def handle(self, *args, **options):
        fs = HfFileSystem()
        method_filter = options["method"]
        dry_run = options["dry_run"]
        recompute = options["recompute_spectra"]

        self.stdout.write(self.style.SUCCESS("🔄 Synchronisation du bucket..."))

        total_imported = 0
        total_skipped = 0
        total_errors = 0

        for prefix in settings.HF_BUCKET_PREFIXES:
            base = f"hf://buckets/{settings.HF_BUCKET_ID}/{prefix}"

            try:
                items = fs.ls(base)
            except FileNotFoundError:
                self.stdout.write(f"   ⚠️ {base} non trouvé")
                continue

            folders = sorted([
                item["name"].split("/")[-1]
                for item in items
                if item["type"] == "directory"
            ])

            self.stdout.write(f"\n📂 {prefix}: {len(folders)} dossiers")

            for folder_name in folders:
                # ── Filtrage par méthode ──
                if method_filter:
                    method = self._detect_method(folder_name)
                    if method != method_filter:
                        continue

                # ── Déjà importé ? ──
                if Experiment.objects.filter(folder_name=folder_name).exists():
                    if not recompute:
                        total_skipped += 1
                        continue

                if dry_run:
                    self.stdout.write(f"   📋 {folder_name}")
                    continue

                # ── Importer ──
                try:
                    self._import_experiment(fs, base, prefix, folder_name, recompute)
                    total_imported += 1
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"   ❌ {folder_name}: {e}")
                    )
                    total_errors += 1

        self.stdout.write(self.style.SUCCESS(
            f"\n✅ Importés: {total_imported} | "
            f"Ignorés: {total_skipped} | "
            f"Erreurs: {total_errors}"
        ))

    def _detect_method(self, folder_name):
        """Détecte la méthode depuis le nom du dossier."""
        prefixes = {
            "cartesian_": "cartesian",
            "cylindrical_": "cylindrical",
            "voronoi_": "voronoi",
            "quantile_": "quantile",
            "octree_": "octree",
            "physics_": "physics",
            "NLT_": "cartesian",  # ancien format
        }
        for prefix, method in prefixes.items():
            if folder_name.startswith(prefix):
                return method
        return "unknown"

    def _load_json(self, fs, path):
        try:
            with fs.open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_matrix(self, fs, path):
        with fs.open(path, "rb") as f:
            return np.load(io.BytesIO(f.read()))

    def _import_experiment(self, fs, base, prefix, folder_name, recompute):
        """Importe une expérience dans la base de données."""
        path = f"{base}/{folder_name}"

        # ── Charger les métadonnées ──
        config = self._load_json(fs, f"{path}/config.json")
        if not config:
            config = self._load_json(fs, f"{path}/params.json")

        stats = self._load_json(fs, f"{path}/stats.json")

        # ── Détecter la méthode ──
        method_name = self._detect_method(folder_name)
        if "method" in config:
            method_name = config["method"]

        # ── Paramètres du partitionneur ──
        method_kwargs = config.get("method_kwargs", {})
        if not method_kwargs and "nx" in config:
            method_kwargs = {
                k: config[k] for k in ["nx", "ny", "nz"]
                if k in config
            }

        # ── N_cells ──
        n_cells = stats.get("n_states", 0)
        if n_cells == 0 and method_kwargs:
            if "n_cells" in method_kwargs:
                n_cells = method_kwargs["n_cells"]
            elif "nx" in method_kwargs:
                n_cells = (
                    method_kwargs.get("nx", 1) *
                    method_kwargs.get("ny", 1) *
                    method_kwargs.get("nz", 1)
                )

        # ── Créer le PartitionMethod ──
        # Label unique pour ce partitionneur
        label = folder_name.split("_NLT")[0].split("_step")[0].split("_start")[0]
        if label.startswith("NLT_"):
            # Ancien format: NLT_100_nx5_ny5_nz5_...
            parts = folder_name.split("_")
            nx = ny = nz = "?"
            for p in parts:
                if p.startswith("nx"):
                    nx = p[2:]
                elif p.startswith("ny"):
                    ny = p[2:]
                elif p.startswith("nz"):
                    nz = p[2:]
            label = f"cartesian_nx{nx}_ny{ny}_nz{nz}"

        partition_method, created = PartitionMethod.objects.update_or_create(
            label=label,
            defaults={
                "name": method_name,
                "parameters": method_kwargs,
                "n_cells": n_cells,
                "n_cells_visited": stats.get("n_states_visited", 0),
                "population_mean": stats.get("pop_mean"),
                "population_std": stats.get("pop_std"),
                "population_cv": (
                    stats.get("pop_std", 0) / max(stats.get("pop_mean", 1), 1e-10)
                    if stats.get("pop_mean") else None
                ),
            },
        )

        # ── Hyperparamètres ──
        nlt = config.get("nlt") or config.get("NLT") or stats.get("n_timesteps_used", 0)
        if isinstance(nlt, str):
            nlt = int(nlt)
        step_size = config.get("step_size", 1)
        start_index = config.get("start_index", 0)

        # Ancien format
        if not nlt and folder_name.startswith("NLT_"):
            try:
                nlt = int(folder_name.split("_")[1])
            except (ValueError, IndexError):
                nlt = 100

        # ── Créer l'Experiment ──
        experiment, _ = Experiment.objects.update_or_create(
            folder_name=folder_name,
            defaults={
                "partition_method": partition_method,
                "nlt": max(nlt or 1, 1),
                "step_size": max(step_size, 1),
                "start_index": start_index,
                "n_states": n_cells,
                "n_timesteps_used": stats.get("n_timesteps_used", nlt or 0),
                "bucket_path": f"{prefix}/{folder_name}",
                "raw_config": config,
                "raw_stats": stats,
            },
        )

        # ── Charger la matrice et calculer les diagnostics ──
        matrix_path = f"{folder_name}/transition_matrix.npy"
        try:
            P = self._load_matrix(fs, f"{path}/transition_matrix.npy")
        except Exception:
            self.stdout.write(f"   ⚠️ Matrice manquante: {folder_name}")
            return

        diag = np.diag(P)
        row_sums = P.sum(axis=1)
        col_sums = P.sum(axis=0)
        visited = row_sums > 0

        # Valeurs propres (si demandé ou si pas encore calculé)
        eigenvalue_2 = None
        spectral_gap = None
        if recompute or not TransitionMatrix.objects.filter(experiment=experiment).exists():
            try:
                if P.shape[0] <= 2000:
                    eigenvalues = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
                    if len(eigenvalues) > 1:
                        eigenvalue_2 = float(eigenvalues[1])
                        spectral_gap = float(1 - eigenvalue_2)
            except Exception:
                pass

        TransitionMatrix.objects.update_or_create(
            experiment=experiment,
            defaults={
                "diagonal_mean": float(diag.mean()),
                "diagonal_std": float(diag.std()),
                "diagonal_min": float(diag.min()),
                "diagonal_max": float(diag[diag > 0].max()) if (diag > 0).any() else 0,
                "row_sum_min": float(row_sums[visited].min()) if visited.any() else 0,
                "row_sum_max": float(row_sums[visited].max()) if visited.any() else 0,
                "row_sum_mean": float(row_sums[visited].mean()) if visited.any() else 0,
                "row_sum_std": float(row_sums[visited].std()) if visited.any() else 0,
                "col_sum_min": float(col_sums.min()),
                "col_sum_max": float(col_sums.max()),
                "col_sum_mean": float(col_sums.mean()),
                "n_states_visited": int(visited.sum()),
                "n_states_empty": int((~visited).sum()),
                "fraction_visited": float(visited.sum()) / max(P.shape[0], 1),
                "eigenvalue_2": eigenvalue_2,
                "spectral_gap": spectral_gap,
                "matrix_bucket_path": matrix_path,
            },
        )

        self.stdout.write(
            f"   ✅ {folder_name} | {method_name} | "
            f"{n_cells} cellules | diag={diag.mean():.3f}"
        )