"""
Modèles de données pour les expériences Markoviennes.

Hiérarchie:
    PartitionMethod → Experiment → TransitionMatrix
                                 → RSDResult
                                 → MatrixDiagnostics
"""

import numpy as np
import json
from django.db import models
from django.core.validators import MinValueValidator


class PartitionMethod(models.Model):
    """
    Type de partitionnement spatial.
    Ex: cartesian, cylindrical, voronoi, quantile, octree, physics
    """
    METHODS = [
        ("cartesian", "Cartésien"),
        ("cylindrical", "Cylindrique"),
        ("voronoi", "Voronoï (K-means)"),
        ("quantile", "Quantile (équi-pop)"),
        ("octree", "Octree adaptatif"),
        ("physics", "Physics-aware"),
    ]

    name = models.CharField(max_length=50, choices=METHODS, db_index=True)
    description = models.TextField(blank=True)

    # Paramètres spécifiques (JSON)
    parameters = models.JSONField(
        default=dict,
        help_text="Paramètres du partitionneur (nx, ny, nz, nr, ntheta, n_cells, ...)"
    )

    # Label unique (ex: "voronoi_125cells", "cartesian_nx5_ny5_nz5")
    label = models.CharField(max_length=200, unique=True, db_index=True)

    # Métriques de qualité du partitionnement
    n_cells = models.IntegerField(default=0)
    n_cells_visited = models.IntegerField(default=0, blank=True)
    population_mean = models.FloatField(null=True, blank=True)
    population_std = models.FloatField(null=True, blank=True)
    population_cv = models.FloatField(
        null=True, blank=True,
        help_text="Coefficient de variation = std/mean (plus petit = mieux)"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name", "n_cells"]
        verbose_name = "Méthode de partitionnement"

    def __str__(self):
        return f"{self.get_name_display()} — {self.label} ({self.n_cells} cellules)"

    @property
    def fill_ratio(self):
        if self.n_cells > 0:
            return self.n_cells_visited / self.n_cells
        return 0


class Experiment(models.Model):
    """
    Une expérience = un jeu de paramètres + une matrice de transition.
    """
    # ── Identifiant ──
    folder_name = models.CharField(
        max_length=300, unique=True, db_index=True,
        help_text="Nom du dossier dans le bucket HF"
    )

    # ── Partitionnement ──
    partition_method = models.ForeignKey(
        PartitionMethod, on_delete=models.CASCADE,
        related_name="experiments"
    )

    # ── Hyperparamètres d'apprentissage ──
    nlt = models.IntegerField(
        "Learning Time (NLT)",
        validators=[MinValueValidator(1)],
        help_text="Nombre de paires de snapshots pour l'apprentissage"
    )
    step_size = models.IntegerField(
        "Pas temporel (step)",
        default=1,
        validators=[MinValueValidator(1)],
        help_text="Intervalle entre deux snapshots d'une paire"
    )
    start_index = models.IntegerField(
        "Index de départ",
        default=0,
        help_text="Premier fichier DEM utilisé"
    )

    # ── Métadonnées ──
    n_states = models.IntegerField("Nombre d'états", default=0)
    n_timesteps_used = models.IntegerField("Timesteps effectifs", default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    bucket_path = models.CharField(max_length=500, blank=True)

    # ── Configuration complète (JSON brut) ──
    raw_config = models.JSONField(default=dict, blank=True)
    raw_stats = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Expérience"
        indexes = [
            models.Index(fields=["nlt"]),
            models.Index(fields=["step_size"]),
            models.Index(fields=["start_index"]),
            models.Index(fields=["n_states"]),
        ]

    def __str__(self):
        return (
            f"{self.partition_method.label} | "
            f"NLT={self.nlt} step={self.step_size} start={self.start_index}"
        )

    @property
    def method_name(self):
        return self.partition_method.name


class TransitionMatrix(models.Model):
    """
    Matrice de transition P et ses diagnostics.
    La matrice elle-même est stockée dans le bucket (trop grande pour la DB).
    """
    experiment = models.OneToOneField(
        Experiment, on_delete=models.CASCADE,
        related_name="matrix"
    )

    # ── Diagnostics de la matrice ──
    diagonal_mean = models.FloatField(
        "P(rester) moyen",
        help_text="Moyenne de la diagonale de P"
    )
    diagonal_std = models.FloatField("σ diagonale", default=0)
    diagonal_min = models.FloatField("Min diagonale", default=0)
    diagonal_max = models.FloatField("Max diagonale", default=0)

    row_sum_min = models.FloatField("Min Σ lignes", default=0)
    row_sum_max = models.FloatField("Max Σ lignes", default=0)
    row_sum_mean = models.FloatField("Moy Σ lignes", default=0)
    row_sum_std = models.FloatField("σ Σ lignes", default=0)

    col_sum_min = models.FloatField("Min Σ colonnes", default=0)
    col_sum_max = models.FloatField("Max Σ colonnes", default=0)
    col_sum_mean = models.FloatField("Moy Σ colonnes", default=0)

    n_states_visited = models.IntegerField("États visités", default=0)
    n_states_empty = models.IntegerField("États vides", default=0)
    fraction_visited = models.FloatField("Fraction visitée", default=0)

    # ── Spectre ──
    eigenvalue_2 = models.FloatField(
        "|λ₂|", null=True, blank=True,
        help_text="2ème plus grande valeur propre (vitesse de mélange)"
    )
    spectral_gap = models.FloatField(
        "Gap spectral", null=True, blank=True,
        help_text="1 - |λ₂| (plus grand = mélange plus rapide)"
    )

    # ── Stockage ──
    matrix_bucket_path = models.CharField(
        max_length=500,
        help_text="Chemin vers transition_matrix.npy dans le bucket"
    )

    class Meta:
        verbose_name = "Matrice de transition"

    def __str__(self):
        return f"P de {self.experiment} | diag={self.diagonal_mean:.3f}"

    def load_matrix(self):
        """Charge la matrice depuis le bucket."""
        from bucket_io import load_matrix_from_bucket
        return load_matrix_from_bucket(self.matrix_bucket_path)


class RSDResult(models.Model):
    """
    Résultat de simulation RSD pour une expérience.
    Peut être DEM (réel) ou Markov (prédit).
    """
    SOURCE_CHOICES = [
        ("dem", "DEM (réel)"),
        ("markov", "Markov (prédit)"),
    ]

    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE,
        related_name="rsd_results"
    )

    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, db_index=True)
    species_criterion = models.CharField(
        max_length=50, default="z_median",
        help_text="Critère de séparation des espèces"
    )
    n_steps = models.IntegerField("Pas de simulation", default=200)

    # ── Métriques ──
    rsd_initial = models.FloatField("RSD initial (%)")
    rsd_final = models.FloatField("RSD final (%)")
    mixing_time_50 = models.IntegerField("t₅₀ (RSD÷2)", null=True, blank=True)
    mixing_time_90 = models.IntegerField("t₉₀ (RSD÷10)", null=True, blank=True)
    entropy_final = models.FloatField("Entropie finale", null=True, blank=True)

    # ── Courbes (JSON arrays) ──
    rsd_curve = models.JSONField(
        default=list, blank=True,
        help_text="RSD(t) en pourcentage"
    )
    entropy_curve = models.JSONField(default=list, blank=True)
    concentration_final = models.JSONField(
        default=list, blank=True,
        help_text="Concentration par cellule au dernier pas"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Résultat RSD"
        unique_together = ["experiment", "source", "species_criterion"]

    def __str__(self):
        return (
            f"RSD {self.get_source_display()} — {self.experiment} | "
            f"{self.rsd_initial:.1f}% → {self.rsd_final:.1f}%"
        )


class DEMSnapshot(models.Model):
    """
    Référence vers un snapshot DEM (fichier CSV dans le bucket).
    """
    file_index = models.IntegerField(unique=True, db_index=True)
    filename = models.CharField(max_length=200)
    n_particles = models.IntegerField(default=0)
    timestamp = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["file_index"]
        verbose_name = "Snapshot DEM"

    def __str__(self):
        return f"t={self.file_index} | {self.n_particles} particules"