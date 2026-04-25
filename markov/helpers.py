"""
Helper functions pour l'analyse Markov.
Fonctions de calcul matriciel, RSD, visualisation 3D, et utilitaires.
"""

import numpy as np
import json
import traceback
import sys
import logging
from pathlib import Path
from scipy.spatial import Voronoi, ConvexHull

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from partitioners import create_partitioner
    from huggingface_hub import HfFileSystem
    import polars as pl
except ImportError:
    pass


def _compute_matrix_metrics(P):
    """Calcule les métriques complètes de la matrice P."""
    n_states = P.shape[0]

    # Diagonale
    diag = np.diag(P)

    # Sommes des lignes et colonnes
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)

    # États visités
    visited = row_sums > 0
    n_visited = visited.sum()
    fraction_visited = n_visited / n_states if n_states > 0 else 0

    # Éléments hors-diagonale
    off_diagonal = P.copy()
    np.fill_diagonal(off_diagonal, 0)

    metrics = {
        # Diagonale (P rester)
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
        # Sommes des lignes (normalisation)
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_std": float(row_sums.std()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        # Sommes des colonnes
        "col_sum_mean": float(col_sums.mean()),
        "col_sum_std": float(col_sums.std()),
        "col_sum_min": float(col_sums.min()),
        "col_sum_max": float(col_sums.max()),
        # Visitabilité
        "n_visited": int(n_visited),
        "fraction_visited": float(fraction_visited * 100),
        # Hors-diagonale
        "off_diagonal_mean": float(
            off_diagonal[off_diagonal > 0].mean() if (off_diagonal > 0).any() else 0
        ),
        # Déterminant et trace
        "trace": float(np.trace(P)),
        "determinant": float(np.linalg.det(P)),
    }

    # Valeurs propres
    try:
        eigenvalues = np.linalg.eigvals(P)
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        eigenvalues_list = eigenvalues_sorted[
            : min(20, len(eigenvalues_sorted))
        ].tolist()

        metrics["eigenvalues"] = eigenvalues_list
        metrics["eigenvalue_1"] = (
            float(eigenvalues_sorted[0]) if len(eigenvalues_sorted) > 0 else 0
        )
        metrics["eigenvalue_2"] = (
            float(eigenvalues_sorted[1]) if len(eigenvalues_sorted) > 1 else 0
        )
        metrics["spectral_gap"] = (
            float(1.0 - metrics["eigenvalue_2"]) if len(eigenvalues_sorted) > 1 else 0
        )
    except Exception:
        metrics["eigenvalues"] = []
        metrics["eigenvalue_1"] = 0
        metrics["eigenvalue_2"] = 0
        metrics["spectral_gap"] = 0

    return metrics


def _load_matrix_for_experiment(experiment):
    """
    Charge la matrice P et calcule les métriques.

    Utilise les métriques pré-calculées de la base de données si la matrice
    ne peut pas être chargée depuis le bucket HuggingFace.
    """
    try:
        # Essayer de charger la matrice depuis le bucket
        P = experiment.matrix.load_matrix()
        metrics = _compute_matrix_metrics(P)

        diag = np.diag(P)
        row_sums = P.sum(axis=1)
        col_sums = P.sum(axis=0)

        return {
            "matrix": P,
            "diagonal": diag.tolist(),
            "row_sums": row_sums.tolist(),
            "col_sums": col_sums.tolist(),
            "heatmap": P.tolist(),
            "shape": list(P.shape),
            "error": None,
            **metrics,  # Inclure tous les metrics
        }
    except FileNotFoundError as e:
        # Matrice non trouvée dans le bucket - utiliser les métriques pré-calculées
        logger.warning(
            f"Matrice non trouvée pour {experiment.folder_name}: {e}. "
            f"Utilisation des métriques pré-calculées."
        )
        if experiment.matrix:
            return {
                "matrix": None,
                "diagonal": None,
                "row_sums": None,
                "col_sums": None,
                "heatmap": None,
                "shape": [experiment.n_states, experiment.n_states],
                "error": f"Matrice non disponible (sera chargée ultérieurement)",
                "diagonal_mean": experiment.matrix.diagonal_mean,
                "eigenvalue_2": experiment.matrix.eigenvalue_2,
                "spectral_gap": experiment.matrix.spectral_gap,
                "from_cached_metrics": True,
            }
        else:
            return {"error": "Pas de matrice enregistrée pour cette expérience"}
    except Exception as e:
        # Erreur inopinée
        error_msg = f"Erreur lors du chargement de la matrice: {str(e)}"
        logger.error(
            f"{error_msg} ({experiment.folder_name})\n"
            f"Traceback: {traceback.format_exc()}"
        )

        # Fallback: utiliser les métriques pré-calculées si disponibles
        if experiment.matrix:
            return {
                "matrix": None,
                "error": error_msg,
                "diagonal_mean": experiment.matrix.diagonal_mean,
                "eigenvalue_2": experiment.matrix.eigenvalue_2,
                "spectral_gap": experiment.matrix.spectral_gap,
                "from_cached_metrics": True,
            }
        else:
            return {"error": error_msg}


def _compute_rsd_from_matrix(P, n_steps=200, initial_split=0.5, initial_state=None):
    """Calcule le RSD depuis une matrice P (sans MarkovAnalyzer).

    Args:
        P: matrice de transition (n_states x n_states)
        n_steps: nombre de pas de prédiction
        initial_split: fraction de cellules remplies (ignoré si initial_state fourni)
        initial_state: vecteur d'état initial réel (si fourni, utilise celui-ci)
    """
    n_states = P.shape[0]
    if initial_state is not None:
        C = np.array(initial_state, dtype=float)
    else:
        C = np.zeros(n_states)
        mid = int(n_states * initial_split)
        C[:mid] = 1.0

    rsd = np.zeros(n_steps)
    entropy = np.zeros(n_steps)
    concentration_history = np.zeros((n_steps, n_states))

    for t in range(n_steps):
        C = C @ P
        concentration_history[t] = C

        visited = C > 1e-12
        if visited.sum() > 1:
            mean_c = C[visited].mean()
            std_c = C[visited].std()
            rsd[t] = std_c / mean_c if mean_c > 0 else 0

        C_pos = C[C > 1e-12]
        if len(C_pos) > 0 and n_states > 1:
            entropy[t] = -np.sum(C_pos * np.log(C_pos)) / np.log(n_states)

    rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
    t50 = next((t for t in range(n_steps) if rsd[t] < 0.5 * rsd_0), None)
    t90 = next((t for t in range(n_steps) if rsd[t] < 0.1 * rsd_0), None)

    return {
        "rsd_percent": (rsd * 100).tolist(),
        "entropy": entropy.tolist(),
        "concentration_final": concentration_history[-1].tolist(),
        "rsd_initial": float(rsd[0] * 100),
        "rsd_final": float(rsd[-1] * 100),
        "mixing_time_50": t50,
        "mixing_time_90": t90,
    }


def _load_partition_data(
    file_index=100, method="cartesian", n_cells=125, sample_every=1
):
    """Charge les données de partitionnement depuis le bucket."""
    HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))

    if file_index >= len(files):
        raise ValueError(
            f"file_index {file_index} hors limites (max: {len(files) - 1})"
        )

    fname = files[file_index]
    with fs.open(fname, "rb") as f:
        df = pl.read_csv(f)

    coords = np.column_stack(
        [
            df["coordinates:0"].to_numpy(),
            df["coordinates:1"].to_numpy(),
            df["coordinates:2"].to_numpy(),
        ]
    )[::sample_every]

    partitioner_kwargs = {}
    if method == "cartesian":
        side = int(np.ceil(n_cells ** (1 / 3)))
        partitioner_kwargs = {"nx": side, "ny": side, "nz": side}
    elif method == "cylindrical":
        n_radial = int(np.ceil((n_cells / 2) ** 0.5))
        partitioner_kwargs = {
            "nr": n_radial,
            "ntheta": n_radial,
            "nz": 2,
            "radial_mode": "equal_dr",
        }
    elif method == "voronoi":
        partitioner_kwargs = {"n_cells": n_cells}
    elif method == "quantile":
        side = int(np.ceil(n_cells ** (1 / 3)))
        partitioner_kwargs = {"nx": side, "ny": side, "nz": side}
    elif method == "octree":
        partitioner_kwargs = {"max_particles": max(10, n_cells // 4), "max_depth": 4}

    partitioner = create_partitioner(method, **partitioner_kwargs)
    partitioner.fit(coords)

    states = partitioner.compute_states(
        df["coordinates:0"].to_numpy()[::sample_every],
        df["coordinates:1"].to_numpy()[::sample_every],
        df["coordinates:2"].to_numpy()[::sample_every],
    )

    return {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "states": states,
        "n_cells": int(partitioner.n_cells),
        "n_particles": len(coords),
        "method": method,
    }


def _get_partitioner_kwargs(method, n_cells):
    """Retourne les kwargs pour créer un partitionneur."""
    if method == "cartesian":
        side = int(np.ceil(n_cells ** (1 / 3)))
        return {"nx": side, "ny": side, "nz": side}
    elif method == "cylindrical":
        n_radial = int(np.ceil((n_cells / 2) ** 0.5))
        return {"nr": n_radial, "ntheta": n_radial, "nz": 2, "radial_mode": "equal_dr"}
    elif method == "voronoi":
        return {"n_cells": n_cells}
    elif method == "quantile":
        side = int(np.ceil(n_cells ** (1 / 3)))
        return {"nx": side, "ny": side, "nz": side}
    elif method == "octree":
        return {"max_particles": max(10, n_cells // 4), "max_depth": 4}
    return {}


def _compute_partition_boundaries(method, partitioner, payload):
    """Calcule les frontières des partitions pour tous les types."""
    boundaries = []
    coords = np.column_stack([payload["x"], payload["y"], payload["z"]])
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    zmin, zmax = coords[:, 2].min(), coords[:, 2].max()

    if method == "cartesian":
        nx, ny, nz = partitioner.nx, partitioner.ny, partitioner.nz
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        z_edges = np.linspace(zmin, zmax, nz + 1)

        for x in x_edges[1:-1]:
            boundaries.append({"type": "x_plane", "value": float(x)})
        for y in y_edges[1:-1]:
            boundaries.append({"type": "y_plane", "value": float(y)})
        for z in z_edges[1:-1]:
            boundaries.append({"type": "z_plane", "value": float(z)})

    elif method == "quantile":
        # Quantile partitioner already has edges stored
        if hasattr(partitioner, "_x_edges") and partitioner._x_edges is not None:
            for x in partitioner._x_edges[1:-1]:
                boundaries.append({"type": "x_plane", "value": float(x)})
            for y in partitioner._y_edges[1:-1]:
                boundaries.append({"type": "y_plane", "value": float(y)})
            for z in partitioner._z_edges[1:-1]:
                boundaries.append({"type": "z_plane", "value": float(z)})

    elif method == "cylindrical":
        # Cylindrical boundaries: radial circles and z-planes
        if hasattr(partitioner, "_r_edges") and partitioner._r_edges is not None:
            x_center = partitioner._x_center
            y_center = partitioner._y_center

            # Add radial boundaries (concentric circles)
            for r in partitioner._r_edges[1:-1]:
                boundaries.append(
                    {
                        "type": "cylinder",
                        "center_x": float(x_center),
                        "center_y": float(y_center),
                        "radius": float(r),
                        "z_min": float(partitioner._z_min),
                        "z_max": float(partitioner._z_max),
                    }
                )

            # Add angular boundaries (meridian planes)
            theta_step = 2 * np.pi / partitioner.ntheta
            for i in range(1, partitioner.ntheta):
                theta = i * theta_step
                boundaries.append(
                    {
                        "type": "meridian",
                        "center_x": float(x_center),
                        "center_y": float(y_center),
                        "theta": float(theta),
                        "r_max": float(partitioner._r_max),
                    }
                )

            # Add z-planes
            z_edges = np.linspace(
                partitioner._z_min, partitioner._z_max, partitioner.nz + 1
            )
            for z in z_edges[1:-1]:
                boundaries.append({"type": "z_plane", "value": float(z)})

    elif method == "voronoi":
        # Voronoi: store centroids as cell centers
        if hasattr(partitioner, "centroids") and partitioner.centroids is not None:
            for i, centroid in enumerate(partitioner.centroids):
                boundaries.append(
                    {
                        "type": "centroid",
                        "cell_id": int(i),
                        "x": float(centroid[0]),
                        "y": float(centroid[1]),
                        "z": float(centroid[2]),
                    }
                )

    elif method == "octree":
        # Octree: store all leaf box boundaries
        if hasattr(partitioner, "_leaves") and partitioner._leaves:
            for i, (
                xmin_leaf,
                xmax_leaf,
                ymin_leaf,
                ymax_leaf,
                zmin_leaf,
                zmax_leaf,
            ) in enumerate(partitioner._leaves):
                boundaries.append(
                    {
                        "type": "octree_box",
                        "cell_id": int(i),
                        "xmin": float(xmin_leaf),
                        "xmax": float(xmax_leaf),
                        "ymin": float(ymin_leaf),
                        "ymax": float(ymax_leaf),
                        "zmin": float(zmin_leaf),
                        "zmax": float(zmax_leaf),
                    }
                )

    return boundaries


def _sanitize_for_json(obj):
    """Convertit récursivement les types numpy/scipy en types Python natifs pour JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════
# VTK.js 3D MESH BUILDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════


def _build_mixer_cylinder(cx, cy, r, z_min, z_max, n_theta=64):
    """Construit le mélangeur cylindrique (surface semi-transparente)."""
    v, f = _build_vtk_cylinder_surface(cx, cy, r, z_min, z_max, n_theta)
    v_top, f_top = _build_vtk_disk(cx, cy, r, z_max, n_theta)
    v_bot, f_bot = _build_vtk_disk(cx, cy, r, z_min, n_theta)
    # Merge
    all_v = list(v) + list(v_top) + list(v_bot)
    offset1 = len(v)
    offset2 = len(v) + len(v_top)
    all_f = (
        list(f)
        + [[i + offset1 for i in face] for face in f_top]
        + [[i + offset2 for i in face] for face in f_bot]
    )
    return all_v, all_f


def _compute_partition_mesh_from_params(
    method, params, bounds, radial_mode="equal_dr", actual_centroids=None
):
    """
    Calcule les maillages de frontières uniquement depuis les paramètres et les bounds.
    Ajoute aussi le mélangeur cylindrique comme maillage séparé.

    Args:
        radial_mode: "equal_dr" (équirayon) ou "equal_area" (équisurface) pour la méthode cylindrique
        actual_centroids: Centroïdes réels pour Voronoi/Physics (optionnel, corrige l'incohérence)
    """
    if bounds is None:
        bounds = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1, "zmin": 0, "zmax": 4}

    xmin = float(bounds.get("xmin", -1))
    xmax = float(bounds.get("xmax", 1))
    ymin = float(bounds.get("ymin", -1))
    ymax = float(bounds.get("ymax", 1))
    zmin = float(bounds.get("zmin", 0))
    zmax = float(bounds.get("zmax", 4))

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    r_mixer = min(xmax - xmin, ymax - ymin) / 2

    mesh_groups = []

    # ── Mélangeur cylindrique (toujours visible) ──
    mv, mf = _build_mixer_cylinder(cx, cy, r_mixer, zmin, zmax)
    if mv:
        mesh_groups.append(
            {
                "label": "Mélangeur",
                "vertices": mv,
                "faces": mf,
                "opacity": 0.08,
                "color": [0.7, 0.85, 1.0],
            }
        )

    if method == "cartesian":
        nx = int(params.get("nx", 5))
        ny = int(params.get("ny", 5))
        nz = int(params.get("nz", 5))
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        z_edges = np.linspace(zmin, zmax, nz + 1)
        all_v, all_f = [], []
        offset = 0
        for x in x_edges[1:-1]:
            v = [
                [float(x), ymin, zmin],
                [float(x), ymax, zmin],
                [float(x), ymax, zmax],
                [float(x), ymin, zmax],
            ]
            all_v.extend(v)
            all_f.append([offset, offset + 1, offset + 2, offset + 3])
            offset += 4
        for y in y_edges[1:-1]:
            v = [
                [xmin, float(y), zmin],
                [xmax, float(y), zmin],
                [xmax, float(y), zmax],
                [xmin, float(y), zmax],
            ]
            all_v.extend(v)
            all_f.append([offset, offset + 1, offset + 2, offset + 3])
            offset += 4
        for z in z_edges[1:-1]:
            v = [
                [xmin, ymin, float(z)],
                [xmax, ymin, float(z)],
                [xmax, ymax, float(z)],
                [xmin, ymax, float(z)],
            ]
            all_v.extend(v)
            all_f.append([offset, offset + 1, offset + 2, offset + 3])
            offset += 4
        if all_v:
            mesh_groups.append(
                {
                    "label": "Planes de division",
                    "vertices": all_v,
                    "faces": all_f,
                    "opacity": 0.15,
                    "color": [0.3, 0.5, 1.0],
                }
            )

    elif method == "cylindrical":
        nr = int(params.get("nr", 3))
        ntheta = int(params.get("ntheta", 4))
        nz = int(params.get("nz", 2))
        # Calcul des rayons selon le mode radial
        if radial_mode == "equal_area":
            # Équisurface : r_i = R × √(i/nr) pour aire π(r_{i+1}² - r_i²) constante
            r_edges = r_mixer * np.sqrt(np.linspace(0, 1, nr + 1))
        else:  # equal_dr (défaut)
            # Équirayon : Δr constant
            r_edges = np.linspace(0, r_mixer, nr + 1)
        for r in r_edges[1:-1]:
            v, f = _build_vtk_cylinder_surface(cx, cy, float(r), zmin, zmax)
            if v:
                mesh_groups.append(
                    {
                        "label": f"r={float(r):.2f}",
                        "vertices": v,
                        "faces": f,
                        "opacity": 0.15,
                        "color": [0.2, 0.7, 0.4],
                    }
                )
        theta_step = 2 * np.pi / ntheta
        for i in range(1, ntheta):
            theta = i * theta_step
            v, f = _build_vtk_meridian_plane(cx, cy, theta, r_mixer, zmin, zmax)
            if v:
                mesh_groups.append(
                    {
                        "label": f"θ={float(np.degrees(theta)):.0f}°",
                        "vertices": v,
                        "faces": f,
                        "opacity": 0.10,
                        "color": [0.9, 0.5, 0.2],
                    }
                )
        z_edges = np.linspace(zmin, zmax, nz + 1)
        for z in z_edges[1:-1]:
            v, f = _build_vtk_disk(cx, cy, r_mixer, float(z))
            if v:
                mesh_groups.append(
                    {
                        "label": f"z={float(z):.2f}",
                        "vertices": v,
                        "faces": f,
                        "opacity": 0.10,
                        "color": [0.6, 0.3, 0.8],
                    }
                )

    elif method == "voronoi" or method == "physics":
        n_cells = int(params.get("n_cells", 125))
        rs = int(params.get("random_state", 42))
        # Utiliser les centroïdes réels si fournis, sinon générer aléatoirement
        if actual_centroids is not None and len(actual_centroids) > 0:
            centroids = np.array(actual_centroids)
        else:
            rng = np.random.RandomState(rs)
            centroids = rng.uniform(
                [xmin, ymin, zmin], [xmax, ymax, zmax], (n_cells, 3)
            )

        # Limites cylindriques du mélangeur (cohérent avec la visualisation du cylindre)
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        radius = min(xmax - xmin, ymax - ymin) / 2
        z_min_cyl = zmin
        z_max_cyl = zmax

        v, f = _build_vtk_voronoi_cells(
            centroids,
            (xmin, xmax, ymin, ymax, zmin, zmax),
            cylindrical_bounds=(x_center, y_center, radius, z_min_cyl, z_max_cyl),
        )
        if v:
            mesh_groups.append(
                {
                    "label": "Cellules Voronoï"
                    if method == "voronoi"
                    else "Cellules Physics",
                    "vertices": v,
                    "faces": f,
                    "opacity": 0.08,
                    "color": [0.5, 0.5, 0.9],
                }
            )

            # Ajouter les centroïdes comme points explicites pour meilleure visibilité
            centroid_v, centroid_f = _build_centroid_points(centroids)
            if centroid_v:
                mesh_groups.append(
                    {
                        "label": "Centroïdes",
                        "vertices": centroid_v,
                        "faces": centroid_f,
                        "opacity": 1.0,
                        "color": [1.0, 0.0, 0.0],  # Rouge vif
                    }
                )

    elif method == "quantile":
        nx = int(params.get("nx", 5))
        ny = int(params.get("ny", 5))
        nz = int(params.get("nz", 5))
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        z_edges = np.linspace(zmin, zmax, nz + 1)
        all_v, all_f = [], []
        offset = 0
        for x in x_edges[1:-1]:
            v = [
                [float(x), ymin, zmin],
                [float(x), ymax, zmin],
                [float(x), ymax, zmax],
                [float(x), ymin, zmax],
            ]
            all_v.extend(v)
            all_f.append([offset, offset + 1, offset + 2, offset + 3])
            offset += 4
        for y in y_edges[1:-1]:
            v = [
                [xmin, float(y), zmin],
                [xmax, float(y), zmin],
                [xmax, float(y), zmax],
                [xmin, float(y), zmax],
            ]
            all_v.extend(v)
            all_f.append([offset, offset + 1, offset + 2, offset + 3])
            offset += 4
        for z in z_edges[1:-1]:
            v = [
                [xmin, ymin, float(z)],
                [xmax, ymin, float(z)],
                [xmax, ymax, float(z)],
                [xmin, ymax, float(z)],
            ]
            all_v.extend(v)
            all_f.append([offset, offset + 1, offset + 2, offset + 3])
            offset += 4
        if all_v:
            mesh_groups.append(
                {
                    "label": "Planes de quantile",
                    "vertices": all_v,
                    "faces": all_f,
                    "opacity": 0.15,
                    "color": [0.8, 0.4, 0.2],
                }
            )

    elif method == "octree":
        max_depth = int(params.get("max_depth", 4))
        n_div = 2 ** max(1, min(max_depth, 4))
        dx = (xmax - xmin) / n_div
        dy = (ymax - ymin) / n_div
        dz = (zmax - zmin) / n_div
        all_v, all_f = [], []
        offset = 0
        for ix in range(n_div):
            for iy in range(n_div):
                for iz in range(n_div):
                    bx = xmin + ix * dx
                    by = ymin + iy * dy
                    bz = zmin + iz * dz
                    bv, bf = _build_vtk_box(bx, bx + dx, by, by + dy, bz, bz + dz)
                    all_v.extend(bv)
                    all_f.extend([[int(idx) + offset for idx in face] for face in bf])
                    offset += len(bv)
        if all_v:
            mesh_groups.append(
                {
                    "label": "Boîtes Octree",
                    "vertices": all_v,
                    "faces": all_f,
                    "opacity": 0.08,
                    "color": [0.3, 0.8, 0.8],
                }
            )

    # Boîte englobante
    v, f = _build_vtk_box(xmin, xmax, ymin, ymax, zmin, zmax)
    mesh_groups.append(
        {
            "label": "Boîte englobante",
            "vertices": v,
            "faces": f,
            "opacity": 0.05,
            "color": [0.5, 0.5, 0.5],
        }
    )

    return mesh_groups


def _build_vtk_box(xmin, xmax, ymin, ymax, zmin, zmax):
    """Construit les 8 sommets et 6 faces d'un box pour vtk.js."""
    vertices = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ]
    faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ]
    return vertices, faces


def _build_vtk_cylinder_surface(cx, cy, r, z_min, z_max, n_theta=64):
    """Construit la surface latérale d'un cylindre pour vtk.js."""
    vertices = []
    faces = []
    for i in range(n_theta):
        theta = 2 * np.pi * i / n_theta
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        vertices.append([x, y, z_min])
        vertices.append([x, y, z_max])

    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        v0 = i * 2
        v1 = i * 2 + 1
        v2 = i_next * 2 + 1
        v3 = i_next * 2
        faces.append([v0, v1, v2, v3])
    return vertices, faces


def _build_vtk_disk(cx, cy, r, z, n_theta=64):
    """Construit un disque (top/bottom) pour un cylindre."""
    vertices = [[cx, cy, z]]
    for i in range(n_theta):
        theta = 2 * np.pi * i / n_theta
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        vertices.append([x, y, z])

    faces = []
    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        faces.append([0, i + 1, i_next + 1])
    return vertices, faces


def _build_vtk_meridian_plane(cx, cy, theta, r_max, z_min, z_max, n_r=16):
    """Construit un plan méridien (secteur angulaire)."""
    vertices = []
    faces = []
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for j in range(n_r + 1):
        r = r_max * j / n_r
        x = cx + r * cos_t
        y = cy + r * sin_t
        vertices.append([x, y, z_min])
        vertices.append([x, y, z_max])

    for j in range(n_r):
        v0 = j * 2
        v1 = j * 2 + 1
        v2 = (j + 1) * 2 + 1
        v3 = (j + 1) * 2
        faces.append([v0, v1, v2, v3])
    return vertices, faces


def _build_vtk_voronoi_cells(centroids, bounds, n_subdiv=8, cylindrical_bounds=None):
    """
    Approximation des cellules de Voronoï par des boîtes autour des centroïdes.

    Args:
        centroids: array-like, shape (n, 3) - les centroïdes Voronoï
        bounds: tuple (xmin, xmax, ymin, ymax, zmin, zmax) - limites de la boîte
        n_subdiv: int - subdivisions (non utilisé, gardé pour compatibilité)
        cylindrical_bounds: tuple (cx, cy, radius, z_min, z_max) ou None -
                          si fourni, clippe les cellules au cylindre physique
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    try:
        # Ajouter des points fictifs pour borner le Voronoi
        # Augmenter le padding si clipping cylindrique
        padding = 5.0 if cylindrical_bounds is None else 2.0
        extra_points = [
            [xmin - padding, ymin - padding, zmin - padding],
            [xmax + padding, ymin - padding, zmin - padding],
            [xmin - padding, ymax + padding, zmin - padding],
            [xmax + padding, ymax + padding, zmin - padding],
            [xmin - padding, ymin - padding, zmax + padding],
            [xmax + padding, ymin - padding, zmax + padding],
            [xmin - padding, ymax + padding, zmax + padding],
            [xmax + padding, ymax + padding, zmax + padding],
        ]
        all_points = np.vstack([centroids, np.array(extra_points)])
        vor = Voronoi(all_points)

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        n_real = len(centroids)
        for i in range(n_real):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                # Fallback: small box around centroid
                c = centroids[i]
                s = 0.02 * max(xmax - xmin, ymax - ymin, zmax - zmin)
                v, f = _build_vtk_box(
                    c[0] - s, c[0] + s, c[1] - s, c[1] + s, c[2] - s, c[2] + s
                )
            else:
                region_verts = vor.vertices[region]
                # Clip to axis-aligned bounds
                region_verts[:, 0] = np.clip(region_verts[:, 0], xmin, xmax)
                region_verts[:, 1] = np.clip(region_verts[:, 1], ymin, ymax)
                region_verts[:, 2] = np.clip(region_verts[:, 2], zmin, zmax)

                # Clipping supplémentaire au cylindre physique si demandé
                if cylindrical_bounds is not None:
                    cx, cy, radius, z_min_cyl, z_max_cyl = cylindrical_bounds
                    # Contraindre au cylindre: (x-cx)^2 + (y-cy)^2 <= radius^2
                    # et z_min_cyl <= z <= z_max_cyl
                    for _ in range(3):  # Plusieurs itérations pour convergence
                        dx = region_verts[:, 0] - cx
                        dy = region_verts[:, 1] - cy
                        radial_dist_sq = dx**2 + dy**2

                        # Trouver les points à l'intérieur du cylindre
                        inside_cylinder = (
                            (radial_dist_sq <= radius**2)
                            & (region_verts[:, 2] >= z_min_cyl)
                            & (region_verts[:, 2] <= z_max_cyl)
                        )

                        if np.all(inside_cylinder) or len(region_verts) < 3:
                            break

                        # Pour les points à l'extérieur, les projeter vers l'intérieur
                        outside = ~inside_cylinder
                        if np.any(outside):
                            # Projection des points extérieurs sur la surface cylindrique
                            for j in np.where(outside)[0]:
                                x, y, z = region_verts[j]

                                # Vérifier et corriger Z si nécessaire
                                if z < z_min_cyl:
                                    z = z_min_cyl
                                elif z > z_max_cyl:
                                    z = z_max_cyl

                                # Projeter X,Y sur le cercle
                                dx = x - cx
                                dy = y - cy
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist > 0 and dist > radius:
                                    scale = radius / dist
                                    x = cx + dx * scale
                                    y = cy + dy * scale

                                region_verts[j] = [x, y, z]

                v = region_verts.tolist()
                # Triangulate the convex polyhedron faces
                f = _triangulate_convex_polyhedron(v)

            all_vertices.extend(v)
            all_faces.extend([[idx + vertex_offset for idx in face] for face in f])
            vertex_offset += len(v)

        return all_vertices, all_faces
    except Exception:
        # Fallback: boxes around centroids
        all_vertices = []
        all_faces = []
        offset = 0
        for c in centroids:
            s = 0.03 * max(xmax - xmin, ymax - ymin, zmax - zmin)
            v, f = _build_vtk_box(
                c[0] - s, c[0] + s, c[1] - s, c[1] + s, c[2] - s, c[2] + s
            )
            all_vertices.extend(v)
            all_faces.extend([[idx + offset for idx in face] for face in f])
            offset += len(v)
        return all_vertices, all_faces


def _build_centroid_points(centroids):
    """
    Construit des points représentant les centroïdes Voronoï pour VTK.js.
    Crée de très petites boîtes (points) à chaque position de centroïde.
    """
    if centroids is None or len(centroids) == 0:
        return [], []

    all_vertices = []
    all_faces = []
    offset = 0

    # Taille des points (relativement petite mais visible)
    point_size = 0.02  # 2% de la taille typique du domaine

    for centroid in centroids:
        x, y, z = centroid[0], centroid[1], centroid[2]
        # Créer une petite boîte centrée sur le centroïde
        v, f = _build_vtk_box(
            x - point_size,
            x + point_size,
            y - point_size,
            y + point_size,
            z - point_size,
            z + point_size,
        )
        all_vertices.extend(v)
        all_faces.extend([[idx + offset for idx in face] for face in f])
        offset += len(v)

    return all_vertices, all_faces


def _triangulate_convex_polyhedron(vertices):
    """Triangule un polyèdre convexe (fan depuis le centroïde)."""
    if len(vertices) < 4:
        return []
    verts = np.array(vertices)
    center = verts.mean(axis=0)

    # Compute convex hull faces
    try:
        hull = ConvexHull(verts)
        faces = [list(simplex) for simplex in hull.simplices]
        return faces
    except Exception:
        # Simple fan triangulation
        faces = []
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            faces.append([0, i, j])
        return faces


def _build_vtk_octree_boxes(leaves, bounds):
    """Construit les wireframes des boîtes octree."""
    all_vertices = []
    all_faces = []
    offset = 0

    for xmin_l, xmax_l, ymin_l, ymax_l, zmin_l, zmax_l in leaves:
        v, f = _build_vtk_box(xmin_l, xmax_l, ymin_l, ymax_l, zmin_l, zmax_l)
        all_vertices.extend(v)
        all_faces.extend([[idx + offset for idx in face] for face in f])
        offset += len(v)
    return all_vertices, all_faces


def _compute_partition_mesh_vtk(method, partitioner, coords):
    """
    Calcule les maillages VTK (vertices + faces) pour les frontières de partition.
    Retourne une liste de groupes de maillages avec labels.
    """
    mesh_groups = []
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    zmin, zmax = float(coords[:, 2].min()), float(coords[:, 2].max())

    if method == "cartesian":
        nx, ny, nz = partitioner.nx, partitioner.ny, partitioner.nz
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        z_edges = np.linspace(zmin, zmax, nz + 1)

        # Internal division planes
        all_v, all_f = [], []
        offset = 0

        # X-planes
        for x in x_edges[1:-1]:
            v = [[x, ymin, zmin], [x, ymax, zmin], [x, ymax, zmax], [x, ymin, zmax]]
            f = [[0, 1, 2, 3]]
            all_v.extend(v)
            all_f.extend([[i + offset for i in face] for face in f])
            offset += len(v)

        # Y-planes
        for y in y_edges[1:-1]:
            v = [[xmin, y, zmin], [xmax, y, zmin], [xmax, y, zmax], [xmin, y, zmax]]
            f = [[0, 1, 2, 3]]
            all_v.extend(v)
            all_f.extend([[i + offset for i in face] for face in f])
            offset += len(v)

        # Z-planes
        for z in z_edges[1:-1]:
            v = [[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]]
            f = [[0, 1, 2, 3]]
            all_v.extend(v)
            all_f.extend([[i + offset for i in face] for face in f])
            offset += len(v)

        if all_v:
            mesh_groups.append(
                {
                    "label": "Planes de division",
                    "vertices": all_v,
                    "faces": all_f,
                    "opacity": 0.15,
                    "color": [0.3, 0.5, 1.0],
                }
            )

    elif method == "cylindrical":
        cx = float(getattr(partitioner, "_x_center", (xmin + xmax) / 2))
        cy = float(getattr(partitioner, "_y_center", (ymin + ymax) / 2))
        z_min_p = float(getattr(partitioner, "_z_min", zmin))
        z_max_p = float(getattr(partitioner, "_z_max", zmax))

        if hasattr(partitioner, "_r_edges") and partitioner._r_edges is not None:
            # Radial boundaries (concentric cylinders)
            for r in partitioner._r_edges[1:-1]:
                v, f = _build_vtk_cylinder_surface(cx, cy, float(r), z_min_p, z_max_p)
                if v:
                    mesh_groups.append(
                        {
                            "label": f"r={float(r):.2f}",
                            "vertices": v,
                            "faces": f,
                            "opacity": 0.12,
                            "color": [0.2, 0.7, 0.4],
                        }
                    )

            # Top and bottom disks for outer radius
            r_max = (
                float(partitioner._r_max)
                if hasattr(partitioner, "_r_max")
                else float(partitioner._r_edges[-1])
            )
            v_top, f_top = _build_vtk_disk(cx, cy, r_max, z_max_p)
            v_bot, f_bot = _build_vtk_disk(cx, cy, r_max, z_min_p)
            if v_top:
                mesh_groups.append(
                    {
                        "label": "Disque haut",
                        "vertices": v_top,
                        "faces": f_top,
                        "opacity": 0.10,
                        "color": [0.8, 0.8, 0.8],
                    }
                )
                mesh_groups.append(
                    {
                        "label": "Disque bas",
                        "vertices": v_bot,
                        "faces": f_bot,
                        "opacity": 0.10,
                        "color": [0.8, 0.8, 0.8],
                    }
                )

            # Angular boundaries (meridian planes)
            theta_step = 2 * np.pi / partitioner.ntheta
            for i in range(1, partitioner.ntheta):
                theta = i * theta_step
                v, f = _build_vtk_meridian_plane(cx, cy, theta, r_max, z_min_p, z_max_p)
                if v:
                    mesh_groups.append(
                        {
                            "label": f"theta={np.degrees(theta):.0f}°",
                            "vertices": v,
                            "faces": f,
                            "opacity": 0.10,
                            "color": [0.9, 0.5, 0.2],
                        }
                    )

        # Z-planes
        if hasattr(partitioner, "nz"):
            z_edges = np.linspace(z_min_p, z_max_p, partitioner.nz + 1)
            r_max = float(
                getattr(
                    partitioner,
                    "_r_max",
                    partitioner._r_edges[-1]
                    if hasattr(partitioner, "_r_edges")
                    and partitioner._r_edges is not None
                    else (xmax - xmin) / 2,
                )
            )
            for z in z_edges[1:-1]:
                v, f = _build_vtk_disk(cx, cy, r_max, float(z))
                if v:
                    mesh_groups.append(
                        {
                            "label": f"z={float(z):.2f}",
                            "vertices": v,
                            "faces": f,
                            "opacity": 0.10,
                            "color": [0.6, 0.3, 0.8],
                        }
                    )

    elif method == "voronoi":
        if hasattr(partitioner, "centroids") and partitioner.centroids is not None:
            centroids = np.array(partitioner.centroids)
            bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
            v, f = _build_vtk_voronoi_cells(centroids, bounds)
            if v:
                mesh_groups.append(
                    {
                        "label": "Cellules Voronoï",
                        "vertices": v,
                        "faces": f,
                        "opacity": 0.08,
                        "color": [0.5, 0.5, 0.9],
                    }
                )

    elif method == "quantile":
        if hasattr(partitioner, "_x_edges") and partitioner._x_edges is not None:
            all_v, all_f = [], []
            offset = 0
            for x in partitioner._x_edges[1:-1]:
                v = [[x, ymin, zmin], [x, ymax, zmin], [x, ymax, zmax], [x, ymin, zmax]]
                all_v.extend(v)
                all_f.extend([[0 + offset, 1 + offset, 2 + offset, 3 + offset]])
                offset += 4
            for y in partitioner._y_edges[1:-1]:
                v = [[xmin, y, zmin], [xmax, y, zmin], [xmax, y, zmax], [xmin, y, zmax]]
                all_v.extend(v)
                all_f.extend([[0 + offset, 1 + offset, 2 + offset, 3 + offset]])
                offset += 4
            for z in partitioner._z_edges[1:-1]:
                v = [[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]]
                all_v.extend(v)
                all_f.extend([[0 + offset, 1 + offset, 2 + offset, 3 + offset]])
                offset += 4
            if all_v:
                mesh_groups.append(
                    {
                        "label": "Planes de quantile",
                        "vertices": all_v,
                        "faces": all_f,
                        "opacity": 0.15,
                        "color": [0.8, 0.4, 0.2],
                    }
                )

    elif method == "octree":
        if hasattr(partitioner, "_leaves") and partitioner._leaves:
            v, f = _build_vtk_octree_boxes(
                partitioner._leaves, (xmin, xmax, ymin, ymax, zmin, zmax)
            )
            if v:
                mesh_groups.append(
                    {
                        "label": "Boîtes Octree",
                        "vertices": v,
                        "faces": f,
                        "opacity": 0.12,
                        "color": [0.3, 0.8, 0.8],
                    }
                )

    # Bounding box
    v, f = _build_vtk_box(xmin, xmax, ymin, ymax, zmin, zmax)
    mesh_groups.append(
        {
            "label": "Boîte englobante",
            "vertices": v,
            "faces": f,
            "opacity": 0.05,
            "color": [0.5, 0.5, 0.5],
        }
    )

    return mesh_groups


def _detect_varying_params(exp_list):
    """Détecte quels paramètres varient entre les expériences sélectionnées."""
    if len(exp_list) <= 1:
        return ["method"]

    varying = []
    # Check each parameter
    checks = {
        "method": lambda d: d["method"],
        "n_states": lambda d: d["n_states"],
        "nlt": lambda d: d["nlt"],
        "step_size": lambda d: d["step_size"],
        "start_index": lambda d: d["start_index"],
    }

    for param, getter in checks.items():
        values = set(getter(d) for d in exp_list)
        if len(values) > 1:
            varying.append(param)

    # Also check nested parameters
    all_params = set()
    for d in exp_list:
        all_params.update(d["parameters"].keys())
    for p in sorted(all_params):
        values = set()
        for d in exp_list:
            v = d["parameters"].get(p)
            if v is not None:
                values.add(str(v))
        if len(values) > 1:
            varying.append(p)

    return varying or ["label"]


def _make_smart_label(exp, varying_params):
    """Crée un label court basé sur les paramètres qui varient."""
    parts = []
    for p in varying_params:
        if p == "method":
            parts.append(exp["method"])
        elif p == "n_states":
            parts.append(f"{exp['n_states']}c")
        elif p == "nlt":
            parts.append(f"nlt={exp['nlt']}")
        elif p == "step_size":
            parts.append(f"step={exp['step_size']}")
        elif p == "start_index":
            parts.append(f"start={exp['start_index']}")
        elif p in exp["parameters"]:
            parts.append(f"{p}={exp['parameters'][p]}")
    if not parts:
        parts.append(exp["label"][:20])
    return " | ".join(parts[:3])  # Max 3 labels
