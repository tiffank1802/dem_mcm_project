"""
API JSON Endpoints for Markov analysis.
Handles matrix data, RSD analysis, partitioning, and 3D visualization.
"""

import numpy as np
import json
import traceback
import sys
import logging
from pathlib import Path
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Avg, Min, Max, Count, Q, F

from .models import PartitionMethod, Experiment, TransitionMatrix, RSDResult
from .helpers import (
    _compute_matrix_metrics,
    _load_matrix_for_experiment,
    _compute_rsd_from_matrix,
    _load_partition_data,
    _get_partitioner_kwargs,
    _compute_partition_boundaries,
    _sanitize_for_json,
    _compute_partition_mesh_from_params,
    _detect_varying_params,
    _make_smart_label,
)

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


# ═══════════════════════════════════════════════════════════════
# API JSON
# ═══════════════════════════════════════════════════════════════


def api_matrix_data(request, pk):
    """Données de la matrice pour Plotly."""
    exp = get_object_or_404(Experiment, pk=pk)
    md = _load_matrix_for_experiment(exp)

    # Même si la matrice complète n'est pas disponible, retourner les métriques pré-calculées
    if md.get("error") and not md.get("from_cached_metrics"):
        return JsonResponse({"error": md["error"]}, status=500)

    return JsonResponse({k: v for k, v in md.items() if k != "matrix"})


def api_rsd_data(request, pk):
    """Données RSD pour Plotly."""
    exp = get_object_or_404(Experiment, pk=pk)
    n_steps = int(request.GET.get("n_steps", 200))

    md = _load_matrix_for_experiment(exp)

    # Si la matrice n'est pas disponible, retourner un message d'erreur clair
    if md.get("matrix") is None:
        return JsonResponse(
            {
                "error": "Matrice non disponible pour cette expérience",
                "message": "Les données RSD nécessitent la matrice de transition complète",
            },
            status=503,
        )

    rsd = _compute_rsd_from_matrix(md["matrix"], n_steps)
    return JsonResponse(rsd)


def api_experiment_stats(request):
    """Stats agrégées pour les graphiques."""
    method = request.GET.get("method")
    qs = TransitionMatrix.objects.select_related("experiment__partition_method")
    if method:
        qs = qs.filter(experiment__partition_method__name=method)

    data = list(
        qs.values(
            "experiment__folder_name",
            "experiment__partition_method__name",
            "experiment__nlt",
            "experiment__step_size",
            "experiment__n_states",
            "diagonal_mean",
            "row_sum_mean",
            "eigenvalue_2",
            "fraction_visited",
        )
    )
    return JsonResponse({"data": data})


def api_compare_rsd(request):
    """Compare le RSD de plusieurs expériences."""
    ids = request.GET.getlist("ids")
    n_steps = int(request.GET.get("n_steps", 200))

    results = []
    for pk in ids:
        try:
            exp = Experiment.objects.select_related("partition_method", "matrix").get(
                pk=pk
            )
            md = _load_matrix_for_experiment(exp)
            if md.get("matrix") is not None:
                rsd = _compute_rsd_from_matrix(md["matrix"], n_steps)
                results.append(
                    {
                        "name": exp.folder_name[:40],
                        "method": exp.partition_method.name,
                        "n_states": exp.n_states,
                        **rsd,
                    }
                )
        except Exception:
            pass

    return JsonResponse({"results": results})


def _load_partition_data(
    file_index=100, method="cartesian", n_cells=125, sample_every=1
):
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


def api_partitions(request):
    """
    API pour charger les positions des particules et les assigner à des cellules.

    Query parameters:
        file_index: index du fichier DEM à charger (défaut: 100)
        method: méthode de partitionnement (cartesian, cylindrical, voronoi, quantile, octree)
        n_cells: nombre de cellules (défaut: 125)
    """
    try:
        file_index = int(request.GET.get("file_index", 100))
        method = request.GET.get("method", "cartesian")
        n_cells = int(request.GET.get("n_cells", 125))
        sample_every = int(request.GET.get("sample_every", 1))

        valid_methods = ["cartesian", "cylindrical", "voronoi", "quantile", "octree"]
        if method not in valid_methods:
            return JsonResponse(
                {
                    "error": f"Méthode inconnue: {method}. Valides: {', '.join(valid_methods)}"
                },
                status=400,
            )

        payload = _load_partition_data(
            file_index=file_index,
            method=method,
            n_cells=n_cells,
            sample_every=sample_every,
        )
        partitioner = create_partitioner(
            method, **_get_partitioner_kwargs(method, n_cells)
        )
        partitioner.fit(
            payload["coords"]
            if "coords" in payload
            else np.column_stack([payload["x"], payload["y"], payload["z"]])
        )

        # Ajouter les frontières de partitions
        boundaries = _compute_partition_boundaries(method, partitioner, payload)

        # conversions pour JSON
        return JsonResponse(
            {
                "x": payload["x"].tolist(),
                "y": payload["y"].tolist(),
                "z": payload["z"].tolist(),
                "states": payload["states"].tolist(),
                "n_cells": payload["n_cells"],
                "n_particles": payload["n_particles"],
                "method": payload["method"],
                "boundaries": boundaries,
            }
        )

    except ImportError as e:
        return JsonResponse({"error": f"Dépendances manquantes: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status=500
        )


def api_dem_particles(request):
    """
    API pour charger les particules DEM à un pas de temps arbitraire.
    Permet de visualiser les particules dans le mélangeur sans calcul de matrice.

    Query parameters:
        file_index: index du fichier DEM à charger (requis)
        sample_every: échantillonnage pour performance (défaut: 1)
        species_criterion: méthode de coloration (défaut: 'none')
            Options: 'z_median', 'r_median', 'quadrant', 'velocity', 'none'
    """
    try:
        # Imports locaux pour éviter les erreurs si les dépendances ne sont pas disponibles
        from huggingface_hub import HfFileSystem
        import polars as pl

        file_index = int(request.GET.get("file_index"))
        sample_every = max(1, int(request.GET.get("sample_every", 1)))
        species_criterion = request.GET.get("species_criterion", "none")

        # Charger les données DEM
        HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
        fs = HfFileSystem()
        files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))

        if file_index < 0 or file_index >= len(files):
            return JsonResponse(
                {
                    "error": f"file_index {file_index} hors limites (max: {len(files) - 1})"
                },
                status=400,
            )

        fname = files[file_index]
        with fs.open(fname, "rb") as f:
            df = pl.read_csv(f)

        # Extraire les coordonnées avec échantillonnage
        total_particles = len(df)
        sample_df = df[::sample_every]
        returned_particles = len(sample_df)

        x = sample_df["coordinates:0"].to_numpy()
        y = sample_df["coordinates:1"].to_numpy()
        z = sample_df["coordinates:2"].to_numpy()

        # Calculer les limites du cylindre du mélangeur
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        zmin, zmax = float(z.min()), float(z.max())

        bounds = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }

        # Données de base des particules
        particles = {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}

        metadata = {
            "file_index": file_index,
            "total_particles": total_particles,
            "returned_particles": returned_particles,
            "species_criterion": species_criterion,
        }

        # Calculer les données de coloration si demandé
        if species_criterion != "none":
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            if species_criterion == "z_median":
                median_z = float(np.median(z))
                species = [1 if zi > median_z else 0 for zi in z]
                particles["species"] = species

            elif species_criterion == "r_median":
                r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                median_r = float(np.median(r))
                species = [1 if ri > median_r else 0 for ri in r]
                particles["species"] = species

            elif species_criterion == "quadrant":
                median_z = float(np.median(z))
                r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                median_r = float(np.median(r))
                species = []
                for i in range(len(x)):
                    above_z = 1 if z[i] > median_z else 0
                    outer_r = 1 if r[i] > median_r else 0
                    species.append(above_z * 2 + outer_r)  # 0-3
                particles["species"] = species

            elif species_criterion == "velocity":
                if "velocity:0" in df.columns:
                    vx = sample_df["velocity:0"].to_numpy()
                    vy = sample_df["velocity:1"].to_numpy()
                    vz = sample_df["velocity:2"].to_numpy()
                    velocity_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                    particles["velocity_mag"] = velocity_mag.tolist()
                    metadata["has_velocity"] = True
                else:
                    metadata["has_velocity"] = False
                    particles["velocity_mag"] = [0.0] * returned_particles

        return JsonResponse(
            {
                "success": True,
                "particles": particles,
                "bounds": bounds,
                "metadata": metadata,
            }
        )

    except Exception as e:
        import traceback

        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status=500
        )


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


def api_partitions_pyvista(request):
    """PyVista désactivé sur macOS (incompatible avec Django threading)."""
    return JsonResponse(
        {
            "error": "PyVista n'est pas compatible avec Django sur macOS. Utilisez le rendu Trame.",
            "info": "Le rendu Trame/vtk.js affiche les partitions avec maillages 3D.",
        },
        status=501,
    )


# ═══════════════════════════════════════════════════════════════
# TRAME / VTK.js 3D PARTITION VISUALIZATION
# ═══════════════════════════════════════════════════════════════


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
    from scipy.spatial import Voronoi

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
        from scipy.spatial import ConvexHull

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


def partitioner_3d_trame(request):
    """Page de visualisation 3D avec Trame/vtk.js."""
    return render(request, "markov/partitioner_3d_trame.html")


def api_partitioner_trame_data(request):
    """
    API 100% DB pour le rendu 3D Trame/vtk.js.

    Paramètre requis: exp_id (ID d'une expérience en base)
    Retourne: géométrie des frontières + statistiques + diagnostics matrice.
    Aucun accès au bucket HuggingFace.
    """
    try:
        from .partitioner_params import get_partitioner_kwargs

        exp_id = request.GET.get("exp_id")
        if not exp_id:
            return JsonResponse({"error": "exp_id requis"}, status=400)

        # Charger l'expérience depuis la DB
        try:
            exp = Experiment.objects.select_related("partition_method", "matrix").get(
                id=int(exp_id)
            )
        except Experiment.DoesNotExist:
            return JsonResponse(
                {"error": f"Expérience {exp_id} non trouvée"}, status=404
            )

        pm = exp.partition_method
        method = pm.name
        params = pm.parameters or {}
        kwargs = get_partitioner_kwargs(method, **params)

        # Bounds par défaut (mélangeur cylindrique typique DEM)
        coord_bounds = {
            "xmin": -1.0,
            "xmax": 1.0,
            "ymin": -1.0,
            "ymax": 1.0,
            "zmin": 0.0,
            "zmax": 4.0,
        }

        # Stats depuis PartitionMethod + raw_stats
        n_states = exp.n_states or pm.n_cells or 0
        raw = exp.raw_stats or {}

        stats = {
            "n_states": int(n_states),
            "n_cells": int(pm.n_cells or n_states),
            "n_visited": int(raw.get("n_states_visited", pm.n_cells_visited or 0)),
            "n_empty": int(raw.get("n_states_empty", 0)),
            "fraction_visited": float(raw.get("fraction_visited", 0)),
            "cv": float(pm.population_cv) if pm.population_cv else 0,
            "population_mean": float(pm.population_mean) if pm.population_mean else 0,
            "population_std": float(pm.population_std) if pm.population_std else 0,
        }

        # Diagnostics matrice depuis TransitionMatrix
        matrix_data = None
        if hasattr(exp, "matrix") and exp.matrix:
            tm = exp.matrix
            matrix_data = {
                "diagonal_mean": float(tm.diagonal_mean) if tm.diagonal_mean else 0,
                "diagonal_std": float(tm.diagonal_std) if tm.diagonal_std else 0,
                "eigenvalue_2": float(tm.eigenvalue_2) if tm.eigenvalue_2 else 0,
                "spectral_gap": float(tm.spectral_gap) if tm.spectral_gap else 0,
                "n_states_visited": int(tm.n_states_visited or 0),
                "fraction_visited": float(tm.fraction_visited or 0),
            }

        # Récupérer le mode radial depuis la requête (pour visualisation cylindrique)
        radial_mode = request.GET.get("radial_mode", "equal_dr")

        # Pour Voronoi/Physics: charger les centroïdes réels depuis le bucket pour garantir la cohérence
        actual_centroids = None
        if method in ["voronoi", "physics"]:
            try:
                # Charger les particules depuis le bucket
                HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
                fs = HfFileSystem()
                csv_files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))

                file_index = exp.start_index or 0
                if file_index < len(csv_files):
                    import polars as pl

                    csv_path = csv_files[file_index]
                    with fs.open(csv_path, "rb") as f:
                        df = pl.read_csv(f)

                    coords = np.column_stack(
                        [
                            df["coordinates:0"].to_numpy(),
                            df["coordinates:1"].to_numpy(),
                            df["coordinates:2"].to_numpy(),
                        ]
                    )

                    # Créer et ajuster le partitionneur pour obtenir les centroïdes réels
                    partitioner = create_partitioner(method, **kwargs)
                    partitioner.fit(coords)

                    # Extraire les centroïdes (fonctionne pour Voronoi et Physics)
                    if hasattr(partitioner, "centroids"):
                        actual_centroids = partitioner.centroids.tolist()
                    elif hasattr(partitioner, "_centroids"):
                        actual_centroids = partitioner._centroids.tolist()

                    # Mettre à jour les bounds réels depuis les données
                    coord_bounds = {
                        "xmin": float(coords[:, 0].min()),
                        "xmax": float(coords[:, 0].max()),
                        "ymin": float(coords[:, 1].min()),
                        "ymax": float(coords[:, 1].max()),
                        "zmin": float(coords[:, 2].min()),
                        "zmax": float(coords[:, 2].max()),
                    }
            except Exception as e:
                # Fallback vers l'ancien comportement si erreur
                import logging

                logging.warning(
                    f"Impossible de charger les centroïdes réels pour {method}: {e}"
                )

        # Calculer la géométrie des frontières avec les centroïdes réels si disponibles
        mesh_groups = _compute_partition_mesh_from_params(
            method,
            params,
            coord_bounds,
            radial_mode=radial_mode,
            actual_centroids=actual_centroids,
        )

        # Ajouter le mode radial aux paramètres retournés
        if method == "cylindrical":
            kwargs["radial_mode"] = radial_mode

        result = {
            "success": True,
            "method": method,
            "start_index": int(exp.start_index or 0),
            "parameters": {
                k: (int(v) if isinstance(v, (np.integer,)) else v)
                for k, v in kwargs.items()
            },
            "statistics": stats,
            "matrix": matrix_data,
            "mesh_groups": _sanitize_for_json(mesh_groups),
        }

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status=500
        )


def api_partitioner_particles(request):
    """
    Charge les particules depuis le bucket HuggingFace pour une expérience.
    Retourne les coordonnées + états de partition pour rendu 3D.
    """
    try:
        from .partitioner_params import get_partitioner_kwargs

        exp_id = request.GET.get("exp_id")
        if not exp_id:
            return JsonResponse({"error": "exp_id requis"}, status=400)

        exp = Experiment.objects.select_related("partition_method").get(id=int(exp_id))
        pm = exp.partition_method
        method = pm.name
        params = pm.parameters or {}
        kwargs = get_partitioner_kwargs(method, **params)

        # Charger depuis bucket
        HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
        fs = HfFileSystem()
        csv_files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))

        file_index = exp.start_index or 0
        if file_index >= len(csv_files):
            return JsonResponse(
                {
                    "error": f"File index {file_index} hors limites ({len(csv_files)} fichiers)"
                },
                status=400,
            )

        csv_path = csv_files[file_index]
        with fs.open(csv_path, "rb") as f:
            df = pl.read_csv(f)

        coords = np.column_stack(
            [
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ]
        )

        partitioner = create_partitioner(method, **kwargs)
        partitioner.fit(coords)
        states = partitioner.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])

        # Sous-échantillonner si trop de points
        max_particles = 50000
        step = max(1, len(coords) // max_particles)
        sampled = coords[::step]
        sampled_states = [int(s) for s in states[::step]]
        n_states = int(len(np.unique(states)))

        return JsonResponse(
            {
                "success": True,
                "particles": {
                    "x": [float(v) for v in sampled[:, 0]],
                    "y": [float(v) for v in sampled[:, 1]],
                    "z": [float(v) for v in sampled[:, 2]],
                    "states": sampled_states,
                    "n_states": n_states,
                    "count": len(sampled_states),
                },
                "bounds": {
                    "xmin": float(coords[:, 0].min()),
                    "xmax": float(coords[:, 0].max()),
                    "ymin": float(coords[:, 1].min()),
                    "ymax": float(coords[:, 1].max()),
                    "zmin": float(coords[:, 2].min()),
                    "zmax": float(coords[:, 2].max()),
                },
            }
        )

    except Exception as e:
        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status=500
        )


def api_comparison(request):
    """API pour charger les données de comparaison pour toutes les méthodes."""
    try:
        file_index = int(request.GET.get("file_index", 100))
        n_cells = int(request.GET.get("n_cells", 125))
        sample_every = int(request.GET.get("sample_every", 1))

        methods = ["cartesian", "cylindrical", "quantile", "voronoi", "octree"]
        comparison_data = {}

        for method in methods:
            try:
                payload = _load_partition_data(
                    file_index=file_index,
                    method=method,
                    n_cells=n_cells,
                    sample_every=sample_every,
                )

                partitioner = create_partitioner(
                    method, **_get_partitioner_kwargs(method, n_cells)
                )
                partitioner.fit(
                    np.column_stack([payload["x"], payload["y"], payload["z"]])
                )

                # Compute metrics
                states = payload["states"]
                n_cells_actual = payload["n_cells"]
                counts = np.bincount(states, minlength=n_cells_actual)

                non_zero = counts[counts > 0]
                mean = float(non_zero.mean()) if len(non_zero) > 0 else 0
                std = float(non_zero.std()) if len(non_zero) > 0 else 0
                cv = std / mean if mean > 0 else 0
                sparsity = int((counts == 0).sum())
                visited = int((counts > 0).sum())

                comparison_data[method] = {
                    "success": True,
                    "n_cells": n_cells_actual,
                    "n_particles": payload["n_particles"],
                    "population_min": int(counts.min()),
                    "population_max": int(counts.max()),
                    "population_mean": float(counts.mean()),
                    "cv": float(cv),
                    "sparsity": int(sparsity),
                    "visited": int(visited),
                    "fraction_visited": float(visited / n_cells_actual),
                }
            except Exception as e:
                comparison_data[method] = {"success": False, "error": str(e)}

        return JsonResponse(comparison_data)

    except Exception as e:
        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status=500
        )


# ═══════════════════════════════════════════════════════════════
# UNIFIED ANALYSIS TABS APIs
# ═══════════════════════════════════════════════════════════════


def api_comparison_tab(request):
    """⚖️ API pour l'onglet Comparaison - données de comparaison entre méthodes."""
    try:
        # Récupérer les données depuis la BD
        qs = (
            Experiment.objects.select_related("partition_method", "matrix")
            .filter(matrix__isnull=False)
            .distinct()
        )

        methods_data = {}

        for exp in qs:
            try:
                method = exp.partition_method.name
                matrix = exp.matrix

                if method not in methods_data:
                    methods_data[method] = {
                        "label": exp.partition_method.label,
                        "experiments": [],
                    }

                methods_data[method]["experiments"].append(
                    {
                        "id": exp.id,
                        "name": exp.folder_name[:50],
                        "n_states": exp.n_states,
                        "nlt": exp.nlt or 0,
                        "step_size": exp.step_size or 0,
                        "diagonal_mean": float(matrix.diagonal_mean)
                        if matrix.diagonal_mean
                        else 0,
                        "eigenvalue_2": float(matrix.eigenvalue_2)
                        if matrix.eigenvalue_2
                        else 0,
                        "spectral_gap": float(matrix.spectral_gap)
                        if matrix.spectral_gap
                        else 0,
                        "fraction_visited": float(matrix.fraction_visited)
                        if matrix.fraction_visited
                        else 0,
                    }
                )
            except Exception as e:
                pass

        # Créer la table de comparaison
        comparison_table = []
        for method, data in sorted(methods_data.items()):
            exps = data["experiments"]
            if exps:
                comparison_table.append(
                    {
                        "method": data["label"],
                        "n_experiments": len(exps),
                        "n_states_avg": float(np.mean([e["n_states"] for e in exps])),
                        "diagonal_mean_avg": float(
                            np.mean([e["diagonal_mean"] for e in exps])
                        ),
                        "spectral_gap_avg": float(
                            np.mean([e["spectral_gap"] for e in exps])
                        ),
                        "eigenvalue_2_avg": float(
                            np.mean([e["eigenvalue_2"] for e in exps])
                        ),
                        "fraction_visited_avg": float(
                            np.mean([e["fraction_visited"] for e in exps])
                        ),
                    }
                )

        return JsonResponse(
            {
                "methods_data": {
                    k: {"label": v["label"], "experiments": v["experiments"][:5]}
                    for k, v in methods_data.items()
                },
                "comparison_table": comparison_table,
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_rsd_tab(request):
    """📉 API pour l'onglet RSD - données de mélange RSD vs temps."""
    try:
        # Récupérer les RSD résultats de la BD
        rsd_results = RSDResult.objects.select_related("experiment").filter(
            source="dem"
        )[:10]  # Limiter à 10

        rsd_data = []

        for result in rsd_results:
            exp = result.experiment
            rsd_data.append(
                {
                    "id": exp.id,
                    "name": exp.folder_name[:50],
                    "method": exp.partition_method.name,
                    "n_states": exp.n_states,
                    "rsd_initial": float(result.rsd_initial)
                    if result.rsd_initial
                    else 0,
                    "rsd_final": float(result.rsd_final) if result.rsd_final else 0,
                    "mixing_time_50": result.mixing_time_50,
                    "mixing_time_90": result.mixing_time_90,
                    "rsd_curve": (result.rsd_curve or [])[
                        :100
                    ],  # Limiter aux 100 premiers
                    "entropy": (result.entropy_curve or [])[:100]
                    if hasattr(result, "entropy_curve")
                    else [],
                }
            )

        return JsonResponse({"rsd_data": rsd_data, "count": len(rsd_data)})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_mixing_tab(request):
    """🌀 API pour l'onglet Mélange - dynamiques de convergence entropie/variance."""
    try:
        # Récupérer les expériences avec données de mélange
        qs = (
            Experiment.objects.select_related("partition_method", "matrix")
            .filter(matrix__isnull=False)
            .distinct()[:10]
        )

        mixing_data = []

        for exp in qs:
            try:
                # Simuler l'évolution d'entropie basée sur les eigenvalues
                matrix = exp.matrix
                eigenvalue_2 = (
                    float(matrix.eigenvalue_2) if matrix.eigenvalue_2 else 0.5
                )
                n_states = exp.n_states

                # Entropie convergence théorique: H(t) = 1 - (1 - H_0) * λ₂^t
                entropy_history = []
                variance_history = []

                for t in range(100):
                    # Convergence exponentielle basée sur λ₂
                    entropy = 1.0 - (1.0 - 1.0 / n_states) * (eigenvalue_2**t)
                    variance = (1.0 - entropy) ** 2 * n_states

                    entropy_history.append(float(entropy))
                    variance_history.append(float(variance))

                mixing_data.append(
                    {
                        "id": exp.id,
                        "method": exp.partition_method.name,
                        "name": exp.folder_name[:50],
                        "n_states": exp.n_states,
                        "eigenvalue_2": eigenvalue_2,
                        "entropy_history": entropy_history,
                        "variance_history": variance_history,
                    }
                )
            except Exception as e:
                pass

        return JsonResponse({"mixing_data": mixing_data, "count": len(mixing_data)})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_eigenvalues_tab(request):
    """🔢 API pour l'onglet Valeurs propres - spectre eigenvalues λ₂."""
    try:
        # Récupérer les expériences avec leurs eigenvalues
        qs = (
            Experiment.objects.select_related("partition_method", "matrix")
            .filter(matrix__isnull=False)
            .distinct()
        )

        eigenvalues_data = []
        methods_lambda2 = {}

        for exp in qs:
            try:
                matrix = exp.matrix
                method = exp.partition_method.name

                if method not in methods_lambda2:
                    methods_lambda2[method] = {
                        "label": exp.partition_method.label,
                        "values": [],
                    }

                eigenvalue_2 = float(matrix.eigenvalue_2) if matrix.eigenvalue_2 else 0
                methods_lambda2[method]["values"].append(eigenvalue_2)

                eigenvalues_data.append(
                    {
                        "id": exp.id,
                        "name": exp.folder_name[:50],
                        "method": method,
                        "n_states": exp.n_states,
                        "eigenvalue_2": eigenvalue_2,
                        "spectral_gap": float(1.0 - eigenvalue_2)
                        if eigenvalue_2 > 0
                        else 0,
                    }
                )
            except Exception as e:
                pass

        # Créer la table de comparaison par méthode
        methods_summary = []
        for method, data in sorted(methods_lambda2.items()):
            values = data["values"]
            if values:
                methods_summary.append(
                    {
                        "method": data["label"],
                        "lambda2_mean": float(np.mean(values)),
                        "lambda2_std": float(np.std(values)),
                        "spectral_gap_mean": float(1.0 - np.mean(values)),
                        "n_experiments": len(values),
                    }
                )

        return JsonResponse(
            {
                "eigenvalues_data": eigenvalues_data,
                "methods_summary": methods_summary,
                "count": len(eigenvalues_data),
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_dem_vs_markov_tab(request):
    """🔬 API pour l'onglet DEM vs Markov - comparaison simulation réelle vs prédite."""
    try:
        # Récupérer les comparaisons DEM vs Markov
        dem_exp_ids = set(
            RSDResult.objects.filter(source="dem").values_list(
                "experiment_id", flat=True
            )
        )
        markov_exp_ids = set(
            RSDResult.objects.filter(source="markov").values_list(
                "experiment_id", flat=True
            )
        )
        both_exp_ids = dem_exp_ids & markov_exp_ids

        experiments_with_both = Experiment.objects.filter(
            id__in=both_exp_ids
        ).select_related("partition_method")[:20]

        dem_markov_data = []

        for exp in experiments_with_both:
            try:
                dem_rsd = RSDResult.objects.filter(experiment=exp, source="dem").first()
                markov_rsd = RSDResult.objects.filter(
                    experiment=exp, source="markov"
                ).first()

                if dem_rsd and markov_rsd:
                    dem_rsd_final = float(dem_rsd.rsd_final) if dem_rsd.rsd_final else 0
                    markov_rsd_final = (
                        float(markov_rsd.rsd_final) if markov_rsd.rsd_final else 0
                    )

                    dem_markov_data.append(
                        {
                            "id": exp.id,
                            "name": exp.folder_name[:50],
                            "method": exp.partition_method.label,
                            "dem": {
                                "rsd_initial": float(dem_rsd.rsd_initial)
                                if dem_rsd.rsd_initial
                                else 0,
                                "rsd_final": dem_rsd_final,
                                "mixing_time_50": dem_rsd.mixing_time_50,
                                "rsd_curve": (dem_rsd.rsd_curve or [])[:100],
                            },
                            "markov": {
                                "rsd_initial": float(markov_rsd.rsd_initial)
                                if markov_rsd.rsd_initial
                                else 0,
                                "rsd_final": markov_rsd_final,
                                "mixing_time_50": markov_rsd.mixing_time_50,
                                "rsd_curve": (markov_rsd.rsd_curve or [])[:100],
                            },
                            "rsd_final_diff": abs(dem_rsd_final - markov_rsd_final),
                            "rsd_final_relative": (
                                100
                                * abs(dem_rsd_final - markov_rsd_final)
                                / (dem_rsd_final + 0.001)
                            ),
                        }
                    )
            except Exception as e:
                pass

        # Créer la table récapitulative
        if dem_markov_data:
            avg_diff = np.mean([d["rsd_final_diff"] for d in dem_markov_data])
            avg_relative = np.mean([d["rsd_final_relative"] for d in dem_markov_data])
        else:
            avg_diff = 0
            avg_relative = 0

        return JsonResponse(
            {
                "dem_markov_data": dem_markov_data,
                "summary": {
                    "count": len(dem_markov_data),
                    "avg_rsd_diff": float(avg_diff),
                    "avg_rsd_relative_diff": float(avg_relative),
                },
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════
# 3D PARTITIONER VISUALIZATION WITH CUSTOM PARAMETERS
# ═══════════════════════════════════════════════════════════════


def api_partitioner_schemas(request):
    """API pour obtenir les schémas de paramètres de tous les partitionneurs."""
    try:
        from .partitioner_params import PARTITIONER_SCHEMAS

        return JsonResponse(
            {
                "schemas": PARTITIONER_SCHEMAS,
                "methods": list(PARTITIONER_SCHEMAS.keys()),
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_partitioner_3d_data(request):
    """API pour générer les données 3D d'un partitionnement avec paramètres personnalisés."""
    try:
        method = request.GET.get("method", "cartesian")
        file_index = int(request.GET.get("file_index", 100))

        # Récupérer TOUS les paramètres depuis la requête
        from .partitioner_params import get_partitioner_kwargs, get_partitioner_schema

        # Construire le dict de paramètres
        schema = get_partitioner_schema(method)
        if not schema:
            return JsonResponse(
                {"error": f"Unknown partitioner method: {method}"}, status=400
            )

        params = {}
        # Pour chaque paramètre du schéma, essayer de le lire depuis la requête
        for param_name in schema["parameters"].keys():
            if param_name in request.GET:
                value = request.GET.get(param_name)
                # Essayer de convertir en int ou float si nécessaire
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value) if value.isdigit() else value
                except:
                    pass
                params[param_name] = value

        # Utiliser get_partitioner_kwargs qui gerera les transformations
        kwargs = get_partitioner_kwargs(method, **params)

        # Charger les particules depuis le bucket HuggingFace
        try:
            from bucket_io import HfFileSystem, load_experiment_from_bucket

            fs = HfFileSystem(repo_id="ktongue/DEM_MCM")
            files = fs.ls("", detail=False)
            csv_files = [f for f in files if f.endswith(".csv")]
            csv_files.sort()

            if file_index < len(csv_files):
                csv_path = csv_files[file_index]
                with fs.open(csv_path, "r") as f:
                    coords = np.loadtxt(f, delimiter=",", skiprows=1)[
                        :, 1:4
                    ]  # Skip ID column
            else:
                return JsonResponse(
                    {"error": f"File index {file_index} out of range"}, status=400
                )
        except Exception as e:
            return JsonResponse(
                {"error": f"Failed to load DEM data: {str(e)}"}, status=500
            )

        # Créer et fitter le partitionneur
        try:
            partitioner = create_partitioner(method, **kwargs)
            partitioner.fit(coords)
            states = partitioner.predict(coords)
        except Exception as e:
            return JsonResponse(
                {"error": f"Failed to create partitioner: {str(e)}"}, status=500
            )

        # Calculer les statistiques
        n_states = len(np.unique(states))
        counts = np.bincount(states, minlength=n_states)
        visited = int((counts > 0).sum())
        sparsity = int((counts == 0).sum())

        non_zero = counts[counts > 0]
        cv = (
            float(non_zero.std() / non_zero.mean())
            if len(non_zero) > 0 and non_zero.mean() > 0
            else 0
        )

        # Préparer les données Plotly
        plot_data = {
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "z": coords[:, 2].tolist(),
            "states": states.tolist(),
            "n_states": n_states,
        }

        return JsonResponse(
            {
                "success": True,
                "method": method,
                "parameters": kwargs,
                "statistics": {
                    "n_particles": len(coords),
                    "n_states": n_states,
                    "n_visited": visited,
                    "n_empty": sparsity,
                    "fraction_visited": float(visited / n_states)
                    if n_states > 0
                    else 0,
                    "cv": float(cv),
                    "population_min": int(counts.min()),
                    "population_max": int(counts.max()),
                    "population_mean": float(counts.mean()),
                },
                "plot_data": plot_data,
                "file_index": file_index,
                "file_count": len(csv_files) if "csv_files" in locals() else 0,
            }
        )

    except Exception as e:
        return JsonResponse(
            {"error": f"Error: {str(e)}", "traceback": traceback.format_exc()},
            status=500,
        )


def api_dem_vs_markov_3d(request):
    """API pour charger les données DEM et Markov pour une expérience en 3D."""
    try:
        exp_id = int(request.GET.get("exp_id"))

        # Charger l'expérience
        experiment = Experiment.objects.select_related(
            "partition_method", "matrix"
        ).get(id=exp_id)

        # Charger les données DEM
        try:
            from bucket_io import HfFileSystem

            fs = HfFileSystem(repo_id="ktongue/DEM_MCM")
            files = fs.ls("", detail=False)
            csv_files = [f for f in files if f.endswith(".csv")]
            csv_files.sort()

            if len(csv_files) > 0:
                # Load first DEM snapshot
                with fs.open(csv_files[0], "r") as f:
                    dem_coords = np.loadtxt(f, delimiter=",", skiprows=1)[:, 1:4]
            else:
                dem_coords = None
        except Exception as e:
            dem_coords = None

        # Charger la matrice Markov
        matrix_data = {}
        if experiment.matrix and experiment.matrix.matrix_bucket_path:
            try:
                from bucket_io import load_matrix_from_bucket

                P = load_matrix_from_bucket(experiment.matrix.matrix_bucket_path)
                matrix_data = {
                    "shape": list(P.shape),
                    "diagonal_mean": float(np.diag(P).mean()),
                    "eigenvalue_2": float(experiment.matrix.eigenvalue_2)
                    if experiment.matrix.eigenvalue_2
                    else 0,
                    "spectral_gap": float(experiment.matrix.spectral_gap)
                    if experiment.matrix.spectral_gap
                    else 0,
                }
            except Exception as e:
                matrix_data = {"error": str(e)}

        return JsonResponse(
            {
                "experiment": {
                    "id": experiment.id,
                    "name": experiment.folder_name,
                    "method": experiment.partition_method.name,
                    "n_states": experiment.n_states,
                },
                "dem": {
                    "loaded": dem_coords is not None,
                    "n_particles": int(len(dem_coords))
                    if dem_coords is not None
                    else 0,
                },
                "markov": matrix_data,
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def partitioner_3d(request):
    """Render the 3D partitioner visualization page."""
    return render(request, "markov/partitioner_3d.html")


# ═══════════════════════════════════════════════════════════════════════
# ANALYSE UNIFIÉE - ENDPOINTS POUR INTERFACE ENRICHIE
# ═══════════════════════════════════════════════════════════════════════


def api_analysis_experiments(request):
    """
    API pour récupérer la liste des expériences avec filtres avancés.

    GET /api/analysis/experiments/
    Paramètres query:
      - method: str (optionnel)
      - nlt: str (comma-separated: "20,100,200")
      - step_size: str (comma-separated)
      - start_index: str (comma-separated)
      - n_states: str (comma-separated)
      - sort_by: str ("method", "n_states", "nlt")

    Retour: {
        "status": "success",
        "filters": {...},
        "experiments": [...],
        "total_count": int
    }
    """
    try:
        # Récupérer les paramètres de filtre
        method_filter = request.GET.get("method", "").strip()
        nlt_filter = [
            int(x) for x in request.GET.get("nlt", "").split(",") if x.strip().isdigit()
        ]
        step_size_filter = [
            int(x)
            for x in request.GET.get("step_size", "").split(",")
            if x.strip().isdigit()
        ]
        start_index_filter = [
            int(x)
            for x in request.GET.get("start_index", "").split(",")
            if x.strip().isdigit()
        ]
        n_states_filter = [
            int(x)
            for x in request.GET.get("n_states", "").split(",")
            if x.strip().isdigit()
        ]
        sort_by = request.GET.get("sort_by", "method").strip()

        # Construire la requête de base
        qs = Experiment.objects.select_related("partition_method", "matrix").all()

        # Appliquer les filtres
        if method_filter:
            qs = qs.filter(partition_method__name=method_filter)
        if nlt_filter:
            qs = qs.filter(nlt__in=nlt_filter)
        if step_size_filter:
            qs = qs.filter(step_size__in=step_size_filter)
        if start_index_filter:
            qs = qs.filter(start_index__in=start_index_filter)
        if n_states_filter:
            qs = qs.filter(n_states__in=n_states_filter)

        # Trier
        if sort_by == "n_states":
            qs = qs.order_by("n_states", "partition_method__name")
        elif sort_by == "nlt":
            qs = qs.order_by("nlt", "partition_method__name")
        else:  # 'method'
            qs = qs.order_by("partition_method__name", "n_states")

        # Récupérer les valeurs uniques pour les filtres
        all_experiments = Experiment.objects.all()
        methods = sorted(
            all_experiments.values_list("partition_method__name", flat=True).distinct()
        )
        nlt_values = sorted(all_experiments.values_list("nlt", flat=True).distinct())
        step_size_values = sorted(
            all_experiments.values_list("step_size", flat=True).distinct()
        )
        start_index_values = sorted(
            all_experiments.values_list("start_index", flat=True).distinct()
        )
        n_states_values = sorted(
            all_experiments.values_list("n_states", flat=True).distinct()
        )

        # Formater la réponse
        experiments_data = []
        for exp in qs:
            exp_data = {
                "id": exp.id,
                "folder_name": exp.folder_name,
                "method": exp.partition_method.name
                if exp.partition_method
                else "unknown",
                "nlt": exp.nlt,
                "step_size": exp.step_size,
                "start_index": exp.start_index,
                "n_states": exp.n_states,
                "n_cells_visited": exp.partition_method.n_cells_visited
                if exp.partition_method
                else 0,
                "diagonal_mean": exp.matrix.diagonal_mean if exp.matrix else 0,
                "diagonal_std": exp.matrix.diagonal_std if exp.matrix else 0,
                "eigenvalue_2": float(exp.matrix.eigenvalue_2)
                if exp.matrix and exp.matrix.eigenvalue_2
                else 0,
                "spectral_gap": float(exp.matrix.spectral_gap)
                if exp.matrix and exp.matrix.spectral_gap
                else 0,
                "fraction_visited": float(exp.matrix.fraction_visited)
                if exp.matrix and exp.matrix.fraction_visited
                else 0,
                "parameters": exp.partition_method.parameters
                if exp.partition_method
                else {},
            }
            experiments_data.append(exp_data)

        return JsonResponse(
            {
                "status": "success",
                "filters": {
                    "methods": [str(m) for m in methods if m],
                    "nlt_values": [int(n) for n in nlt_values if n],
                    "step_size_values": [int(s) for s in step_size_values if s],
                    "start_index_values": [int(s) for s in start_index_values if s],
                    "n_states_values": [int(n) for n in n_states_values if n],
                },
                "experiments": experiments_data,
                "total_count": len(experiments_data),
            }
        )

    except Exception as e:
        logger.error(f"Erreur dans api_analysis_experiments: {traceback.format_exc()}")
        return JsonResponse(
            {
                "status": "error",
                "message": f"Erreur lors de la récupération des expériences: {str(e)}",
            },
            status=500,
        )


def api_analysis_generate_images(request):
    """
    API pour générer les images d'analyse.

    POST /api/analysis/generate-images/
    Body JSON:
    {
        "experiment_ids": [1, 2, 3],
        "analyses": [
            {
                "type": "plot_experiment",
                "for_each_experiment": true,
                "params": {"n_steps": 200}
            },
            ...
        ],
        "global_params": {
            "figsize": [16, 10],
            "n_steps": 200,
            "dem_criterion": "z_median",
            "export_format": "PNG"
        }
    }

    Retour: {
        "status": "success",
        "images": [...],
        "tabs_generated": [...],
        "generation_time": float,
        "message": str
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)

        experiment_ids = data.get("experiment_ids", [])
        analyses_config = data.get("analyses", [])
        global_params = data.get("global_params", {})

        # Valider les paramètres
        if not experiment_ids:
            return JsonResponse(
                {"status": "error", "message": "Aucune expérience sélectionnée"},
                status=400,
            )

        if not analyses_config:
            return JsonResponse(
                {"status": "error", "message": "Aucune analyse sélectionnée"},
                status=400,
            )

        # Utiliser le wrapper
        from .markov_analyzer_wrapper import DjangoMarkovAnalyzerWrapper

        wrapper = DjangoMarkovAnalyzerWrapper()
        result = wrapper.generate_analysis_images(
            experiment_ids=experiment_ids,
            analyses_config=analyses_config,
            global_params=global_params,
        )

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "JSON invalide"}, status=400)

    except Exception as e:
        logger.error(
            f"Erreur dans api_analysis_generate_images: {traceback.format_exc()}"
        )
        return JsonResponse(
            {
                "status": "error",
                "message": f"Erreur lors de la génération: {str(e)}",
                "error": traceback.format_exc(),
            },
            status=500,
        )


def api_analysis_data(request):
    """
    Retourne les données d'analyse pour les expériences sélectionnées.

    GET /api/analysis/data/?exp_ids=1,2,3
    Retourne: comparaison, eigenvalues, entropy_histories avec labels intelligents.
    """
    try:
        exp_ids_str = request.GET.get("exp_ids", "")
        if not exp_ids_str:
            return JsonResponse({"error": "exp_ids requis"}, status=400)

        exp_ids = [int(x) for x in exp_ids_str.split(",") if x.strip().isdigit()]
        experiments = Experiment.objects.select_related(
            "partition_method", "matrix"
        ).filter(id__in=exp_ids)

        # Collecter les données
        exp_data_list = []
        for exp in experiments:
            pm = exp.partition_method
            matrix = exp.matrix if hasattr(exp, "matrix") else None
            raw = exp.raw_stats or {}

            d = {
                "id": exp.id,
                "name": exp.folder_name,
                "method": pm.name,
                "label": pm.label,
                "n_states": exp.n_states or pm.n_cells or 0,
                "nlt": exp.nlt or 0,
                "step_size": exp.step_size or 1,
                "start_index": exp.start_index or 0,
                "parameters": pm.parameters or {},
                "diagonal_mean": float(matrix.diagonal_mean)
                if matrix and matrix.diagonal_mean
                else 0,
                "diagonal_std": float(matrix.diagonal_std)
                if matrix and matrix.diagonal_std
                else 0,
                "eigenvalue_2": float(matrix.eigenvalue_2)
                if matrix and matrix.eigenvalue_2
                else 0,
                "spectral_gap": float(matrix.spectral_gap)
                if matrix and matrix.spectral_gap
                else 0,
                "n_states_visited": int(matrix.n_states_visited) if matrix else 0,
                "fraction_visited": float(matrix.fraction_visited) if matrix else 0,
            }

            # Compute RSD and entropy from matrix
            d["rsd_markov"] = []
            d["rsd_markov_times"] = []
            d["entropy_history"] = []
            d["rsd_initial"] = 0
            d["rsd_final"] = 0
            d["mixing_time_50"] = None
            d["mixing_time_90"] = None
            if matrix and matrix.matrix_bucket_path:
                try:
                    from bucket_io import load_matrix_from_bucket

                    P = load_matrix_from_bucket(matrix.matrix_bucket_path)
                    # 6000 pas = 60 secondes (dt=0.01s)
                    n_steps_rsd = min(6000, P.shape[0])
                    rsd_result = _compute_rsd_from_matrix(P, n_steps=n_steps_rsd)
                    d["rsd_markov"] = rsd_result["rsd_percent"]
                    d["entropy_history"] = rsd_result["entropy"]
                    d["rsd_initial"] = rsd_result["rsd_initial"]
                    d["rsd_final"] = rsd_result["rsd_final"]
                    d["mixing_time_50"] = rsd_result["mixing_time_50"]
                    d["mixing_time_90"] = rsd_result["mixing_time_90"]

                    # Calculate actual time for Markov RSD based on step_size
                    # Each Markov step represents step_size DEM timesteps
                    # DEM timestep = 0.01s, so Markov step = step_size * 0.01s
                    markov_dt = d["step_size"] * 0.01  # seconds per Markov step
                    d["rsd_markov_times"] = [
                        i * markov_dt for i in range(len(d["rsd_markov"]))
                    ]
                except Exception as e:
                    logger.warning(f"RSD computation failed for {exp.folder_name}: {e}")

            exp_data_list.append(d)

        # Détecter les paramètres qui varient
        varying_params = _detect_varying_params(exp_data_list)

        # Générer les labels intelligents
        for d in exp_data_list:
            d["smart_label"] = _make_smart_label(d, varying_params)

        return JsonResponse(
            {
                "success": True,
                "experiments": exp_data_list,
                "varying_params": varying_params,
                "count": len(exp_data_list),
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


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


def api_rsd_comparison(request):
    """
    Calcule et compare les courbes RSD Markov et DEM pour une expérience.

    GET /api/rsd-comparison/?exp_id=123
    Retourne: rsd_markov[], rsd_dem[], time_steps[]
    """
    try:
        from bucket_io import HfFileSystem
        from .partitioner_params import get_partitioner_kwargs

        exp_id = request.GET.get("exp_id")
        if not exp_id:
            return JsonResponse({"error": "exp_id requis"}, status=400)

        exp = Experiment.objects.select_related("partition_method", "matrix").get(
            id=int(exp_id)
        )
        pm = exp.partition_method
        method = pm.name
        params = pm.parameters or {}
        kwargs = get_partitioner_kwargs(method, **params)

        start = exp.start_index or 0
        step_size = exp.step_size or 1

        # Time: DEM step = 0.01s, Markov step = step_size * 0.01s
        # Each Markov step represents step_size DEM timesteps
        # 60s total -> DEM: 6000 steps, Markov: ceil(60 / (step_size*0.01)) steps
        dem_dt = 0.01
        markov_dt = step_size * dem_dt  # Each Markov step = step_size * 0.01s
        total_time = 60.0
        n_dem_steps = int(total_time / dem_dt)
        n_markov_steps = int(total_time / markov_dt)

        # Sous-echantillonnage
        dem_sample_every = max(1, n_dem_steps // 100)

        # --- Load DEM data and partitioner ---
        rsd_dem = []
        rsd_markov = []
        rsd_markov_entropy = []
        try:
            HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
            fs = HfFileSystem()
            all_csv = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))

            # Load reference snapshot to fit partitioner
            ref_idx = min(start, len(all_csv) - 1)
            with fs.open(all_csv[ref_idx], "rb") as f:
                df_ref = pl.read_csv(f)
            coords_ref = np.column_stack(
                [
                    df_ref["coordinates:0"].to_numpy(),
                    df_ref["coordinates:1"].to_numpy(),
                    df_ref["coordinates:2"].to_numpy(),
                ]
            )
            partitioner = create_partitioner(method, **kwargs)
            partitioner.fit(coords_ref)
            states_ref = partitioner.compute_states(
                coords_ref[:, 0], coords_ref[:, 1], coords_ref[:, 2]
            )

            # Initial state vector from DEM (particle counts per cell, normalized)
            n_states = int(states_ref.max()) + 1
            initial_counts = np.bincount(states_ref, minlength=n_states).astype(float)
            initial_state = (
                initial_counts / initial_counts.sum()
                if initial_counts.sum() > 0
                else initial_counts
            )

            # --- Markov RSD from matrix with SAME initial state ---
            if exp.matrix and exp.matrix.matrix_bucket_path:
                try:
                    from bucket_io import load_matrix_from_bucket

                    P = load_matrix_from_bucket(exp.matrix.matrix_bucket_path)
                    mr = _compute_rsd_from_matrix(
                        P, n_steps=n_markov_steps, initial_state=initial_state
                    )
                    rsd_markov = mr["rsd_percent"]
                    rsd_markov_entropy = mr["entropy"]
                    # Create time array for Markov RSD (actual time in seconds)
                    markov_times = [i * markov_dt for i in range(len(rsd_markov))]
                except Exception as e:
                    logger.warning(f"Markov RSD failed: {e}")
                    markov_times = []

            # --- DEM RSD from snapshots at each timestep ---
            for t in range(0, n_dem_steps, dem_sample_every):
                file_idx = start + t  # DEM files are at consecutive indices
                if file_idx >= len(all_csv):
                    break
                try:
                    with fs.open(all_csv[file_idx], "rb") as f:
                        df = pl.read_csv(f)
                    coords = np.column_stack(
                        [
                            df["coordinates:0"].to_numpy(),
                            df["coordinates:1"].to_numpy(),
                            df["coordinates:2"].to_numpy(),
                        ]
                    )
                    states = partitioner.compute_states(
                        coords[:, 0], coords[:, 1], coords[:, 2]
                    )
                    counts = np.bincount(states, minlength=n_states)
                    visited = counts[counts > 0]
                    if len(visited) > 1:
                        rsd_val = float(visited.std() / visited.mean() * 100)
                    else:
                        rsd_val = 0.0
                    rsd_dem.append({"t": t * dem_dt, "rsd": rsd_val})
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"DEM/Markov RSD failed: {e}")

        return JsonResponse(
            {
                "success": True,
                "method": method,
                "step_size": step_size,
                "n_markov_steps": n_markov_steps,
                "n_dem_steps": n_dem_steps,
                "markov_dt": markov_dt,
                "dem_dt": dem_dt,
                "total_time_s": total_time,
                "rsd_markov": rsd_markov,
                "rsd_markov_entropy": rsd_markov_entropy,
                "rsd_dem": rsd_dem,
                "label": _make_smart_label(
                    {
                        "method": method,
                        "n_states": exp.n_states or 0,
                        "nlt": exp.nlt or 0,
                        "step_size": step_size,
                        "start_index": start,
                        "parameters": params,
                        "label": pm.label,
                    },
                    _detect_varying_params(
                        [
                            {
                                "method": method,
                                "n_states": exp.n_states or 0,
                                "nlt": exp.nlt or 0,
                                "step_size": step_size,
                                "start_index": start,
                                "parameters": params,
                                "label": pm.label,
                            }
                        ]
                    ),
                ),
            }
        )

    except Exception as e:
        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status=500
        )


# ═══════════════════════════════════════════════════════════════
# RESULTS GALLERY VIEW
# ═══════════════════════════════════════════════════════════════


def results_gallery(request):
    """Display gallery of all experiment results with glass effect."""

    # Fetch all experiments with related data
    experiments = (
        Experiment.objects.select_related("partition_method", "matrix")
        .prefetch_related("rsd_results")
        .all()
    )

    # Map experiments to result cards
    results = []
    for exp in experiments:
        rsd_dem = exp.rsd_results.filter(source="dem").first()
        rsd_markov = exp.rsd_results.filter(source="markov").first()

        results.append(
            {
                "id": exp.id,
                "name": exp.folder_name[:50],
                "analysis_type": exp.partition_method.get_name_display(),
                "status": "completed" if (rsd_dem and rsd_markov) else "pending",
                "get_status_display": "Complétée"
                if (rsd_dem and rsd_markov)
                else "En cours",
                "image_url": None,  # No image URL for experiments
                "iterations": exp.n_states,
                "accuracy": float(exp.matrix.diagonal_mean * 100) if exp.matrix else 0,
                "duration": f"{exp.nlt} paires",
                "detail_url": f"/markov/experiments/{exp.id}/",
                "download_url": f"/markov/api/download-experiment/{exp.id}/",
            }
        )

    # Count statuses
    total_results = len(results)
    completed_count = sum(1 for r in results if r["status"] == "completed")
    pending_count = sum(1 for r in results if r["status"] == "pending")
    failed_count = 0  # No failed results in our case

    return render(
        request,
        "markov/results_gallery.html",
        {
            "results": results,
            "total_results": total_results,
            "completed_count": completed_count,
            "pending_count": pending_count,
            "failed_count": failed_count,
        },
    )


def image_gallery(request):
    """Display image gallery with glass effect (placeholder for future image storage)."""

    # For now, we'll create a gallery from experiment matrices
    # In future, this could use a dedicated GalleryImage model

    experiments = Experiment.objects.select_related("partition_method", "matrix").all()

    gallery_items = []
    for exp in experiments:
        gallery_items.append(
            {
                "id": exp.id,
                "title": exp.folder_name[:50],
                "description": f"{exp.partition_method.label} - {exp.n_states} états",
                "image_url": f"/markov/api/experiment-thumbnail/{exp.id}/",
                "full_url": f"/markov/experiments/{exp.id}/",
            }
        )

    return render(
        request,
        "markov/image_gallery.html",
        {
            "gallery_items": gallery_items,
            "total_items": len(gallery_items),
        },
    )
