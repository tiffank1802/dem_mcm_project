"""
===================================================================================
PARTITIONERS — Méthodes de partitionnement spatial pour chaînes de Markov
===================================================================================

Interface commune:
    partitioner = create_partitioner("voronoi", n_cells=125)
    partitioner.fit(coordinates)                    # (N, 3) numpy array
    states = partitioner.compute_states(x, y, z)    # → indices int64
    partitioner.save("output/")
    partitioner.load("output/")

Méthodes disponibles:
    cartesian    — grille régulière (x, y, z)
    cylindrical  — grille cylindrique (r, θ, z)
    voronoi      — clustering K-means / cellules de Voronoï
    quantile     — grille avec bords par quantiles (équi-population)
    octree       — octree adaptatif à la densité
    physics      — K-means sur position + champs physiques
===================================================================================
"""

import numpy as np
import os
import json
from abc import ABC, abstractmethod

__all__ = [
 
    "BasePartitioner",
    "CartesianPartitioner",
    "CylindricalPartitioner",
    "VoronoiPartitioner",
    "QuantileGridPartitioner",
    "OctreePartitioner",
    "PhysicsAwarePartitioner",
    "adaptive",   
    "multizone", 
    "single",   
    "create_partitioner",
    "REGISTRY",
]


# =============================================================================
# CLASSE DE BASE
# =============================================================================


class BasePartitioner(ABC):
    """Interface commune pour tous les partitionneurs."""

    @property
    @abstractmethod
    def n_cells(self)-> int:
        """Nombre total d'états."""
        ...

    @property
    @abstractmethod
    def label(self)-> str:
        """Identifiant unique (utilisé pour le nom de dossier)."""
        ...

    @abstractmethod
    def fit(self, coordinates: np.ndarray)->object:
        """
        Apprend le partitionnement sur des données représentatives.

        Args:
            coordinates: np.ndarray shape (N, 3)
        Returns:
            self
        """
        ...

    @abstractmethod
    def compute_states(self, x:np.ndarray, y:np.ndarray, z:np.ndarray)->np.ndarray:
        """
        Assigne un indice d'état à chaque particule.

        Args:
            x, y, z: arrays ou Polars Series
        Returns:
            np.ndarray dtype int64
        """
        ...

    def save(self, path: str):
        """Sauvegarde le partitionneur dans un dossier."""
        os.makedirs(path, # The above code is not valid Python code. It appears to be a comment with
        # the text "ex" followed by multiple pound symbols.
        exist_ok=True)
        meta = {
            "type": type(self).__name__,
            "label": self.label,
            "n_cells": self.n_cells,
        }
        with open(os.path.join(path, "partitioner_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._save_data(path)

    def _save_data(self, path):
        pass

    def load(self, path):
        """Charge le partitionneur depuis un dossier."""
        self._load_data(path)
        return self

    def _load_data(self, path):
        pass

    def diagnostics(self, coordinates):
        """
        Statistiques de population par cellule.

        Args:
            coordinates: np.ndarray (N, 3)
        Returns:
            dict avec min, max, mean, std, n_empty
        """
        states = self.compute_states(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        )
        counts = np.bincount(states, minlength=self.n_cells)
        return {
            "pop_min": int(counts.min()),
            "pop_max": int(counts.max()),
            "pop_mean": float(counts.mean()),
            "pop_std": float(counts.std()),
            "n_empty": int((counts == 0).sum()),
            "n_visited": int((counts > 0).sum()),
            "fraction_visited": float((counts > 0).sum() / self.n_cells),
        }


# =============================================================================
# 1. CARTÉSIEN
# =============================================================================


class CartesianPartitioner(BasePartitioner):
    """
    Grille cartésienne régulière.

    Découpe le domaine en nx × ny × nz cellules de taille égale.
    Simple mais inadapté aux géométries cylindriques (coins vides).
    """

    def __init__(self, nx=5, ny=5, nz=5):
        self.nx, self.ny, self.nz = nx, ny, nz
        self._bounds = None

    @property
    def n_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def label(self)-> str:
        return f"cartesian_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self, coordinates:np.ndarray):
        eps = 0.001
        coordinates=np.asarray(coordinates) # contient les coordonnées [x,y,z] de toutes les particules
        mins = coordinates.min(axis=0) - eps # contient le minimum de [x,y,z]
        maxs = coordinates.max(axis=0) + eps # contient le maximum de [x,y,z]
        self._bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]) # (min_x,max_x,min_y,max_y,min_z,max_z)
        return self

    def compute_states(self, x:np.ndarray, y:np.ndarray, z:np.ndarray)-> int:
        """"Cette fonction permet de determiner l'état de la particule: la partition dans laquelle la particule reside."""
        # convertion des coordonnées en tableaux numpy
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds

        ix = np.clip(
            ((x - xmin) * self.nx / (xmax - xmin)).astype(np.int64), 0, self.nx - 1 # attribut une partition suivant l'axe des abcisses à chacune des particules
            # la fonction clip permet de normaliser la position de la particule dans l'ensemble des partitions
        )
        iy = np.clip(
            ((y - ymin) * self.ny / (ymax - ymin)).astype(np.int64), 0, self.ny - 1
        )
        iz = np.clip(
            ((z - zmin) * self.nz / (zmax - zmin)).astype(np.int64), 0, self.nz - 1
        )
        return ix + iy * self.nx + iz * self.nx * self.ny

    def _save_data(self, path):
        np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self, path):
        self._bounds = tuple(np.load(os.path.join(path, "bounds.npy")))


# =============================================================================
# 2. CYLINDRIQUE
# =============================================================================


class CylindricalPartitioner(BasePartitioner):
    """
    Grille cylindrique (r, θ, z).

    Idéal pour les mélangeurs à symétrie axiale.
    Deux modes radiaux:
      - "equal_dr"  : Δr constant
      - "equal_area": aire de section constante (recommandé)

    Avec ntheta=1 → partitionnement purement axisymétrique.
    """

    def __init__(self, nr=5, ntheta=8, nz=5, radial_mode="equal_area"):
        self.nr = nr
        self.ntheta = ntheta
        self.nz = nz
        self.radial_mode = radial_mode
        self._x_center = None
        self._y_center = None
        self._r_max = None
        self._z_min = None
        self._z_max = None
        self._r_edges = None

    @property
    def n_cells(self):
        return self.nr * self.ntheta * self.nz

    @property
    def label(self):
        return (
            f"cylindrical_nr{self.nr}_nth{self.ntheta}"
            f"_nz{self.nz}_{self.radial_mode}"
        )

    def fit(self, coordinates):
        eps = 0.00
        coordinates=np.asarray(coordinates)
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        self._x_center = (x.min() + x.max()) / 2  # est un scalaire       # position en x du centre de la distribution des particules 
        self._y_center = (y.min() + y.max()) / 2  # est un scalaire       # position en y du centre de la distribution des particules

        r = np.sqrt((x - self._x_center) ** 2 + (y - self._y_center) ** 2) # rayon issue des positions recentrées des particules
        self._r_max = r.max() + eps
        self._z_min = z.min() - eps
        self._z_max = z.max() + eps

        if self.radial_mode == "equal_area":
            # aire π(r_{i+1}² - r_i²) = constante → r_i = R√(i/nr)
            self._r_edges = self._r_max * np.sqrt(np.linspace(0, 1, self.nr + 1)) # construction de la liste des Rayons pour respecter le fait que les surfaces soient identiques
        elif self.radial_mode == "equal_dr":
            self._r_edges = np.linspace(0, self._r_max, self.nr + 1)
        else:
            raise ValueError(f"radial_mode inconnu: {self.radial_mode}")

        return self

    def compute_states(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self._x_center
        dy = y - self._y_center
        # convertit la position des particules du système de coordonnées cartésiens vers le système de coordonnées cylindriques
        r = np.sqrt(dx**2 + dy**2) 
        theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)  # [0, 2π]

        ir = np.clip(
            np.searchsorted(self._r_edges, r, side="right") - 1, # renvoit la liste d'indices  de la liste des partitions(selon le rayon) dans laquelle les rayons des particules ont été insérés 
            # le vecteur que renvoir la fonction searchsorted est de dimension de r (nombres de particules)
            0, self.nr - 1  # les particules sont raménées dans l'intervalle des partitions suivant le rayon
        )
        itheta = np.clip(
            (theta * self.ntheta / (2 * np.pi)).astype(np.int64), 0, self.ntheta - 1 # le cylindre est partionné sur toute la circonference de sa base
            # chaque particule est placée dans une partition en fonction de son angle theta
        )
        dz = (self._z_max - self._z_min) / self.nz
        iz = np.clip(
            ((z - self._z_min) / dz).astype(np.int64), 0, self.nz - 1
        )

        return ir + itheta * self.nr + iz * self.nr * self.ntheta # la numérotation des partitons se fait partant des rayons, puis les angles et enfin les hauteurs z

    def _save_data(self, path):
        params = {
            "x_center": self._x_center,
            "y_center": self._y_center,
            "r_max": self._r_max,
            "z_min": self._z_min,
            "z_max": self._z_max,
        }
        with open(os.path.join(path, "cylindrical_params.json"), "w") as f:
            json.dump(params, f, indent=2)
        np.save(os.path.join(path, "r_edges.npy"), self._r_edges)

    def _load_data(self, path):
        with open(os.path.join(path, "cylindrical_params.json")) as f:
            p = json.load(f)
        self._x_center = p["x_center"]
        self._y_center = p["y_center"]
        self._r_max = p["r_max"]
        self._z_min = p["z_min"]
        self._z_max = p["z_max"]
        self._r_edges = np.load(os.path.join(path, "r_edges.npy"))


# =============================================================================
# 3. VORONOÏ (K-MEANS)
# =============================================================================


class VoronoiPartitioner(BasePartitioner):
    """
    Partitionnement Voronoï par K-means.

    Chaque cellule = le bassin d'attraction du centroïde le plus proche.
    S'adapte naturellement à la densité de particules.

    C'est la méthode de référence en MCM (Fan et al., Doucet et al.).
    """

    def __init__(self, n_cells=125, random_state=42):
        self._n_cells = n_cells
        self.random_state = random_state
        self.centroids = None
        self._tree = None

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def label(self):
        return f"voronoi_{self._n_cells}cells"

    def fit(self, coordinates):
        coordinates=np.asarray(coordinates)
        from sklearn.cluster import MiniBatchKMeans
        from scipy.spatial import cKDTree

        # Sous-échantillonner si trop gros
        rng = np.random.RandomState(self.random_state)
        if len(coordinates) > 500_000:
            idx = rng.choice(len(coordinates), 500_000, replace=False)
            fit_data = coordinates[idx]
        else:
            fit_data = coordinates
        # Création du cluster
        kmeans = MiniBatchKMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state,
            batch_size=min(10_000, len(fit_data)),
            n_init=10,
        )
        kmeans.fit(fit_data) # determination des centres des distributions dans chaque partition
        self.centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self.centroids)
        return self

    def compute_states(self, x, y, z):
        coords = np.column_stack(
            [np.asarray(x), np.asarray(y), np.asarray(z)]
        )
        _, indices = self._tree.query(coords)
        return indices.astype(np.int64)

    def _save_data(self, path):
        np.save(os.path.join(path, "centroids.npy"), self.centroids)

    def _load_data(self, path):
        from scipy.spatial import cKDTree

        self.centroids = np.load(os.path.join(path, "centroids.npy"))
        self._tree = cKDTree(self.centroids)
        self._n_cells = len(self.centroids)


# =============================================================================
# 4. GRILLE PAR QUANTILES (ÉQU-POPULATION)
# =============================================================================


class QuantileGridPartitioner(BasePartitioner):
    """
    Grille dont les bords sont des quantiles des données.
    plus il y aura une concentration de points en un endroit et plus la grille sera grande à cet endroit.

    Chaque cellule contient approximativement le même nombre de particules
    (équi-population marginale sur chaque axe).

    Meilleure homogénéité statistique que la grille cartésienne régulière.
    """

    def __init__(self, nx=5, ny=5, nz=5):
        self.nx, self.ny, self.nz = nx, ny, nz
        self._x_edges = None
        self._y_edges = None
        self._z_edges = None

    @property
    def n_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def label(self):
        return f"quantile_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self, coordinates):
        coordinates=np.asarray(coordinates)
        eps = 0.001
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        # chaque edge est un vecteur de taille self.nx+1 ou self.ny+1 ou self.nz+1 dont chaque indice correspond à la valeur de x correspondant au quantile donné
        self._x_edges = np.quantile(x, np.linspace(0, 1, self.nx + 1))
        self._y_edges = np.quantile(y, np.linspace(0, 1, self.ny + 1))
        self._z_edges = np.quantile(z, np.linspace(0, 1, self.nz + 1))

        # Élargir les bords extrêmes
        self._x_edges[0] -= eps
        self._x_edges[-1] += eps
        self._y_edges[0] -= eps 
        self._y_edges[-1] += eps
        self._z_edges[0] -= eps
        self._z_edges[-1] += eps

        return self

    def compute_states(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        ix = np.clip(
            np.searchsorted(self._x_edges, x, side="right") - 1, 0, self.nx - 1
        )
        iy = np.clip(
            np.searchsorted(self._y_edges, y, side="right") - 1, 0, self.ny - 1
        )
        iz = np.clip(
            np.searchsorted(self._z_edges, z, side="right") - 1, 0, self.nz - 1
        )
        return ix + iy * self.nx + iz * self.nx * self.ny

    def _save_data(self, path):
        np.savez(
            os.path.join(path, "edges.npz"),
            x=self._x_edges,
            y=self._y_edges,
            z=self._z_edges,
        )

    def _load_data(self, path):
        data = np.load(os.path.join(path, "edges.npz"))
        self._x_edges = data["x"]
        self._y_edges = data["y"]
        self._z_edges = data["z"]


# =============================================================================
# 5. OCTREE ADAPTATIF
# =============================================================================


class OctreePartitioner(BasePartitioner):
    """
    Octree adaptatif.

    Subdivise récursivement les cellules contenant plus de max_particles
    particules, jusqu'à max_depth niveaux.

    Avantage : raffine automatiquement les zones denses.
    Inconvénient : nombre de cellules non contrôlé a priori.
    """

    def __init__(self, max_particles=100, max_depth=5):
        self.max_particles = max_particles
        self.max_depth = max_depth
        self._leaves = []  # liste de tuples (xmin, xmax, ymin, ymax, zmin, zmax)
        self._bounds = None

    @property
    def n_cells(self):
        return len(self._leaves) if self._leaves else 0

    @property
    def label(self):
        return f"octree_mp{self.max_particles}_md{self.max_depth}"

    def fit(self, coordinates):
        coordinates=np.asarray(coordinates)
        eps = 0.001
        self._bounds = (
            coordinates[:, 0].min() - eps,
            coordinates[:, 0].max() + eps,
            coordinates[:, 1].min() - eps,
            coordinates[:, 1].max() + eps,
            coordinates[:, 2].min() - eps,
            coordinates[:, 2].max() + eps,
        )
        self._leaves = []
        self._subdivide(coordinates, self._bounds, depth=0)
        return self

    def _subdivide(self, coords, bounds, depth):
        """Subdivision récursive."""
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        n_in = len(coords)

        # Condition d'arrêt
        if n_in <= self.max_particles or depth >= self.max_depth:
            self._leaves.append(bounds)
            return

        # Point de coupe = milieu
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        # Assigner chaque particule à un octant (0-7)
        octant = (
            (coords[:, 0] >= xmid).astype(np.int64)
            + (coords[:, 1] >= ymid).astype(np.int64) * 2
            + (coords[:, 2] >= zmid).astype(np.int64) * 4
        )

        # Récursion sur les 8 enfants
        for idx in range(8):
            ix, iy, iz = idx % 2, (idx // 2) % 2, idx // 4
            child_bounds = (
                xmid if ix else xmin,
                xmax if ix else xmid,
                ymid if iy else ymin,
                ymax if iy else ymid,
                zmid if iz else zmin,
                zmax if iz else zmid,
            )
            child_mask = octant == idx
            self._subdivide(coords[child_mask], child_bounds, depth + 1)

    def compute_states(self, x, y, z):
        coords = np.column_stack(
            [np.asarray(x, dtype=np.float64),
             np.asarray(y, dtype=np.float64),
             np.asarray(z, dtype=np.float64)]
        )
        n = len(coords)
        states = np.full(n, -1, dtype=np.int64)

        # Assignation par bounding box
        for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
            mask = (
                (coords[:, 0] >= xmin) & (coords[:, 0] < xmax)
                & (coords[:, 1] >= ymin) & (coords[:, 1] < ymax)
                & (coords[:, 2] >= zmin) & (coords[:, 2] < zmax)
            )
            states[mask] = cell_id

        # Points non assignés → cellule la plus proche
        unassigned = states == -1
        if unassigned.any():
            from scipy.spatial import cKDTree

            centers = np.array(
                [
                    (
                        (b[0] + b[1]) / 2,
                        (b[2] + b[3]) / 2,
                        (b[4] + b[5]) / 2,
                    )
                    for b in self._leaves
                ]
            )
            tree = cKDTree(centers)
            _, idx = tree.query(coords[unassigned])
            states[unassigned] = idx

        return states

    def _save_data(self, path):
        leaves_arr = np.array(self._leaves)
        np.save(os.path.join(path, "leaves.npy"), leaves_arr)
        if self._bounds:
            np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self, path):
        leaves_arr = np.load(os.path.join(path, "leaves.npy"))
        self._leaves = [tuple(row) for row in leaves_arr]
        bounds_path = os.path.join(path, "bounds.npy")
        if os.path.exists(bounds_path):
            self._bounds = tuple(np.load(bounds_path))


# =============================================================================
# 6. PHYSIQUE-AWARE (POSITION + VITESSE)
# =============================================================================


class PhysicsAwarePartitioner(BasePartitioner):
    """
    K-means sur des features physiques (position + vitesse optionnelle).

    Par défaut, fonctionne sur les positions normalisées (équivalent Voronoï).
    Si des vitesses sont fournies via fit_with_physics(), le clustering
    tient aussi compte de la norme de vitesse.

    Usage avancé:
        part = PhysicsAwarePartitioner(n_cells=125, velocity_weight=0.3)
        part.fit_with_physics(positions, velocities)
        states = part.compute_states_with_physics(x, y, z, vx, vy, vz)
    """

    def __init__(self, n_cells=125, velocity_weight=0.3, random_state=42):
        self._n_cells = n_cells
        self.velocity_weight = velocity_weight
        self.random_state = random_state
        self._centroids = None
        self._tree = None
        self._mean = None
        self._std = None
        self._n_features = 3  # 3 = position seule, 4 = position + vitesse

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def label(self):
        suffix = "withvel" if self._n_features > 3 else "pos"
        return f"physics_{self._n_cells}cells_{suffix}"

    def fit(self, coordinates):
        """Fit sur positions seules (équivalent Voronoï normalisé)."""
        coordinates=np.asarray(coordinates)
        return self._fit_internal(coordinates)

    def fit_with_physics(self, positions, velocities):
        """
        Fit sur positions + norme de vitesse.

        Args:
            positions: (N, 3)
            velocities: (N, 3)
        """
        speed = np.linalg.norm(velocities, axis=1, keepdims=True)
        features = np.hstack([positions, speed * self.velocity_weight])
        self._n_features = 4
        return self._fit_internal(features)

    def _fit_internal(self, features):
        from sklearn.cluster import MiniBatchKMeans
        from scipy.spatial import cKDTree

        self._n_features = features.shape[1]

        # Normalisation
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0

        X = (features - self._mean) / self._std

        # Sous-échantillonner
        rng = np.random.RandomState(self.random_state)
        if len(X) > 500_000:
            idx = rng.choice(len(X), 500_000, replace=False)
            X_fit = X[idx]
        else:
            X_fit = X

        kmeans = MiniBatchKMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state,
            batch_size=min(10_000, len(X_fit)),
            n_init=10,
        )
        kmeans.fit(X_fit)
        self._centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self._centroids)
        return self

    def compute_states(self, x, y, z):
        """Assigne les états (position seule, vitesse=0 si fitté avec)."""
        coords = np.column_stack(
            [np.asarray(x), np.asarray(y), np.asarray(z)]
        )
        if self._n_features > 3:
            padding = np.zeros((len(coords), self._n_features - 3))
            coords = np.hstack([coords, padding])

        X = (coords - self._mean) / self._std
        _, indices = self._tree.query(X)
        return indices.astype(np.int64)

    def compute_states_with_physics(self, x, y, z, vx, vy, vz):
        """Assigne les états avec vitesse."""
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
        vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        features = np.hstack([pos, speed * self.velocity_weight])

        X = (features - self._mean) / self._std
        _, indices = self._tree.query(X)
        return indices.astype(np.int64)

    def _save_data(self, path):
        np.save(os.path.join(path, "centroids.npy"), self._centroids)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "physics_params.json"), "w") as f:
            json.dump({"n_features": self._n_features}, f)

    def _load_data(self, path):
        from scipy.spatial import cKDTree

        self._centroids = np.load(os.path.join(path, "centroids.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._centroids)
        self._n_cells = len(self._centroids)
        with open(os.path.join(path, "physics_params.json")) as f:
            self._n_features = json.load(f)["n_features"]


# =============================================================================
# 7. PARTITIONNEMENT ADAPTATIF HAUT/BAS
# =============================================================================


class AdaptiveZPartitioner(BasePartitioner):
    """
    Partitionnement adaptatif en z.
    
    Divise le domaine en deux zones:
      - Zone haute (z > z_split): peu de cellules (grossier)
      - Zone basse (z ≤ z_split): partitionnement fin
    
    Utile pour les mélangeurs où la partie haute est moins intéressante
    (zone de chute libre, espace vide, etc.)
    
    Args:
        z_split: altitude de séparation (ou quantile si z_split_mode="quantile")
        z_split_mode: "absolute" ou "quantile" (ex: 0.7 = 70% des particules en dessous)
        n_cells_top: nombre de cellules pour la zone haute (défaut=1)
        bottom_method: méthode de partitionnement pour la zone basse
        bottom_kwargs: arguments pour le partitionneur du bas
    
    Exemple:
        # Zone haute = 1 cellule, zone basse = grille cylindrique fine
        part = AdaptiveZPartitioner(
            z_split_mode="quantile",
            z_split=0.8,           # 80% des particules en bas
            n_cells_top=1,
            bottom_method="cylindrical",
            bottom_kwargs={"nr": 5, "ntheta": 8, "nz": 10}
        )
    """
    
    def __init__(
        self,
        z_split: float = None,
        z_split_mode: str = "quantile",  # "absolute" ou "quantile"
        n_cells_top: int = 1,
        top_method: str = "single",      # "single", "cartesian", "cylindrical"
        top_kwargs: dict = None,
        bottom_method: str = "cylindrical",
        bottom_kwargs: dict = None,
    ):
        self.z_split_input = z_split
        self.z_split_mode = z_split_mode
        self.n_cells_top_target = n_cells_top
        self.top_method = top_method
        self.top_kwargs = top_kwargs or {}
        self.bottom_method = bottom_method
        self.bottom_kwargs = bottom_kwargs or {}
        
        # Calculés au fit
        self._z_split = None
        self._z_min = None
        self._z_max = None
        self._top_partitioner = None
        self._bottom_partitioner = None
        self._n_cells_top = None
        self._n_cells_bottom = None
    
    @property
    def n_cells(self):
        if self._n_cells_top is None or self._n_cells_bottom is None:
            return 0
        return self._n_cells_top + self._n_cells_bottom
    
    @property
    def n_cells_top(self):
        return self._n_cells_top
    
    @property
    def n_cells_bottom(self):
        return self._n_cells_bottom
    
    @property
    def label(self):
        return (
            f"adaptive_z_{self.bottom_method}"
            f"_top{self._n_cells_top}_bot{self._n_cells_bottom}_split{self.z_split_input}_mode_split{self.z_split_mode}_m_bot{self.bottom_method}_n_top{self.n_cells_top}"
        )
    
    def fit(self, coordinates: np.ndarray):
        coordinates = np.asarray(coordinates)
        z = coordinates[:, 2]
        
        self._z_min = z.min()
        self._z_max = z.max()
        
        # ── Déterminer z_split ──
        if self.z_split_mode == "quantile":
            quantile = self.z_split_input if self.z_split_input else 0.7
            self._z_split = np.quantile(z, quantile)
        elif self.z_split_mode == "absolute":
            if self.z_split_input is None:
                # Par défaut : milieu
                self._z_split = (self._z_min + self._z_max) / 2
            else:
                self._z_split = self.z_split_input
        else:
            raise ValueError(f"z_split_mode inconnu: {self.z_split_mode}")
        
        # ── Séparer les données ──
        mask_bottom = z <= self._z_split
        mask_top = z > self._z_split
        
        coords_bottom = coordinates[mask_bottom]
        coords_top = coordinates[mask_top]
        
        n_bottom = len(coords_bottom)
        n_top = len(coords_top)
        
        print(f"   📊 Split z = {self._z_split:.4f}")
        print(f"      Zone basse: {n_bottom} particules ({100*n_bottom/(n_bottom+n_top):.1f}%)")
        print(f"      Zone haute: {n_top} particules ({100*n_top/(n_bottom+n_top):.1f}%)")
        
        # ── Fit zone basse ──
        self._bottom_partitioner = create_partitioner(
            self.bottom_method, **self.bottom_kwargs
        )
        if len(coords_bottom) > 0:
            self._bottom_partitioner.fit(coords_bottom)
        self._n_cells_bottom = self._bottom_partitioner.n_cells
        
        # ── Fit zone haute ──
        if self.top_method == "single":
            # Une seule cellule pour toute la zone haute
            self._top_partitioner = None
            self._n_cells_top = 1
        else:
            self._top_partitioner = create_partitioner(
                self.top_method, **self.top_kwargs
            )
            if len(coords_top) > 0:
                self._top_partitioner.fit(coords_top)
            self._n_cells_top = self._top_partitioner.n_cells
        
        print(f"      Cellules bas: {self._n_cells_bottom}, haut: {self._n_cells_top}")
        print(f"      Total: {self.n_cells} cellules")
        
        return self
    
    def compute_states(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        
        n = len(x)
        states = np.zeros(n, dtype=np.int64)
        
        mask_bottom = z <= self._z_split
        mask_top = ~mask_bottom
        
        # ── Zone basse : états 0 à n_cells_bottom-1 ──
        if mask_bottom.any():
            states[mask_bottom] = self._bottom_partitioner.compute_states(
                x[mask_bottom], y[mask_bottom], z[mask_bottom]
            )
        
        # ── Zone haute : états n_cells_bottom à n_cells-1 ──
        if mask_top.any():
            if self._top_partitioner is None:
                # Une seule cellule
                states[mask_top] = self._n_cells_bottom
            else:
                top_states = self._top_partitioner.compute_states(
                    x[mask_top], y[mask_top], z[mask_top]
                )
                states[mask_top] = top_states + self._n_cells_bottom
        
        return states
    
    def _save_data(self, path):
        params = {
            "z_split": self._z_split,
            "z_min": self._z_min,
            "z_max": self._z_max,
            "n_cells_top": self._n_cells_top,
            "n_cells_bottom": self._n_cells_bottom,
            "top_method": self.top_method,
            "bottom_method": self.bottom_method,
            "top_kwargs": self.top_kwargs,
            "bottom_kwargs": self.bottom_kwargs,
        }
        with open(os.path.join(path, "adaptive_params.json"), "w") as f:
            json.dump(params, f, indent=2)
        
        # Sauvegarder les sous-partitionneurs
        bottom_path = os.path.join(path, "bottom")
        self._bottom_partitioner.save(bottom_path)
        
        if self._top_partitioner is not None:
            top_path = os.path.join(path, "top")
            self._top_partitioner.save(top_path)
    
    def _load_data(self, path):
        with open(os.path.join(path, "adaptive_params.json")) as f:
            params = json.load(f)
        
        self._z_split = params["z_split"]
        self._z_min = params["z_min"]
        self._z_max = params["z_max"]
        self._n_cells_top = params["n_cells_top"]
        self._n_cells_bottom = params["n_cells_bottom"]
        self.top_method = params["top_method"]
        self.bottom_method = params["bottom_method"]
        self.top_kwargs = params.get("top_kwargs", {})
        self.bottom_kwargs = params.get("bottom_kwargs", {})
        
        # Charger le partitionneur du bas
        bottom_path = os.path.join(path, "bottom")
        self._bottom_partitioner = create_partitioner(
            self.bottom_method, **self.bottom_kwargs
        )
        self._bottom_partitioner.load(bottom_path)
        
        # Charger le partitionneur du haut (si existe)
        top_path = os.path.join(path, "top")
        if self.top_method != "single" and os.path.exists(top_path):
            self._top_partitioner = create_partitioner(
                self.top_method, **self.top_kwargs
            )
            self._top_partitioner.load(top_path)
        else:
            self._top_partitioner = None
    
    def diagnostics(self, coordinates):
        """Diagnostics étendus avec stats par zone."""
        base_diag = super().diagnostics(coordinates)
        
        z = coordinates[:, 2]
        mask_bottom = z <= self._z_split
        
        # Stats zone basse
        if mask_bottom.any():
            bottom_diag = self._bottom_partitioner.diagnostics(
                coordinates[mask_bottom]
            )
        else:
            bottom_diag = {}
        
        base_diag["bottom_stats"] = bottom_diag
        base_diag["z_split"] = self._z_split
        base_diag["fraction_in_bottom"] = float(mask_bottom.mean())
        
        return base_diag
    # À ajouter dans la classe AdaptiveZPartitioner

    def visualize_profile(self, size=700):
        """
        Visualise le partitionnement adaptatif en vue de profil (cylindre vu de côté).
        
        Détecte automatiquement les méthodes utilisées dans les zones haute/basse
        et adapte le rendu en conséquence.
        
        Args:
            size: taille du canvas en pixels
        
        Returns:
            HTML object pour affichage Jupyter
        """
        import uuid
        from IPython.display import HTML
        
        if self._z_split is None:
            raise ValueError("❌ Le partitionneur doit être fitté avant visualisation (appeler .fit())")
        
        cid = f"adaptive_viz_{uuid.uuid4().hex}"
        
        # Normaliser z_split entre 0 et 1
        z_range = self._z_max - self._z_min
        z_split_norm = (self._z_split - self._z_min) / z_range if z_range > 0 else 0.5
        
        # Préparer les données de visualisation
        viz_data = self._prepare_visualization_data()
        
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <h3 style="margin-bottom:12px; color:#2c3e50;">
            🔍 Vue de profil — Partitionnement adaptatif
        </h3>
        
        <div style="margin:12px 0; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius:8px; color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div>
                    <div style="font-size:13px; opacity:0.9; margin-bottom:4px;">Zone basse (z ≤ {self._z_split:.4f})</div>
                    <div style="font-size:18px; font-weight:bold;">{self.bottom_method}</div>
                    <div style="font-size:14px; margin-top:4px;">{self._n_cells_bottom} cellules</div>
                </div>
                <div>
                    <div style="font-size:13px; opacity:0.9; margin-bottom:4px;">Zone haute (z > {self._z_split:.4f})</div>
                    <div style="font-size:18px; font-weight:bold;">{self.top_method}</div>
                    <div style="font-size:14px; margin-top:4px;">{self._n_cells_top} cellule(s)</div>
                </div>
            </div>
            <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.3); 
                        text-align:center; font-size:16px; font-weight:bold;">
                Total : {self.n_cells} cellules
            </div>
        </div>
        
        <canvas id="{cid}" width="{size}" height="{size}"
                style="border:2px solid #bdc3c7; border-radius:10px; 
                    box-shadow:0 8px 16px rgba(0,0,0,0.15); display:block; margin:20px auto;
                    background:white;"></canvas>
        
        <div style="margin-top:12px; padding:12px; background:#ecf0f1; border-radius:6px; font-size:13px; color:#34495e;">
            <strong>💡 Légende :</strong>
            <ul style="margin:8px 0; padding-left:20px;">
                <li>Les couleurs représentent les différents états (cellules)</li>
                <li>La ligne rouge en pointillés marque la séparation z_split</li>
                <li>Les numéros indiquent l'identifiant de chaque état (0 à {self.n_cells-1})</li>
            </ul>
        </div>
        
        </div>
        
        <script>
        (function() {{
            const canvas = document.getElementById("{cid}");
            const ctx = canvas.getContext("2d");
            
            const W = canvas.width;
            const H = canvas.height;
            const cx = W / 2;
            const cy = H / 2;
            const R = Math.min(W, H) * 0.38;
            
            const zMin = {self._z_min};
            const zMax = {self._z_max};
            const zSplit = {self._z_split};
            const zSplitNorm = {z_split_norm};
            
            const vizData = {json.dumps(viz_data)};
            
            // ═══════════════════════════════════════════════════════════
            // UTILITAIRES
            // ═══════════════════════════════════════════════════════════
            
            function zToCanvas(z) {{
                const zNorm = (z - zMin) / (zMax - zMin);
                return cy + R * (1 - 2 * zNorm);
            }}
            
            function colorForState(state, total) {{
                const hue = (state * 360) / Math.max(total, 1);
                return `hsla(${{hue}}, 75%, 62%, 0.88)`;
            }}
            
            function darkenColor(state, total) {{
                const hue = (state * 360) / Math.max(total, 1);
                return `hsla(${{hue}}, 75%, 40%, 1)`;
            }}
            
            // ═══════════════════════════════════════════════════════════
            // DESSIN
            // ═══════════════════════════════════════════════════════════
            
            ctx.clearRect(0, 0, W, H);
            
            // Fond du cylindre
            ctx.beginPath();
            ctx.arc(cx, cy, R, 0, 2*Math.PI);
            ctx.fillStyle = "#f8f9fa";
            ctx.fill();
            
            // Clip en forme de cercle
            ctx.save();
            ctx.beginPath();
            ctx.arc(cx, cy, R, 0, 2*Math.PI);
            ctx.clip();
            
            // Dessiner toutes les cellules
            vizData.cells.forEach(cell => {{
                const x = cx + cell.x * R;
                const y = cy - cell.y * R;
                const w = cell.w * R;
                const h = cell.h * R;
                
                // Remplissage
                ctx.fillStyle = colorForState(cell.state, vizData.total_cells);
                ctx.fillRect(x - w/2, y - h/2, w, h);
                
                // Bordure
                ctx.strokeStyle = darkenColor(cell.state, vizData.total_cells);
                ctx.lineWidth = 1.2;
                ctx.strokeRect(x - w/2, y - h/2, w, h);
                
                // Label (numéro d'état)
                if (w > 15 && h > 15) {{
                    ctx.fillStyle = "#000";
                    ctx.font = "bold 11px monospace";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(String(cell.state), x, y);
                }}
            }});
            
            ctx.restore();
            
            // Ligne de séparation z_split
            const ySplit = zToCanvas(zSplit);
            ctx.beginPath();
            ctx.moveTo(cx - R, ySplit);
            ctx.lineTo(cx + R, ySplit);
            ctx.setLineDash([8, 5]);
            ctx.strokeStyle = "#e74c3c";
            ctx.lineWidth = 3;
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Label z_split
            ctx.fillStyle = "#e74c3c";
            ctx.font = "bold 13px sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(`z_split = ${{zSplit.toFixed(4)}}`, cx - R + 15, ySplit - 12);
            
            // Contour du cylindre
            ctx.beginPath();
            ctx.arc(cx, cy, R, 0, 2*Math.PI);
            ctx.strokeStyle = "#2c3e50";
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Annotations zones
            ctx.fillStyle = "#34495e";
            ctx.font = "14px sans-serif";
            ctx.textAlign = "center";
            
            const yTopLabel = cy - R * 0.72;
            const yBotLabel = cy + R * 0.72;
            
            ctx.fillText(`Zone haute : ${{vizData.top_method}}`, cx, yTopLabel);
            ctx.fillText(`(${{vizData.n_cells_top}} cellule(s))`, cx, yTopLabel + 16);
            
            ctx.fillText(`Zone basse : ${{vizData.bottom_method}}`, cx, yBotLabel);
            ctx.fillText(`(${{vizData.n_cells_bottom}} cellules)`, cx, yBotLabel + 16);
            
            // Flèche z
            ctx.strokeStyle = "#7f8c8d";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx + R + 30, cy + R);
            ctx.lineTo(cx + R + 30, cy - R);
            ctx.stroke();
            
            // Pointe de flèche
            ctx.beginPath();
            ctx.moveTo(cx + R + 30, cy - R);
            ctx.lineTo(cx + R + 25, cy - R + 8);
            ctx.lineTo(cx + R + 35, cy - R + 8);
            ctx.closePath();
            ctx.fillStyle = "#7f8c8d";
            ctx.fill();
            
            ctx.fillStyle = "#7f8c8d";
            ctx.font = "italic 13px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("z", cx + R + 30, cy - R - 12);
            
        }})();
        </script>
        """
        
        return HTML(html)


    def _prepare_visualization_data(self):
        """
        Prépare les données de visualisation adaptées à chaque méthode.
        
        Returns:
            dict: données pour le rendu JavaScript
        """
        z_range = self._z_max - self._z_min
        z_split_norm = (self._z_split - self._z_min) / z_range if z_range > 0 else 0.5
        
        data = {
            "total_cells": self.n_cells,
            "n_cells_top": self._n_cells_top,
            "n_cells_bottom": self._n_cells_bottom,
            "top_method": self.top_method,
            "bottom_method": self.bottom_method,
            "cells": []
        }
        
        # ══════════════════════════════════════════════════════════════
        # ZONE BASSE
        # ══════════════════════════════════════════════════════════════
        
        if self.bottom_method == "cylindrical":
            data["cells"].extend(
                self._render_cylindrical(
                    self._bottom_partitioner,
                    z_min=0,
                    z_max=z_split_norm,
                    state_offset=0
                )
            )
        elif self.bottom_method == "cartesian":
            data["cells"].extend(
                self._render_cartesian(
                    self._bottom_partitioner,
                    z_min=0,
                    z_max=z_split_norm,
                    state_offset=0
                )
            )
        elif self.bottom_method == "voronoi":
            data["cells"].extend(
                self._render_voronoi(
                    self._bottom_partitioner,
                    z_min=0,
                    z_max=z_split_norm,
                    state_offset=0
                )
            )
        elif self.bottom_method == "quantile":
            data["cells"].extend(
                self._render_quantile(
                    self._bottom_partitioner,
                    z_min=0,
                    z_max=z_split_norm,
                    state_offset=0
                )
            )
        elif self.bottom_method == "single":
            data["cells"].append({
                "state": 0,
                "x": 0,
                "y": z_split_norm / 2,
                "w": 2,
                "h": z_split_norm
            })
        else:
            # Rendu générique
            data["cells"].extend(
                self._render_generic(
                    self._bottom_partitioner,
                    z_min=0,
                    z_max=z_split_norm,
                    state_offset=0
                )
            )
        
        # ══════════════════════════════════════════════════════════════
        # ZONE HAUTE
        # ══════════════════════════════════════════════════════════════
        
        if self.top_method == "single" or self._top_partitioner is None:
            data["cells"].append({
                "state": self._n_cells_bottom,
                "x": 0,
                "y": (z_split_norm + 1) / 2,
                "w": 2,
                "h": (1 - z_split_norm)
            })
        elif self.top_method == "cylindrical":
            data["cells"].extend(
                self._render_cylindrical(
                    self._top_partitioner,
                    z_min=z_split_norm,
                    z_max=1,
                    state_offset=self._n_cells_bottom
                )
            )
        elif self.top_method == "cartesian":
            data["cells"].extend(
                self._render_cartesian(
                    self._top_partitioner,
                    z_min=z_split_norm,
                    z_max=1,
                    state_offset=self._n_cells_bottom
                )
            )
        else:
            data["cells"].extend(
                self._render_generic(
                    self._top_partitioner,
                    z_min=z_split_norm,
                    z_max=1,
                    state_offset=self._n_cells_bottom
                )
            )
        
        return data


    # ═══════════════════════════════════════════════════════════════════
    # MÉTHODES DE RENDU PAR TYPE DE PARTITIONNEUR
    # ═══════════════════════════════════════════════════════════════════

    def _render_cylindrical(self, part, z_min, z_max, state_offset):
        """Rendu d'un partitionneur cylindrique en vue de profil."""
        cells = []
        
        nr, ntheta, nz = part.nr, part.ntheta, part.nz
        z_edges = np.linspace(z_min, z_max, nz + 1)
        
        for iz in range(nz):
            z_mid = (z_edges[iz] + z_edges[iz + 1]) / 2
            z_h = z_edges[iz + 1] - z_edges[iz]
            
            for itheta in range(ntheta):
                theta_mid = (itheta + 0.5) * 2 * np.pi / ntheta
                
                for ir in range(nr):
                    # Rayons
                    if part.radial_mode == "equal_area":
                        r_inner = np.sqrt(ir / nr)
                        r_outer = np.sqrt((ir + 1) / nr)
                    else:
                        r_inner = ir / nr
                        r_outer = (ir + 1) / nr
                    
                    r_mid = (r_inner + r_outer) / 2
                    
                    # Projection : vue de profil du cylindre
                    x = r_mid * np.cos(theta_mid)
                    
                    # Largeur radiale projetée
                    w = (r_outer - r_inner) * 2
                    
                    state = ir + itheta * nr + iz * nr * ntheta
                    
                    cells.append({
                        "state": state + state_offset,
                        "x": x,
                        "y": z_mid,
                        "w": w,
                        "h": z_h
                    })
        
        return cells


    def _render_cartesian(self, part, z_min, z_max, state_offset):
        """Rendu d'un partitionneur cartésien en vue de profil."""
        cells = []
        
        nx, ny, nz = part.nx, part.ny, part.nz
        
        x_edges = np.linspace(-1, 1, nx + 1)
        z_edges = np.linspace(z_min, z_max, nz + 1)
        
        # En vue de profil, on prend une "tranche" en y=0
        for iz in range(nz):
            z_mid = (z_edges[iz] + z_edges[iz + 1]) / 2
            z_h = z_edges[iz + 1] - z_edges[iz]
            
            for ix in range(nx):
                x_mid = (x_edges[ix] + x_edges[ix + 1]) / 2
                x_w = x_edges[ix + 1] - x_edges[ix]
                
                # On prend la tranche centrale en y
                iy = ny // 2
                state = ix + iy * nx + iz * nx * ny
                
                cells.append({
                    "state": state + state_offset,
                    "x": x_mid,
                    "y": z_mid,
                    "w": x_w,
                    "h": z_h
                })
        
        return cells


    def _render_voronoi(self, part, z_min, z_max, state_offset):
        """Rendu d'un partitionneur Voronoi en vue de profil."""
        cells = []
        
        centers = part.centers
        avg_size = 2.0 / np.sqrt(len(centers))
        
        for i, (x, y, z) in enumerate(centers):
            # Normaliser z
            z_norm = (z - self._z_min) / (self._z_max - self._z_min) if self._z_max > self._z_min else 0.5
            
            # Vérifier si dans la zone
            if z_min <= z_norm <= z_max:
                # Projection radiale pour vue de profil
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                x_proj = r * np.cos(theta)
                
                cells.append({
                    "state": i + state_offset,
                    "x": x_proj,
                    "y": z_norm,
                    "w": avg_size,
                    "h": avg_size
                })
        
        return cells


    def _render_quantile(self, part, z_min, z_max, state_offset):
        """Rendu d'un partitionneur par quantiles (similaire au cartésien)."""
        return self._render_cartesian(part, z_min, z_max, state_offset)


    def _render_generic(self, part, z_min, z_max, state_offset):
        """Rendu générique pour méthodes inconnues."""
        cells = []
        n = part.n_cells
        n_cols = int(np.ceil(np.sqrt(n)))
        
        cell_w = 2.0 / n_cols
        cell_h = (z_max - z_min) / n_cols
        
        for i in range(n):
            ix = i % n_cols
            iy = i // n_cols
            
            cells.append({
                "state": i + state_offset,
                "x": -1 + cell_w * (ix + 0.5),
                "y": z_min + cell_h * (iy + 0.5),
                "w": cell_w,
                "h": cell_h
            })
        
        return cells


# =============================================================================
# 8. PARTITIONNEMENT MULTI-ZONES (généralisation)
# =============================================================================


class MultiZonePartitioner(BasePartitioner):
    """
    Partitionnement multi-zones généralisé.
    
    Permet de définir plusieurs zones avec des partitionnements différents.
    Plus flexible que AdaptiveZPartitioner.
    
    Args:
        zones: liste de dicts définissant chaque zone
            [
                {"z_min": -inf, "z_max": 0.5, "method": "cylindrical", "kwargs": {...}},
                {"z_min": 0.5, "z_max": 0.8, "method": "voronoi", "kwargs": {"n_cells": 50}},
                {"z_min": 0.8, "z_max": inf, "method": "single", "kwargs": {}},
            ]
        z_mode: "absolute" ou "quantile"
    """
    
    def __init__(self, zones: list, z_mode: str = "absolute"):
        self.zones_config = zones
        self.z_mode = z_mode
        self._zones = []  # [(z_min, z_max, partitioner), ...]
        self._cell_offsets = []
        self._total_cells = 0
    
    @property
    def n_cells(self):
        return self._total_cells
    
    @property
    def label(self):
        methods = "_".join(z["method"] for z in self.zones_config)
        return f"multizone_{len(self.zones_config)}zones_{methods}"
    
    def fit(self, coordinates):
        coordinates = np.asarray(coordinates)
        z = coordinates[:, 2]
        
        self._zones = []
        self._cell_offsets = [0]
        
        for i, zone_cfg in enumerate(self.zones_config):
            # Convertir les bornes si mode quantile
            if self.z_mode == "quantile":
                z_min = np.quantile(z, zone_cfg.get("z_min", 0))
                z_max = np.quantile(z, zone_cfg.get("z_max", 1))
            else:
                z_min = zone_cfg.get("z_min", z.min())
                z_max = zone_cfg.get("z_max", z.max())
            
            # Sélectionner les particules de cette zone
            mask = (z >= z_min) & (z < z_max)
            if i == len(self.zones_config) - 1:
                mask = (z >= z_min) & (z <= z_max)  # inclure le max pour la dernière
            
            coords_zone = coordinates[mask]
            
            method = zone_cfg.get("method", "single")
            kwargs = zone_cfg.get("kwargs", {})
            
            if method == "single":
                partitioner = SingleCellPartitioner()
            else:
                partitioner = create_partitioner(method, **kwargs)
            
            if len(coords_zone) > 0:
                partitioner.fit(coords_zone)
            
            self._zones.append((z_min, z_max, partitioner))
            self._cell_offsets.append(
                self._cell_offsets[-1] + partitioner.n_cells
            )
            
            print(f"   Zone {i}: z ∈ [{z_min:.3f}, {z_max:.3f}], "
                  f"{partitioner.n_cells} cellules, {len(coords_zone)} particules")
        
        self._total_cells = self._cell_offsets[-1]
        print(f"   Total: {self._total_cells} cellules")
        
        return self
    
    def compute_states(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        
        n = len(x)
        states = np.zeros(n, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        
        for i, (z_min, z_max, partitioner) in enumerate(self._zones):
            if i == len(self._zones) - 1:
                mask = (z >= z_min) & (z <= z_max) & ~assigned
            else:
                mask = (z >= z_min) & (z < z_max) & ~assigned
            
            if mask.any():
                zone_states = partitioner.compute_states(
                    x[mask], y[mask], z[mask]
                )
                states[mask] = zone_states + self._cell_offsets[i]
                assigned[mask] = True
        
        return states
    
    def _save_data(self, path):
        config = {
            "zones_config": self.zones_config,
            "z_mode": self.z_mode,
            "cell_offsets": self._cell_offsets,
            "zones_bounds": [(z_min, z_max) for z_min, z_max, _ in self._zones],
        }
        with open(os.path.join(path, "multizone_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        for i, (_, _, partitioner) in enumerate(self._zones):
            zone_path = os.path.join(path, f"zone_{i}")
            partitioner.save(zone_path)
    
    def _load_data(self, path):
        with open(os.path.join(path, "multizone_config.json")) as f:
            config = json.load(f)
        
        self.zones_config = config["zones_config"]
        self.z_mode = config["z_mode"]
        self._cell_offsets = config["cell_offsets"]
        self._total_cells = self._cell_offsets[-1]
        
        self._zones = []
        for i, (zone_cfg, bounds) in enumerate(
            zip(self.zones_config, config["zones_bounds"])
        ):
            z_min, z_max = bounds
            method = zone_cfg.get("method", "single")
            kwargs = zone_cfg.get("kwargs", {})
            
            if method == "single":
                partitioner = SingleCellPartitioner()
            else:
                partitioner = create_partitioner(method, **kwargs)
            
            zone_path = os.path.join(path, f"zone_{i}")
            partitioner.load(zone_path)
            
            self._zones.append((z_min, z_max, partitioner))


class SingleCellPartitioner(BasePartitioner):
    """Une seule cellule pour tout le domaine."""
    
    @property
    def n_cells(self):
        return 1
    
    @property
    def label(self):
        return "single_cell"
    
    def fit(self, coordinates):
        return self
    
    def compute_states(self, x, y, z):
        return np.zeros(len(np.asarray(x)), dtype=np.int64)


# =============================================================================
# MISE À JOUR DU REGISTRY
# =============================================================================

REGISTRY = {
    "cartesian": CartesianPartitioner,
    "cylindrical": CylindricalPartitioner,
    "voronoi": VoronoiPartitioner,
    "quantile": QuantileGridPartitioner,
    "octree": OctreePartitioner,
    "physics": PhysicsAwarePartitioner,
    "adaptive": AdaptiveZPartitioner,      # ← nouveau
    "multizone": MultiZonePartitioner,     # ← nouveau
    "single": SingleCellPartitioner,       # ← nouveau
}


# =============================================================================
# FACTORY
# =============================================================================

# REGISTRY = {
#     "cartesian": CartesianPartitioner,
#     "cylindrical": CylindricalPartitioner,
#     "voronoi": VoronoiPartitioner,
#     "quantile": QuantileGridPartitioner,
#     "octree": OctreePartitioner,
#     "physics": PhysicsAwarePartitioner,
# }


def create_partitioner(method:str, **kwargs)-> BasePartitioner:
    """
    Crée un partitionneur.

    Args:
        method: "cartesian", "cylindrical", "voronoi", "quantile",
                "octree", "physics"
        **kwargs: arguments passés au constructeur

    Returns:
        instance de BasePartitioner

    Exemple:
        p = create_partitioner("voronoi", n_cells=125)
        p = create_partitioner("cylindrical", nr=5, ntheta=8, nz=5)
    """
    if method not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Méthode inconnue: '{method}'. Disponibles: {available}")
    return REGISTRY[method](**kwargs) # crée une instance de la classe de partitionnement souhaité