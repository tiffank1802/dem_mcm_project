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
    def n_cells(self):
        """Nombre total d'états."""
        ...

    @property
    @abstractmethod
    def label(self):
        """Identifiant unique (utilisé pour le nom de dossier)."""
        ...

    @abstractmethod
    def fit(self, coordinates):
        """
        Apprend le partitionnement sur des données représentatives.

        Args:
            coordinates: np.ndarray shape (N, 3)
        Returns:
            self
        """
        ...

    @abstractmethod
    def compute_states(self, x, y, z):
        """
        Assigne un indice d'état à chaque particule.

        Args:
            x, y, z: arrays ou Polars Series
        Returns:
            np.ndarray dtype int64
        """
        ...

    def save(self, path):
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
# FACTORY
# =============================================================================

REGISTRY = {
    "cartesian": CartesianPartitioner,
    "cylindrical": CylindricalPartitioner,
    "voronoi": VoronoiPartitioner,
    "quantile": QuantileGridPartitioner,
    "octree": OctreePartitioner,
    "physics": PhysicsAwarePartitioner,
}


def create_partitioner(method, **kwargs):
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