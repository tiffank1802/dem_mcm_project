"""
===================================================================================
VISUALISATION 3D DU MÉLANGEUR + DÉCOUPAGE SPATIAL
===================================================================================

Visualise les particules colorées par cellule pour chaque type de partitionnement.

Usage:
    python visualize_partitioning.py

Depuis un notebook:
    from visualize_partitioning import PartitionVisualizer
    viz = PartitionVisualizer()
    viz.load_particles()
    viz.show_all_methods()
===================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import io
import polars as pl
from huggingface_hub import HfFileSystem

# Import des partitionneurs
from .partitioners import (
    create_partitioner,
    CartesianPartitioner,
    CylindricalPartitioner,
    VoronoiPartitioner,
    QuantileGridPartitioner,
    OctreePartitioner,
    PhysicsAwarePartitioner,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================


class PartitionVisualizer:
    """Visualise le mélangeur avec les différents découpages."""

    def __init__(self):
        self.fs = HfFileSystem()
        self.coords = None  # (N, 3) array des positions
        self.files = None

    # ─────────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────────────

    def load_particles(self, file_index=100, sample_every=1):
        """
        Charge les positions des particules depuis un fichier du bucket.

        Args:
            file_index: index du fichier à charger ou encore pas de temps auquel on veut afficher l'état du système
            sample_every: sous-échantillonnage (1 = toutes les particules)
        """
        self.files = sorted(self.fs.glob(f"{HF_FOLDER}/*.csv"))
        print(f"📁 {len(self.files)} fichiers disponibles")

        fname = self.files[file_index]
        print(f"📂 Chargement: {fname.split('/')[-1]}")

        with self.fs.open(fname, "rb") as f:
            df = pl.read_csv(f)

        self.coords = np.column_stack(
            [
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ]
        )[::sample_every]

        print(f"   {len(self.coords)} particules chargées")
        print(f"   X: [{self.coords[:, 0].min():.4f}, {self.coords[:, 0].max():.4f}]")
        print(f"   Y: [{self.coords[:, 1].min():.4f}, {self.coords[:, 1].max():.4f}]")
        print(f"   Z: [{self.coords[:, 2].min():.4f}, {self.coords[:, 2].max():.4f}]")

    def load_multiple_snapshots(self, indices=None, sample_every=5):
        """Charge plusieurs snapshots pour un fit plus représentatif."""
        if indices is None:
            indices = list(range(0, len(self.files), 50))[:20]

        all_coords = []
        for idx in indices:
            with self.fs.open(self.files[idx], "rb") as f:
                df = pl.read_csv(f)
            coords = np.column_stack(
                [
                    df["coordinates:0"].to_numpy(),
                    df["coordinates:1"].to_numpy(),
                    df["coordinates:2"].to_numpy(),
                ]
            )
            all_coords.append(coords[::sample_every])

        return np.vstack(all_coords)

    # ─────────────────────────────────────────────────────────────────
    # CRÉATION DES PARTITIONNEURS
    # ─────────────────────────────────────────────────────────────────

    def get_default_partitioners(self):
        """Retourne un dict de partitionneurs avec des paramètres comparables (~125 états)."""
        return {
            "Cartésien\n(5×5×5 = 125)": {
                "method": "cartesian",
                "kwargs": {"nx": 5, "ny": 5, "nz": 5},
            },
            "Cylindrique equal_dr\n(5r × 5θ × 5z = 125)": {
                "method": "cylindrical",
                "kwargs": {"nr": 5, "ntheta": 5, "nz": 5, "radial_mode": "equal_dr"},
            },
            "Cylindrique equal_area\n(5r × 5θ × 5z = 125)": {
                "method": "cylindrical",
                "kwargs": {"nr": 5, "ntheta": 5, "nz": 5, "radial_mode": "equal_area"},
            },
            "Voronoï (K-means)\n(125 cellules)": {
                "method": "voronoi",
                "kwargs": {"n_cells": 125},
            },
            "Quantile (équi-pop)\n(5×5×5 = 125)": {
                "method": "quantile",
                "kwargs": {"nx": 5, "ny": 5, "nz": 5},
            },
            "Octree adaptatif\n(max_part=50, depth=4)": {
                "method": "octree",
                "kwargs": {"max_particles": 50, "max_depth": 4},
            },
            "Adaptatif Z\n70% bas (cylindrique)": {
                "method": "adaptive",
                "kwargs": {
                    "z_split_mode": "quantile",
                    "z_split": 0.7,  # 70% des particules en bas
                    "n_cells_top": 1,
                    "top_method": "single",
                    "bottom_method": "cylindrical",
                    "bottom_kwargs": {
                        "nr": 5,
                        "ntheta": 5,
                        "nz": 10,
                        "radial_mode": "equal_area",
                    },
                },
            },
        }

    # ─────────────────────────────────────────────────────────────────
    # VISUALISATIONS 3D
    # ─────────────────────────────────────────────────────────────────

    def _make_colormap(self, n_states):
        """Crée une colormap avec des couleurs bien séparées."""
        if n_states <= 20:
            base = plt.cm.tab20(np.linspace(0, 1, 20))
            return ListedColormap(base[:n_states])
        else:
            return plt.cm.nipy_spectral

    def plot_3d_particles(self, ax, states, n_states, title, point_size=2, alpha=0.6):
        """
        Dessine les particules colorées par état sur un axe 3D.

        Args:
            ax: matplotlib 3D axis
            states: array d'indices d'état
            n_states: nombre total d'états
            title: titre du subplot
            point_size: taille des points
            alpha: transparence
        """
        cmap = self._make_colormap(n_states)

        if n_states > 20:
            colors = cmap(states / max(states.max(), 1))
        else:
            colors = cmap(states)

        ax.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            self.coords[:, 2],
            c=states,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )

        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.tick_params(labelsize=6)

    def _draw_cartesian_grid(self, ax, partitioner, alpha=0.15):
        """Dessine les bords de la grille cartésienne."""
        xmin, xmax, ymin, ymax, zmin, zmax = partitioner._bounds
        nx, ny, nz = partitioner.nx, partitioner.ny, partitioner.nz

        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        z_edges = np.linspace(zmin, zmax, nz + 1)

        # Lignes selon X
        for yi in y_edges:
            for zi in z_edges:
                ax.plot([xmin, xmax], [yi, yi], [zi, zi], "k-", alpha=alpha, lw=0.3)
        # Lignes selon Y
        for xi in x_edges:
            for zi in z_edges:
                ax.plot([xi, xi], [ymin, ymax], [zi, zi], "k-", alpha=alpha, lw=0.3)
        # Lignes selon Z
        for xi in x_edges:
            for yi in y_edges:
                ax.plot([xi, xi], [yi, yi], [zmin, zmax], "k-", alpha=alpha, lw=0.3)

    def _draw_cylindrical_grid(self, ax, partitioner, alpha=0.2):
        """Dessine les anneaux et secteurs du cylindrique."""
        xc = partitioner._x_center
        yc = partitioner._y_center
        zmin = partitioner._z_min
        zmax = partitioner._z_max
        r_edges = partitioner._r_edges
        ntheta = partitioner.ntheta
        nz = partitioner.nz

        theta = np.linspace(0, 2 * np.pi, 100)
        z_edges = np.linspace(zmin, zmax, nz + 1)

        # Cercles concentriques à chaque z
        for r in r_edges:
            for z in z_edges:
                ax.plot(
                    xc + r * np.cos(theta),
                    yc + r * np.sin(theta),
                    np.full_like(theta, z),
                    "k-",
                    alpha=alpha,
                    lw=0.4,
                )

        # Secteurs angulaires
        theta_edges = np.linspace(0, 2 * np.pi, ntheta + 1)
        r_max = r_edges[-1]
        for t in theta_edges:
            for z in z_edges:
                ax.plot(
                    [xc, xc + r_max * np.cos(t)],
                    [yc, yc + r_max * np.sin(t)],
                    [z, z],
                    "k-",
                    alpha=alpha,
                    lw=0.4,
                )

        # Lignes verticales sur le cercle extérieur
        for t in theta_edges:
            ax.plot(
                [xc + r_max * np.cos(t)] * 2,
                [yc + r_max * np.sin(t)] * 2,
                [zmin, zmax],
                "k-",
                alpha=alpha,
                lw=0.4,
            )

    def _draw_voronoi_centroids(self, ax, partitioner, alpha=0.8):
        """Dessine les centroïdes du Voronoï."""
        if partitioner.centroids is not None:
            ax.scatter(
                partitioner.centroids[:, 0],
                partitioner.centroids[:, 1],
                partitioner.centroids[:, 2],
                c="red",
                s=30,
                marker="*",
                alpha=alpha,
                edgecolors="black",
                linewidths=0.5,
                zorder=10,
                label="Centroïdes",
            )

    def _draw_octree_boxes(self, ax, partitioner, alpha=0.08):
        """Dessine les boîtes de l'octree."""
        for leaf in partitioner._leaves:
                xmin, xmax, ymin, ymax, zmin, zmax = leaf

                # 8 coins
                corners = np.array(
                    [
                        [xmin, ymin, zmin],
                        [xmax, ymin, zmin],
                        [xmax, ymax, zmin],
                        [xmin, ymax, zmin],
                        [xmin, ymin, zmax],
                        [xmax, ymin, zmax],
                        [xmax, ymax, zmax],
                        [xmin, ymax, zmax],
                    ]
                )

                # 6 faces
                faces = [
                    [corners[0], corners[1], corners[2], corners[3]],
                    [corners[4], corners[5], corners[6], corners[7]],
                    [corners[0], corners[1], corners[5], corners[4]],
                    [corners[2], corners[3], corners[7], corners[6]],
                    [corners[0], corners[3], corners[7], corners[4]],
                    [corners[1], corners[2], corners[6], corners[5]],
                ]

     poly = Poly3DCollection(faces, alpha=alpha, edgecolors="k", linewidths=0.3)
            poly.set_facecolor((0.5, 0.5, 0.5, alpha))
            ax.add_collection3d(poly)

    def _draw_adaptive_split(self, ax, partitioner, alpha=0.3):
        """Dessine le plan de séparation adaptative en z."""
        xmin, xmax, ymin, ymax, zmin, zmax = partitioner._bounds
        z_split = partitioner._z_split
        
        # Dessiner le plan de séparation comme un rectangle rouge semi-transparent
        xx = np.linspace(xmin, xmax, 20)
        yy = np.linspace(ymin, ymax, 20)
        X, Y = np.meshgrid(xx, yy)
        Z = np.full_like(X, z_split)
        
        ax.plot_surface(
            X, Y, Z, color='red', alpha=alpha, edgecolor='none', zorder=5
        )
        
        # Annoter les zones
        mid_z = (zmin + z_split) / 2
        ax.text(
            xmax, ymax, mid_z,
            f'Zone basse\n{partitioner._n_cells_bottom} cellules',
            color='blue', fontsize=7
        )
        
        mid_z_top = (z_split + zmax) / 2
        ax.text(
            xmax, ymax, mid_z_top,
            f'Zone haute\n{partitioner._n_cells_top} cellules',
            color='red', fontsize=7
        )

        # Annoter les zones
        mid_z = (zmin + z_split) / 2
        ax.text(
            xmax,
            ymax,
            mid_z,
            f"Zone basse\n{partitioner._n_cells_bottom} cellules",
            color="blue",
            fontsize=7,
        )

        mid_z_top = (z_split + zmax) / 2
        ax.text(
            xmax,
            ymax,
            mid_z_top,
            f"Zone haute\n{partitioner._n_cells_top} cellules",
            color="red",
            fontsize=7,
        )

    # ─────────────────────────────────────────────────────────────────
    # VUES 2D (COUPES)
    # ─────────────────────────────────────────────────────────────────

    def plot_2d_slice(
        self,
        ax,
        states,
        n_states,
        axis="z",
        slice_frac=0.5,
        title="",
        point_size=5,
        thickness_frac=0.1,
    ):
        """
        Coupe 2D du mélangeur à une position donnée.

        Args:
            ax: matplotlib 2D axis
            states: array d'indices d'état
            axis: "x", "y" ou "z" — axe perpendiculaire à la coupe
            slice_frac: position de la coupe (0=min, 1=max)
            thickness_frac: épaisseur relative de la tranche
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        other_axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
        axis_labels = {"x": ("Y", "Z"), "y": ("X", "Z"), "z": ("X", "Y")}

        ai = axis_map[axis]
        a1, a2 = other_axes[axis]

        vals = self.coords[:, ai]
        vmin, vmax = vals.min(), vals.max()
        center = vmin + slice_frac * (vmax - vmin)
        half_thick = thickness_frac * (vmax - vmin) / 2

        mask = np.abs(vals - center) < half_thick

        cmap = self._make_colormap(n_states)

        ax.scatter(
            self.coords[mask, a1],
            self.coords[mask, a2],
            c=states[mask],
            cmap=cmap,
            s=point_size,
            alpha=0.7,
            edgecolors="none",
        )

        xlabel, ylabel = axis_labels[axis]
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title}\nCoupe {axis}={center:.3f}", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # ─────────────────────────────────────────────────────────────────
    # VISUALISATION PRINCIPALE
    # ─────────────────────────────────────────────────────────────────

    def show_all_methods(self, figsize=(24, 16), elev=25, azim=45):
        """
        Affiche une grille 3×2 avec toutes les méthodes de partitionnement.
        Vue 3D avec les particules colorées par cellule.
        """
        if self.coords is None:
            raise ValueError("Appelez load_particles() d'abord")

        partitioners_config = self.get_default_partitioners()
        n_methods = len(partitioners_config)

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            "COMPARAISON DES MÉTHODES DE PARTITIONNEMENT\n"
            f"({len(self.coords)} particules)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        for i, (label, config) in enumerate(partitioners_config.items()):
            print(f"🔧 {label.split(chr(10))[0]}...")

            # Créer et fitter le partitionneur
            part = create_partitioner(config["method"], **config["kwargs"])
            part.fit(self.coords)

            # Calculer les états
            states = part.compute_states(
                self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
            )

            # Diagnostics
            diag = part.diagnostics(self.coords)
            n_states = part.n_cells
            subtitle = (
                f"{n_states} cellules | {diag['n_visited']} visitées\n"
                f"pop: [{diag['pop_min']}, {diag['pop_max']}] "
                f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}"
            )

            # Plot 3D
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            self.plot_3d_particles(
                ax, states, n_states, f"{label}\n{subtitle}", point_size=1.5
            )
            ax.view_init(elev=elev, azim=azim)

 # Dessiner les grilles/frontières
        method = config["method"]
        if method == "cartesian":
            self._draw_cartesian_grid(ax, part)
        elif method == "cylindrical":
            self._draw_cylindrical_grid(ax, part)
        elif method == "voronoi":
            self._draw_voronoi_centroids(ax, part)
        elif method == "octree":
            self._draw_octree_boxes(ax, part)
        elif method == "adaptive":
            self._draw_adaptive_split(ax, part)
            # Dessiner aussi les grilles du partitionneur de la zone basse si possible
            if hasattr(part, "_bottom_partitioner") and hasattr(part._bottom_partitioner, "method"):
                bottom_method = part._bottom_partitioner.__class__.__name__.replace("Partitioner", "").lower()
                if bottom_method == "cartesian":
                    self._draw_cartesian_grid(ax, part._bottom_partitioner)
                elif bottom_method == "cylindrical":
                    self._draw_cylindrical_grid(ax, part._bottom_partitioner)
                elif bottom_method == "voronoi":
                    self._draw_voronoi_centroids(ax, part._bottom_partitioner)
                elif bottom_method == "octree":
                    self._draw_octree_boxes(ax, part._bottom_partitioner)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(
            "images/partitioning_3d_comparison.png", dpi=200, bbox_inches="tight"
        )
        plt.show()
        print("✅ Sauvegardé: images/partitioning_3d_comparison.png")

    def show_all_methods_2d(self, slice_axis="z", slice_frac=0.5, figsize=(24, 16)):
        """
        Affiche les coupes 2D pour toutes les méthodes.
        """
        if self.coords is None:
            raise ValueError("Appelez load_particles() d'abord")

        partitioners_config = self.get_default_partitioners()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            f"COUPES 2D — axe {slice_axis} (position {slice_frac:.0%})\n"
            f"({len(self.coords)} particules)",
            fontsize=16,
            fontweight="bold",
        )

        for i, (label, config) in enumerate(partitioners_config.items()):
            row, col = divmod(i, 3)
            ax = axes[row, col]

            print(f"🔧 {label.split(chr(10))[0]}...")

            part = create_partitioner(config["method"], **config["kwargs"])
            part.fit(self.coords)

            states = part.compute_states(
                self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
            )

            diag = part.diagnostics(self.coords)
            n_states = part.n_cells
            full_label = (
                f"{label}\n{n_states} cellules | "
                f"pop: μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}"
            )

            self.plot_2d_slice(
                ax,
                states,
                n_states,
                axis=slice_axis,
                slice_frac=slice_frac,
                title=full_label,
                point_size=8,
            )

        plt.tight_layout()
        plt.savefig(
            "images/partitioning_2d_comparison.png", dpi=200, bbox_inches="tight"
        )
        plt.show()
        print("✅ Sauvegardé: images/partitioning_2d_comparison.png")

    def show_single_method_detailed(
        self, method, method_kwargs, figsize=(20, 15), elev=25, azim=45
    ):
        """
        Vue détaillée d'une seule méthode: 3D + 3 coupes + histogramme.
        """
        if self.coords is None:
            raise ValueError("Appelez load_particles() d'abord")

        part = create_partitioner(method, **method_kwargs)
        part.fit(self.coords)
        states = part.compute_states(
            self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        )
        n_states = part.n_cells
        diag = part.diagnostics(self.coords)

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"{method.upper()} — {part.label}\n"
            f"{n_states} cellules | {diag['n_visited']} visitées | "
            f"pop: [{diag['pop_min']}, {diag['pop_max']}] "
            f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}",
            fontsize=14,
            fontweight="bold",
        )

        # ── 3D principal ──
        ax1 = fig.add_subplot(2, 3, (1, 4), projection="3d")
        self.plot_3d_particles(ax1, states, n_states, "Vue 3D", point_size=2)
        ax1.view_init(elev=elev, azim=azim)

        if method == "cartesian":
            self._draw_cartesian_grid(ax1, part)
        elif method == "cylindrical":
            self._draw_cylindrical_grid(ax1, part)
        elif method == "voronoi":
            self._draw_voronoi_centroids(ax1, part)
        elif method == "octree":
            self._draw_octree_boxes(ax1, part)
        elif method == "adaptive":
            self._draw_adaptive_split(ax1, part)
            # Dessiner aussi les grilles du sous-partitionneur de la zone basse si possible
            if hasattr(part, "_bottom_partitioner") and hasattr(
                part._bottom_partitioner, "__class__"
            ):
                bottom_method = (
                    part._bottom_partitioner.__class__.__name__
                    .replace("Partitioner", "")
                    .lower()
                )
                if bottom_method == "cartesian":
                    self._draw_cartesian_grid(ax1, part._bottom_partitioner)
                elif bottom_method == "cylindrical":
                    self._draw_cylindrical_grid(ax1, part._bottom_partitioner)
                elif bottom_method == "voronoi":
                    self._draw_voronoi_centroids(ax1, part._bottom_partitioner)
                elif bottom_method == "octree":
                    self._draw_octree_boxes(ax1, part._bottom_partitioner)

        # ── Coupes 2D ──
        for j, (axis, frac) in enumerate([("z", 0.3), ("z", 0.5), ("z", 0.7)]):
            ax = fig.add_subplot(2, 3, j + 2)
            self.plot_2d_slice(
                ax,
                states,
                n_states,
                axis=axis,
                slice_frac=frac,
                title=f"Coupe {axis}={frac:.0%}",
                point_size=10,
            )

        # ── Histogramme population ──
        ax5 = fig.add_subplot(2, 3, 5)
        counts = np.bincount(states, minlength=n_states)
        ax5.bar(range(n_states), counts, color="steelblue", alpha=0.8, width=1.0)
        ax5.axhline(
            counts[counts > 0].mean(),
            color="red",
            ls="--",
            label=f"μ={counts[counts > 0].mean():.0f}",
        )
        ax5.set_xlabel("Index de cellule")
        ax5.set_ylabel("Nombre de particules")
        ax5.set_title("Population par cellule")
        ax5.legend()

        # ── Histogramme taille ──
        ax6 = fig.add_subplot(2, 3, 6)
        counts_nonzero = counts[counts > 0]
        ax6.hist(
            counts_nonzero, bins=30, color="steelblue", alpha=0.8, edgecolor="white"
        )
        ax6.axvline(
            counts_nonzero.mean(),
            color="red",
            ls="--",
            label=f"μ={counts_nonzero.mean():.0f}",
        )
        ax6.axvline(
            counts_nonzero.median(),
            color="orange",
            ls="--",
            label=f"med={np.median(counts_nonzero):.0f}",
        )
        ax6.set_xlabel("Particules par cellule")
        ax6.set_ylabel("Nombre de cellules")
        ax6.set_title("Distribution de population")
        ax6.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"detailed_{method}.png", dpi=200, bbox_inches="tight")
        plt.show()

    def show_cylindrical_comparison(self, figsize=(20, 10)):
        """Compare equal_dr vs equal_area pour le cylindrique."""
        if self.coords is None:
            raise ValueError("Appelez load_particles() d'abord")

        fig, axes_grid = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle(
            "CYLINDRIQUE: equal_dr vs equal_area", fontsize=14, fontweight="bold"
        )

        for row, mode in enumerate(["equal_dr", "equal_area"]):
            part = create_partitioner(
                "cylindrical", nr=5, ntheta=8, nz=5, radial_mode=mode
            )
            part.fit(self.coords)
            states = part.compute_states(
                self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
            )
            n_states = part.n_cells
            diag = part.diagnostics(self.coords)

            # 3D
            ax3d = fig.add_subplot(2, 4, row * 4 + 1, projection="3d")
            self.plot_3d_particles(
                ax3d, states, n_states, f"{mode}\nσ={diag['pop_std']:.0f}"
            )
            self._draw_cylindrical_grid(ax3d, part)
            ax3d.view_init(elev=20, azim=45)

            # Coupes
            for j, frac in enumerate([0.3, 0.5, 0.7]):
                ax = axes_grid[row, j + 1]
                self.plot_2d_slice(
                    ax,
                    states,
                    n_states,
                    axis="z",
                    slice_frac=frac,
                    title=f"{mode} z={frac:.0%}",
                    point_size=6,
                )

        plt.tight_layout()
        plt.savefig("cylindrical_comparison.png", dpi=200, bbox_inches="tight")
        plt.show()

    def show_resolution_sweep(self, method="cartesian", figsize=(24, 12)):
        """Compare différentes résolutions pour une même méthode."""
        if self.coords is None:
            raise ValueError("Appelez load_particles() d'abord")

        if method == "cartesian":
            configs = [
                {"nx": 3, "ny": 3, "nz": 3},
                {"nx": 5, "ny": 5, "nz": 5},
                {"nx": 8, "ny": 8, "nz": 8},
                {"nx": 10, "ny": 10, "nz": 10},
                {"nx": 15, "ny": 15, "nz": 15},
                {"nx": 20, "ny": 20, "nz": 20},
            ]
        elif method == "voronoi":
            configs = [
                {"n_cells": 8},
                {"n_cells": 27},
                {"n_cells": 64},
                {"n_cells": 125},
                {"n_cells": 512},
                {"n_cells": 1000},
            ]
        elif method == "cylindrical":
            configs = [
                {"nr": 3, "ntheta": 4, "nz": 3, "radial_mode": "equal_area"},
                {"nr": 5, "ntheta": 6, "nz": 5, "radial_mode": "equal_area"},
                {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_area"},
                {"nr": 8, "ntheta": 8, "nz": 8, "radial_mode": "equal_area"},
                {"nr": 10, "ntheta": 12, "nz": 10, "radial_mode": "equal_area"},
                {"nr": 15, "ntheta": 16, "nz": 15, "radial_mode": "equal_area"},
            ]
        elif method == "quantile":
            configs = [
                {"nx": 3, "ny": 3, "nz": 3},
                {"nx": 5, "ny": 5, "nz": 5},
                {"nx": 8, "ny": 8, "nz": 8},
                {"nx": 10, "ny": 10, "nz": 10},
                {"nx": 15, "ny": 15, "nz": 15},
                {"nx": 20, "ny": 20, "nz": 20},
            ]
        elif method == "octree":
            configs = [
                {"max_particles": 500, "max_depth": 3},
                {"max_particles": 200, "max_depth": 4},
                {"max_particles": 100, "max_depth": 4},
                {"max_particles": 50, "max_depth": 5},
                {"max_particles": 20, "max_depth": 5},
                {"max_particles": 10, "max_depth": 6},
            ]
        else:
            print(f"Méthode {method} non supportée pour le sweep")
            return

        n_configs = len(configs)
        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"SWEEP DE RÉSOLUTION — {method.upper()}\n({len(self.coords)} particules)",
            fontsize=14,
            fontweight="bold",
        )

        # Ligne 1 : vues 3D
        # Ligne 2 : coupes 2D
        for i, kwargs in enumerate(configs):
            print(f"   [{i + 1}/{n_configs}] {kwargs}")

            part = create_partitioner(method, **kwargs)
            part.fit(self.coords)
            states = part.compute_states(
                self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
            )
            n_states = part.n_cells
            diag = part.diagnostics(self.coords)

            # 3D
            ax3d = fig.add_subplot(2, n_configs, i + 1, projection="3d")
            self.plot_3d_particles(
                ax3d,
                states,
                n_states,
                f"{n_states} cellules\nσ/μ={diag['pop_std'] / max(diag['pop_mean'], 1):.2f}",
                point_size=1,
            )
            ax3d.view_init(elev=25, azim=45)

            # Coupe 2D
            ax2d = fig.add_subplot(2, n_configs, n_configs + i + 1)
            self.plot_2d_slice(
                ax2d,
                states,
                n_states,
                axis="z",
                slice_frac=0.5,
                title=f"{n_states} cellules",
                point_size=5,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(
            f"/images/resolution_sweep_{method}.png", dpi=200, bbox_inches="tight"
        )
        plt.show()

    def show_population_comparison(self, figsize=(18, 10)):
        """Compare la distribution de population entre méthodes."""
        if self.coords is None:
            raise ValueError("Appelez load_particles() d'abord")

        partitioners_config = self.get_default_partitioners()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            "DISTRIBUTION DE POPULATION PAR CELLULE\n(la méthode idéale a σ/μ → 0)",
            fontsize=14,
            fontweight="bold",
        )

        summary = []

        for i, (label, config) in enumerate(partitioners_config.items()):
            row, col = divmod(i, 3)
            ax = axes[row, col]

            part = create_partitioner(config["method"], **config["kwargs"])
            part.fit(self.coords)
            states = part.compute_states(
                self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
            )
            n_states = part.n_cells

            counts = np.bincount(states, minlength=n_states)
            counts_nz = counts[counts > 0]

            cv = counts_nz.std() / counts_nz.mean() if counts_nz.mean() > 0 else 0

            ax.hist(
                counts_nz,
                bins=30,
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                density=True,
            )
            ax.axvline(
                counts_nz.mean(),
                color="red",
                ls="--",
                lw=2,
                label=f"μ={counts_nz.mean():.0f}",
            )
            ax.axvline(
                np.median(counts_nz),
                color="orange",
                ls="--",
                lw=2,
                label=f"med={np.median(counts_nz):.0f}",
            )

            short_label = label.split("\n")[0]
            ax.set_title(
                f"{short_label}\nCV={cv:.2f} | {(counts == 0).sum()} vides/{n_states}",
                fontsize=10,
            )
            ax.set_xlabel("Particules/cellule")
            ax.set_ylabel("Densité")
            ax.legend(fontsize=8)

            summary.append(
                {
                    "method": short_label,
                    "n_cells": n_states,
                    "n_empty": int((counts == 0).sum()),
                    "cv": cv,
                    "mean": counts_nz.mean(),
                }
            )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig("population_comparison.png", dpi=200, bbox_inches="tight")
        plt.show()

        # Résumé
        print("\n📊 Résumé (CV = coefficient de variation, plus petit = mieux):")
        print(
            f"{'Méthode':30s} {'Cellules':>8s} {'Vides':>6s} {'CV':>6s} {'μ pop':>8s}"
        )
        print("-" * 65)
        for s in sorted(summary, key=lambda x: x["cv"]):
            print(
                f"{s['method']:30s} {s['n_cells']:8d} {s['n_empty']:6d} "
                f"{s['cv']:6.3f} {s['mean']:8.1f}"
            )


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    viz = PartitionVisualizer()

    # Charger les particules
    viz.load_particles(file_index=100)

    # ── Vue d'ensemble : toutes les méthodes ──
    print("\n" + "=" * 60)
    print("1. COMPARAISON 3D DE TOUTES LES MÉTHODES")
    print("=" * 60)
    viz.show_all_methods()

    # ── Coupes 2D ──
    print("\n" + "=" * 60)
    print("2. COUPES 2D")
    print("=" * 60)
    viz.show_all_methods_2d(slice_axis="z", slice_frac=0.5)

    # ── Comparaison des populations ──
    print("\n" + "=" * 60)
    print("3. DISTRIBUTION DE POPULATION")
    print("=" * 60)
    viz.show_population_comparison()

    # ── Détail cylindrique ──
    print("\n" + "=" * 60)
    print("4. CYLINDRIQUE: equal_dr vs equal_area")
    print("=" * 60)
    viz.show_cylindrical_comparison()

    # ── Sweep résolution Voronoï ──
    print("\n" + "=" * 60)
    print("5. SWEEP RÉSOLUTION VORONOÏ")
    print("=" * 60)
    viz.show_resolution_sweep(method="voronoi")

    # ── Vue détaillée Voronoï ──
    print("\n" + "=" * 60)
    print("6. VUE DÉTAILLÉE VORONOÏ")
    print("=" * 60)
    viz.show_single_method_detailed("voronoi", {"n_cells": 125})

    print("\n✨ Toutes les visualisations générées!")
