"""
Schémas des paramètres des partitionneurs pour validation et génération UI
"""

PARTITIONER_SCHEMAS = {
    "cartesian": {
        "label": "Cartésien (Grille Régulière)",
        "description": "Découpe l'espace en grille régulière 3D",
        "parameters": {
            "nx": {
                "type": "int",
                "min": 2,
                "max": 20,
                "default": 5,
                "label": "Nombre de divisions X",
                "help": "Nombre de cellules selon l'axe X"
            },
            "ny": {
                "type": "int",
                "min": 2,
                "max": 20,
                "default": 5,
                "label": "Nombre de divisions Y",
                "help": "Nombre de cellules selon l'axe Y"
            },
            "nz": {
                "type": "int",
                "min": 2,
                "max": 20,
                "default": 5,
                "label": "Nombre de divisions Z",
                "help": "Nombre de cellules selon l'axe Z"
            }
        }
    },
    "cylindrical": {
        "label": "Cylindrique",
        "description": "Partitionnement cylindrique (optimal pour mélangeurs)",
        "parameters": {
            "nr": {
                "type": "int",
                "min": 2,
                "max": 10,
                "default": 3,
                "label": "Nombres de zones radiales",
                "help": "Nombre de couronnes concentriques"
            },
            "ntheta": {
                "type": "int",
                "min": 3,
                "max": 12,
                "default": 4,
                "label": "Nombre de secteurs angulaires",
                "help": "Nombre de parts de tarte"
            },
            "nz": {
                "type": "int",
                "min": 1,
                "max": 10,
                "default": 2,
                "label": "Nombre de niveaux axiaux",
                "help": "Nombre d'étages"
            },
            "radial_mode": {
                "type": "select",
                "options": ["equal_dr", "equal_area"],
                "default": "equal_dr",
                "label": "Mode radial",
                "help": "equal_dr: rayons égaux | equal_area: surfaces égales"
            }
        }
    },
    "quantile": {
        "label": "Quantile",
        "description": "Grille basée sur quantiles (distribution uniforme)",
        "parameters": {
            "nx": {
                "type": "int",
                "min": 2,
                "max": 20,
                "default": 5,
                "label": "Nombre de divisions X",
                "help": "Nombre de cellules selon l'axe X"
            },
            "ny": {
                "type": "int",
                "min": 2,
                "max": 20,
                "default": 5,
                "label": "Nombre de divisions Y",
                "help": "Nombre de cellules selon l'axe Y"
            },
            "nz": {
                "type": "int",
                "min": 2,
                "max": 20,
                "default": 5,
                "label": "Nombre de divisions Z",
                "help": "Nombre de cellules selon l'axe Z"
            }
        }
    },
    "voronoi": {
        "label": "Voronoï",
        "description": "Cellules de Voronoï autour de centres aléatoires",
        "parameters": {
            "n_cells": {
                "type": "int",
                "min": 8,
                "max": 250,
                "default": 125,
                "label": "Nombre de cellules",
                "help": "Nombre total de cellules Voronoï"
            },
            "random_state": {
                "type": "int",
                "min": 0,
                "max": 1000,
                "default": 42,
                "label": "Graine aléatoire",
                "help": "Pour reproductibilité des centroids"
            }
        }
    },
    "octree": {
        "label": "Octree",
        "description": "Partitionnement adaptatif (plus dense où il y a plus de particules)",
        "parameters": {
            "max_particles": {
                "type": "int",
                "min": 10,
                "max": 1000,
                "default": 50,
                "label": "Max particules par nœud feuille",
                "help": "Seuil pour la subdivision récursive"
            },
            "max_depth": {
                "type": "int",
                "min": 2,
                "max": 8,
                "default": 4,
                "label": "Profondeur max de l'arbre",
                "help": "Profondeur maximale de récursion"
            }
        }
    },
    "physics": {
        "label": "Physics-Aware",
        "description": "Partitionnement basé sur la dynamique physique",
        "parameters": {
            "n_cells": {
                "type": "int",
                "min": 8,
                "max": 250,
                "default": 125,
                "label": "Nombre de cellules",
                "help": "Nombre total de cellules"
            },
            "velocity_weight": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "label": "Poids de la vélocité",
                "help": "Équilibre entre position et vélocité"
            },
            "random_state": {
                "type": "int",
                "min": 0,
                "max": 1000,
                "default": 42,
                "label": "Graine aléatoire",
                "help": "Pour reproductibilité"
            }
        }
    },
    "adaptive": {
        "label": "Adaptatif (Zone Haute/Basse)",
        "description": "Deux zones avec partitionnements différents: zone haute (grossière) et zone basse (fine)",
        "parameters": {
            "z_split": {
                "type": "float",
                "min": 0.1,
                "max": 0.9,
                "default": 0.75,
                "label": "Position de la séparation (z)",
                "help": "Altitude ou quantile où séparer les zones"
            },
            "z_split_mode": {
                "type": "select",
                "options": ["quantile", "absolute"],
                "default": "quantile",
                "label": "Mode de séparation",
                "help": "quantile: % du domaine | absolute: position en z"
            },
            "n_cells_top": {
                "type": "int",
                "min": 1,
                "max": 10,
                "default": 1,
                "label": "Cellules en zone haute",
                "help": "Nombre de cellules pour la zone haute (généralement 1)"
            },
            "top_method": {
                "type": "select",
                "options": ["single", "cartesian", "cylindrical"],
                "default": "single",
                "label": "Méthode pour zone haute",
                "help": "Type de partitionnement pour la zone basse"
            },
            "bottom_method": {
                "type": "select",
                "options": ["cartesian", "cylindrical", "voronoi", "octree"],
                "default": "cylindrical",
                "label": "Méthode pour zone basse",
                "help": "Type de partitionnement pour la zone basse (généralement cylindrique)"
            },
            "bottom_nr": {
                "type": "int",
                "min": 1,
                "max": 10,
                "default": 3,
                "label": "Bottom: zones radiales (si cylindrique)",
                "help": "Paramètre nr pour cylindrique"
            },
            "bottom_ntheta": {
                "type": "int",
                "min": 1,
                "max": 12,
                "default": 4,
                "label": "Bottom: secteurs angulaires (si cylindrique)",
                "help": "Paramètre ntheta pour cylindrique"
            },
            "bottom_nz": {
                "type": "int",
                "min": 1,
                "max": 10,
                "default": 3,
                "label": "Bottom: niveaux axiaux (si cylindrique)",
                "help": "Paramètre nz pour cylindrique"
            }
        }
    },
    "multizone": {
        "label": "Multi-Zones",
        "description": "N zones avec partitionnements différents (plus flexible que adaptatif)",
        "parameters": {
            "n_zones": {
                "type": "int",
                "min": 2,
                "max": 5,
                "default": 2,
                "label": "Nombre de zones",
                "help": "Nombre de zones à créer"
            },
            "z_mode": {
                "type": "select",
                "options": ["quantile", "absolute"],
                "default": "quantile",
                "label": "Mode de séparation",
                "help": "quantile: % du domaine | absolute: position en z"
            },
            "zone1_method": {
                "type": "select",
                "options": ["cartesian", "cylindrical", "voronoi", "octree", "single"],
                "default": "cylindrical",
                "label": "Zone 1 - Méthode",
                "help": "Partition pour la zone 1"
            },
            "zone2_method": {
                "type": "select",
                "options": ["cartesian", "cylindrical", "voronoi", "octree", "single"],
                "default": "single",
                "label": "Zone 2 - Méthode",
                "help": "Partition pour la zone 2"
            },
            "zone3_method": {
                "type": "select",
                "options": ["cartesian", "cylindrical", "voronoi", "octree", "single"],
                "default": "single",
                "label": "Zone 3 - Méthode (optionnel)",
                "help": "Partition pour la zone 3 (si n_zones >= 3)"
            },
            "z1_split": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "label": "Zone 1 - Limite haute",
                "help": "Où séparer zone 1 et zone 2"
            },
            "z2_split": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.8,
                "label": "Zone 2 - Limite haute",
                "help": "Où séparer zone 2 et zone 3"
            }
        }
    },
    "single": {
        "label": "Cellule Unique",
        "description": "Une seule cellule pour tout le domaine (cas trivial)",
        "parameters": {}
    }
}

def get_partitioner_schema(method):
    """Get parameter schema for a partitioner method"""
    return PARTITIONER_SCHEMAS.get(method)

def get_partitioner_kwargs(method, **params):
    """
    Build kwargs for partitioner from user params.
    Validates and applies defaults.
    
    Special handling for:
    - adaptive: constructs top_kwargs, bottom_kwargs from individual params
    - multizone: constructs zones list from zone_*_method params
    """
    schema = get_partitioner_schema(method)
    if not schema:
        raise ValueError(f"Unknown partitioner method: {method}")
    
    kwargs = {}
    
    # Handle adaptive partition specifically
    if method == "adaptive":
        kwargs = _handle_adaptive_kwargs(params, schema)
    # Handle multizone partition specifically
    elif method == "multizone":
        kwargs = _handle_multizone_kwargs(params, schema)
    # Handle single partition (no parameters)
    elif method == "single":
        kwargs = {}
    # Generic path for other methods
    else:
        for param_name, param_spec in schema["parameters"].items():
            if param_name in params:
                value = params[param_name]
                # Type conversion
                if param_spec["type"] in ["int"]:
                    value = int(value)
                elif param_spec["type"] == "float":
                    value = float(value)
                # Validation
                if "min" in param_spec:
                    if value < param_spec["min"]:
                        value = param_spec["min"]
                if "max" in param_spec:
                    if value > param_spec["max"]:
                        value = param_spec["max"]
                kwargs[param_name] = value
            else:
                # Use default
                kwargs[param_name] = param_spec["default"]
    
    return kwargs


def _handle_adaptive_kwargs(params, schema):
    """
    Construct adaptive partition kwargs from flat params dict.
    
    Converts:
    - z_split, z_split_mode, n_cells_top, top_method
    - bottom_method, bottom_nr, bottom_ntheta, bottom_nz
    
    Into:
    - z_split, z_split_mode, n_cells_top, top_method, top_kwargs
    - bottom_method, bottom_kwargs
    """
    kwargs = {}
    
    # Simple scalar parameters
    for key in ["z_split", "z_split_mode", "n_cells_top", "top_method", "bottom_method"]:
        if key in params:
            value = params[key]
            param_spec = schema["parameters"][key]
            if param_spec["type"] == "float":
                value = float(value)
            elif param_spec["type"] == "int":
                value = int(value)
            kwargs[key] = value
        else:
            kwargs[key] = schema["parameters"][key]["default"]
    
    # Build top_kwargs (usually empty for "single")
    kwargs["top_kwargs"] = {}
    # (Could be expanded if supporting cartesian/cylindrical in top)
    
    # Build bottom_kwargs based on bottom_method
    bottom_kwargs = {}
    bottom_method = kwargs.get("bottom_method", "cylindrical")
    
    if bottom_method == "cylindrical":
        # Cylindrical expects: nr, ntheta, nz, radial_mode
        bottom_kwargs["nr"] = int(params.get("bottom_nr", 3))
        bottom_kwargs["ntheta"] = int(params.get("bottom_ntheta", 4))
        bottom_kwargs["nz"] = int(params.get("bottom_nz", 3))
        bottom_kwargs["radial_mode"] = "equal_dr"  # default
    
    elif bottom_method == "cartesian":
        # Cartesian expects: nx, ny, nz
        bottom_kwargs["nx"] = int(params.get("bottom_nx", 5))
        bottom_kwargs["ny"] = int(params.get("bottom_ny", 5))
        bottom_kwargs["nz"] = int(params.get("bottom_nz", 3))
    
    elif bottom_method == "voronoi":
        # Voronoi expects: n_cells
        bottom_kwargs["n_cells"] = int(params.get("bottom_n_cells", 50))
    
    elif bottom_method == "octree":
        # Octree expects: max_particles, max_depth
        bottom_kwargs["max_particles"] = int(params.get("bottom_max_particles", 50))
        bottom_kwargs["max_depth"] = int(params.get("bottom_max_depth", 4))
    
    kwargs["bottom_kwargs"] = bottom_kwargs
    
    return kwargs


def _handle_multizone_kwargs(params, schema):
    """
    Construct multizone partition kwargs from flat params dict.
    
    Converts zone_*_method, z1_split, z2_split into zones list format.
    """
    kwargs = {}
    
    # Get number of zones
    n_zones = int(params.get("n_zones", 2))
    kwargs["n_zones"] = n_zones
    
    # Get z_mode
    z_mode = params.get("z_mode", "quantile")
    kwargs["z_mode"] = z_mode
    
    # Build zones list dynamically
    zones = []
    z_boundaries = [0.0]
    
    # Collect split boundaries
    if n_zones >= 2:
        z1_split = float(params.get("z1_split", 0.5))
        z_boundaries.append(z1_split)
    
    if n_zones >= 3:
        z2_split = float(params.get("z2_split", 0.8))
        z_boundaries.append(z2_split)
    
    if n_zones >= 4:
        z_boundaries.append(1.0)
    else:
        z_boundaries.append(1.0)
    
    # Create zone dictionaries
    for i in range(n_zones):
        zone_method_key = f"zone{i+1}_method"
        zone_method = params.get(zone_method_key, "single")
        
        zone_dict = {
            "z_min": z_boundaries[i],
            "z_max": z_boundaries[i + 1] if i + 1 < len(z_boundaries) else 1.0,
            "method": zone_method,
            "kwargs": _get_zone_kwargs(zone_method, params, i + 1)
        }
        zones.append(zone_dict)
    
    kwargs["zones"] = zones
    
    return kwargs


def _get_zone_kwargs(method, params, zone_num):
    """Get kwargs for a zone with given method."""
    kwargs = {}
    
    if method == "cylindrical":
        kwargs["nr"] = int(params.get(f"zone{zone_num}_nr", 3))
        kwargs["ntheta"] = int(params.get(f"zone{zone_num}_ntheta", 4))
        kwargs["nz"] = int(params.get(f"zone{zone_num}_nz", 1))
        kwargs["radial_mode"] = "equal_dr"
    
    elif method == "cartesian":
        kwargs["nx"] = int(params.get(f"zone{zone_num}_nx", 5))
        kwargs["ny"] = int(params.get(f"zone{zone_num}_ny", 5))
        kwargs["nz"] = int(params.get(f"zone{zone_num}_nz", 1))
    
    elif method == "voronoi":
        kwargs["n_cells"] = int(params.get(f"zone{zone_num}_n_cells", 50))
        kwargs["random_state"] = 42
    
    elif method == "octree":
        kwargs["max_particles"] = int(params.get(f"zone{zone_num}_max_particles", 50))
        kwargs["max_depth"] = int(params.get(f"zone{zone_num}_max_depth", 4))
    
    # "single" has no parameters
    
    return kwargs
