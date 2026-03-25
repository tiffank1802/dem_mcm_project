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
    }
}

def get_partitioner_schema(method):
    """Get parameter schema for a partitioner method"""
    return PARTITIONER_SCHEMAS.get(method)

def get_partitioner_kwargs(method, **params):
    """
    Build kwargs for partitioner from user params.
    Validates and applies defaults.
    """
    schema = get_partitioner_schema(method)
    if not schema:
        raise ValueError(f"Unknown partitioner method: {method}")
    
    kwargs = {}
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
