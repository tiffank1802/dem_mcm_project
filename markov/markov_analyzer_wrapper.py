"""
DjangoMarkovAnalyzerWrapper — Interface Django pour MarkovAnalyzer

Gère la génération d'images matplotlib et leur conversion en base64
pour affichage dans l'interface web.
"""

import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import tempfile
import uuid
import traceback
import sys
import os
from pathlib import Path
from django.conf import settings
from datetime import datetime

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analyze_results import MarkovAnalyzer
except ImportError as e:
    raise ImportError(f"Impossible d'importer MarkovAnalyzer: {e}")


# Cache global pour l'instance MarkovAnalyzer
_analyzer_instance = None
_analyzer_lock = __import__('threading').Lock()


def get_analyzer():
    """
    Obtenir ou créer une instance de MarkovAnalyzer (avec cache).
    Lazy initialization pour éviter les chargements inutiles.
    """
    global _analyzer_instance
    
    if _analyzer_instance is None:
        with _analyzer_lock:
            if _analyzer_instance is None:
                print("⏳ Initializing MarkovAnalyzer...")
                analyzer = MarkovAnalyzer()
                # Charger tous les résultats (peut prendre 2+ minutes)
                print("📥 Loading experiments from HuggingFace (this will take ~2-3 minutes on first startup)...")
                analyzer.load_all()
                _analyzer_instance = analyzer
                print(f"✅ MarkovAnalyzer ready with {len(analyzer.results)} experiments")
    
    return _analyzer_instance


class DjangoMarkovAnalyzerWrapper:
    """
    Wrapper de MarkovAnalyzer pour l'environnement Django.
    
    Gère:
    - Chargement des données (lazy, avec cache)
    - Génération des images matplotlib
    - Conversion en base64
    - Organisation par onglets
    """
    
    def __init__(self):
        """Initialiser avec l'analyzer en cache."""
        self.analyzer = get_analyzer()
        self._cache_images = {}
        self._temp_dir = None
    
    def generate_analysis_images(self, experiment_ids, analyses_config, global_params):
        """
        Génère les images matplotlib et les convertit en base64.
        
        Args:
            experiment_ids: Liste des IDs d'expériences sélectionnées
            analyses_config: Liste de {type, for_each_experiment, params}
            global_params: {figsize, n_steps, dem_criterion, export_format}
        
        Returns:
            {
                'images': [...],
                'tabs_generated': [...],
                'generation_time': float,
                'status': 'success' ou 'error',
                'message': str
            }
        """
        start_time = datetime.now()
        images = []
        errors = []
        
        try:
            # Vérifier que le analyzer est prêt (avec timeout)
            print("⏳ Waiting for MarkovAnalyzer to be ready...")
            max_wait = 300  # 5 minutes max
            elapsed = 0
            interval = 2
            while elapsed < max_wait:
                analyzer = self.analyzer
                if hasattr(analyzer, 'results') and analyzer.results:
                    print(f"✅ MarkovAnalyzer ready after {elapsed}s")
                    break
                print(f"   Still waiting... ({elapsed}s/{max_wait}s)")
                import time
                time.sleep(interval)
                elapsed += interval
            
            if not hasattr(analyzer, 'results') or not analyzer.results:
                return {
                    'status': 'error',
                    'message': f'MarkovAnalyzer did not initialize within {max_wait}s. Try again in a moment.',
                    'images': [],
                    'tabs_generated': [],
                    'generation_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Importer le modèle Experiment
            from markov.models import Experiment
            
            # Mapper les IDs aux folder_names
            experiments = Experiment.objects.filter(id__in=experiment_ids)
            id_to_folder = {exp.id: exp.folder_name for exp in experiments}
            
            # Extraire les folder_names dans le même ordre que experiment_ids
            folder_names = [id_to_folder.get(eid) for eid in experiment_ids if eid in id_to_folder]
            
            # Extraire les figsize et autres paramètres
            figsize = tuple(global_params.get('figsize', [16, 10]))
            n_steps = global_params.get('n_steps', 200)
            dem_criterion = global_params.get('dem_criterion', 'z_median')
            
            # Traiter chaque analyse demandée
            for analysis_config in analyses_config:
                analysis_type = analysis_config.get('type')
                for_each = analysis_config.get('for_each_experiment', False)
                params = analysis_config.get('params', {})
                
                try:
                    if analysis_type == 'plot_experiment':
                        # Générer une image pour chaque expérience
                        for i, folder_name in enumerate(folder_names):
                            try:
                                fig = self.analyzer.plot_experiment(
                                    folder_name,
                                    n_steps=params.get('n_steps', n_steps),
                                    figsize=figsize
                                )
                                
                                img_data = self._fig_to_base64(fig)
                                metadata = self._extract_metadata_plot_experiment(folder_name)
                                
                                images.append({
                                    'id': str(uuid.uuid4()),
                                    'type': 'plot_experiment',
                                    'tab': 'experiments',
                                    'title': f"Plot Experiment - {metadata['folder_name']}",
                                    'description': "6 subplots: matrice, diag, RSD, entropie, distribution, validation",
                                    'experiment_ids': [experiment_ids[i]],
                                    'experiment_names': [folder_name],
                                    'image_base64': img_data,
                                    'metadata': metadata
                                })
                                
                                plt.close(fig)
                            except Exception as e:
                                errors.append(f"plot_experiment - {folder_name}: {str(e)}")
                    
                    elif analysis_type == 'plot_matrix':
                        # Générer une heatmap pour chaque expérience
                        for i, folder_name in enumerate(folder_names):
                            try:
                                fig, ax = plt.subplots(figsize=figsize)
                                M = self.analyzer.results[folder_name]['matrix']
                                im = ax.imshow(M, cmap='viridis', aspect='auto')
                                ax.set_xlabel('Destination')
                                ax.set_ylabel('Source')
                                ax.set_title(f"Matrice de Transition - {folder_name}")
                                plt.colorbar(im, ax=ax, label='Probabilité')
                                
                                img_data = self._fig_to_base64(fig)
                                metadata = self._extract_metadata_plot_matrix(folder_name)
                                
                                images.append({
                                    'id': str(uuid.uuid4()),
                                    'type': 'plot_matrix',
                                    'tab': 'experiments',
                                    'title': f"Matrice - {metadata['folder_name']}",
                                    'description': "Heatmap de la matrice de transition P",
                                    'experiment_ids': [experiment_ids[i]],
                                    'experiment_names': [folder_name],
                                    'image_base64': img_data,
                                    'metadata': metadata
                                })
                                
                                plt.close(fig)
                            except Exception as e:
                                errors.append(f"plot_matrix - {folder_name}: {str(e)}")
                    
                    elif analysis_type == 'compare_methods':
                        # Une seule image pour toutes les expériences
                        try:
                            fig = self.analyzer.compare_methods(
                                metric=params.get('metric', 'diag_mean'),
                                figsize=figsize
                            )
                            
                            img_data = self._fig_to_base64(fig)
                            
                            images.append({
                                'id': str(uuid.uuid4()),
                                'type': 'compare_methods',
                                'tab': 'compare',
                                'title': f"Comparaison des Méthodes ({params.get('metric', 'diag_mean')})",
                                'description': "Bar chart comparant les méthodes sur une métrique",
                                'experiment_ids': experiment_ids,
                                'experiment_names': folder_names,
                                'image_base64': img_data,
                                'metadata': {
                                    'metric': params.get('metric', 'diag_mean'),
                                    'n_experiments': len(folder_names),
                                    'methods': list(self.analyzer.by_method.keys())
                                }
                            })
                            
                            plt.close(fig)
                        except Exception as e:
                            errors.append(f"compare_methods: {str(e)}")
                    
                    elif analysis_type == 'compare_rsd':
                        # Comparaison RSD
                        try:
                            fig = self.analyzer.plot_rsd_comparison(
                                folder_names=folder_names if folder_names else None,
                                n_steps=params.get('n_steps', n_steps),
                                figsize=figsize
                            )
                            
                            img_data = self._fig_to_base64(fig)
                            
                            images.append({
                                'id': str(uuid.uuid4()),
                                'type': 'compare_rsd',
                                'tab': 'rsd',
                                'title': "Comparaison RSD",
                                'description': "4 subplots: RSD linear, log, entropy, summary table",
                                'experiment_ids': experiment_ids,
                                'experiment_names': folder_names,
                                'image_base64': img_data,
                                'metadata': {
                                    'n_steps': n_steps,
                                    'n_experiments': len(folder_names)
                                }
                            })
                            
                            plt.close(fig)
                        except Exception as e:
                            errors.append(f"compare_rsd: {str(e)}")
                    
                    elif analysis_type == 'compare_mixing':
                        # Comparaison Mixing
                        try:
                            fig = self.analyzer.plot_mixing_comparison(
                                folder_names=folder_names if folder_names else None,
                                n_steps=params.get('n_steps', n_steps),
                                figsize=figsize
                            )
                            
                            img_data = self._fig_to_base64(fig)
                            
                            images.append({
                                'id': str(uuid.uuid4()),
                                'type': 'compare_mixing',
                                'tab': 'mixing',
                                'title': "Comparaison Mélange",
                                'description': "2 subplots: entropy growth, variance decay",
                                'experiment_ids': experiment_ids,
                                'experiment_names': folder_names,
                                'image_base64': img_data,
                                'metadata': {
                                    'n_steps': n_steps,
                                    'n_experiments': len(folder_names)
                                }
                            })
                            
                            plt.close(fig)
                        except Exception as e:
                            errors.append(f"compare_mixing: {str(e)}")
                    
                    elif analysis_type == 'compare_eigenvalues':
                        # Comparaison Eigenvalues
                        try:
                            fig = self.analyzer.plot_eigenvalues(
                                folder_names=folder_names if folder_names else None,
                                n_eigenvalues=params.get('n_eigenvalues', 20),
                                figsize=figsize
                            )
                            
                            img_data = self._fig_to_base64(fig)
                            
                            images.append({
                                'id': str(uuid.uuid4()),
                                'type': 'compare_eigenvalues',
                                'tab': 'eigenvalues',
                                'title': "Comparaison Valeurs Propres",
                                'description': "2 subplots: spectrum, 2nd eigenvalue",
                                'experiment_ids': experiment_ids,
                                'experiment_names': folder_names,
                                'image_base64': img_data,
                                'metadata': {
                                    'n_experiments': len(folder_names)
                                }
                            })
                            
                            plt.close(fig)
                        except Exception as e:
                            errors.append(f"compare_eigenvalues: {str(e)}")
                    
                    elif analysis_type == 'compare_within_method':
                        # Comparaison paramétrée pour chaque méthode
                        try:
                            for method in self.analyzer.by_method.keys():
                                fig = self.analyzer.compare_within_method(
                                    method=method,
                                    sweep_param=params.get('sweep_param', 'n_states'),
                                    figsize=figsize
                                )
                                
                                img_data = self._fig_to_base64(fig)
                                method_exps = [f for f in folder_names if self._get_method(f) == method]
                                method_ids = [experiment_ids[i] for i, f in enumerate(folder_names) if self._get_method(f) == method]
                                
                                images.append({
                                    'id': str(uuid.uuid4()),
                                    'type': 'compare_within_method',
                                    'tab': 'compare',
                                    'title': f"Comparaison Paramétrée - {method.upper()}",
                                    'description': "4 subplots: diag_mean, visited%, row_sum, population",
                                    'experiment_ids': method_ids,
                                    'experiment_names': method_exps,
                                    'image_base64': img_data,
                                    'metadata': {
                                        'method': method,
                                        'sweep_param': params.get('sweep_param', 'n_states')
                                    }
                                })
                                
                                plt.close(fig)
                        except Exception as e:
                            errors.append(f"compare_within_method: {str(e)}")
                    
                    elif analysis_type == 'rsd_vs_resolution':
                        # RSD vs Resolution pour chaque méthode
                        try:
                            for method in self.analyzer.by_method.keys():
                                if self.analyzer.by_method[method]:
                                    fig = self.analyzer.plot_rsd_vs_resolution(
                                        method=method,
                                        n_steps=params.get('n_steps', n_steps),
                                        figsize=figsize
                                    )
                                    
                                    img_data = self._fig_to_base64(fig)
                                    
                                    images.append({
                                        'id': str(uuid.uuid4()),
                                        'type': 'rsd_vs_resolution',
                                        'tab': 'rsd',
                                        'title': f"RSD vs Résolution - {method.upper()}",
                                        'description': "3 subplots: final RSD, mixing times, entropy",
                                        'experiment_ids': [],
                                        'experiment_names': [f for f in folder_names if self._get_method(f) == method],
                                        'image_base64': img_data,
                                        'metadata': {
                                            'method': method,
                                            'n_steps': n_steps
                                        }
                                    })
                                    
                                    plt.close(fig)
                        except Exception as e:
                            errors.append(f"rsd_vs_resolution: {str(e)}")
                
                except Exception as e:
                    errors.append(f"Erreur générale pour {analysis_type}: {str(e)}")
            
            # Organiser les images par onglet
            tabs_generated = self._organize_tabs(images)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'success',
                'images': images,
                'tabs_generated': tabs_generated,
                'generation_time': elapsed,
                'message': f"{len(images)} images générées avec succès en {elapsed:.1f}s",
                'errors': errors if errors else None
            }
        
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            return {
                'status': 'error',
                'images': [],
                'tabs_generated': [],
                'generation_time': elapsed,
                'message': f"Erreur lors de la génération: {str(e)}",
                'error': traceback.format_exc()
            }
    
    def _fig_to_base64(self, fig):
        """Convertit une figure matplotlib en base64."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def _extract_metadata_plot_experiment(self, folder_name):
        """Extrait les métadonnées pour plot_experiment."""
        try:
            data = self.analyzer.results[folder_name]
            M = data['matrix']
            method = data.get('method', 'unknown')
            
            rsd_data = self.analyzer.compute_rsd(folder_name, n_steps=200)
            
            return {
                'folder_name': folder_name,
                'method': method,
                'n_states': M.shape[0],
                'n_visited': (M.sum(axis=1) > 0).sum(),
                'diagonal_mean': float(np.diag(M).mean()),
                'rsd_initial': f"{rsd_data['rsd_percent'][0]:.1f}%",
                'rsd_final': f"{rsd_data['rsd_percent'][-1]:.1f}%",
                'mixing_time_50': int(rsd_data.get('mixing_time_50', 0)),
                'mixing_time_90': int(rsd_data.get('mixing_time_90', 0)),
                'spectral_gap': float(data.get('eigenvalue_2', 0))
            }
        except Exception as e:
            return {'folder_name': folder_name, 'error': str(e)}
    
    def _extract_metadata_plot_matrix(self, folder_name):
        """Extrait les métadonnées pour plot_matrix."""
        try:
            data = self.analyzer.results[folder_name]
            M = data['matrix']
            method = data.get('method', 'unknown')
            
            return {
                'folder_name': folder_name,
                'method': method,
                'n_states': M.shape[0],
                'diagonal_mean': float(np.diag(M).mean()),
                'diagonal_std': float(np.diag(M).std())
            }
        except Exception as e:
            return {'folder_name': folder_name, 'error': str(e)}
    
    def _get_method(self, folder_name):
        """Récupère la méthode d'une expérience."""
        try:
            return self.analyzer.results[folder_name].get('method', 'unknown')
        except:
            return 'unknown'
    
    def _organize_tabs(self, images):
        """Organise les images par onglets."""
        tabs_map = {}
        
        for img in images:
            tab_name = img['tab']
            if tab_name not in tabs_map:
                tabs_map[tab_name] = {
                    'tab_name': tab_name,
                    'image_ids': [],
                    'image_count': 0
                }
            
            tabs_map[tab_name]['image_ids'].append(img['id'])
            tabs_map[tab_name]['image_count'] += 1
        
        # Mapper les noms techniques aux labels affichés
        tab_labels = {
            'experiments': 'Expériences',
            'compare': 'Comparaison',
            'rsd': 'RSD',
            'mixing': 'Mélange',
            'eigenvalues': 'Valeurs propres',
            'dem_markov': 'DEM vs Markov'
        }
        
        tabs_generated = []
        for tab_name, tab_data in sorted(tabs_map.items()):
            tabs_generated.append({
                'tab_name': tab_name,
                'tab_label': tab_labels.get(tab_name, tab_name),
                'image_count': tab_data['image_count'],
                'image_ids': tab_data['image_ids']
            })
        
        return tabs_generated
